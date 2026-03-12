#%%
import os
import time 
from copy import deepcopy
import wandb

import numpy as np
import torch
import torch.nn as nn

from modules.utils import Scheduler, EarlyStopping

from modules.cramer import cramer_wold_distance

from modules.dilate_loss import dilate_loss
from modules.dilate_loss_cuda import DilateLossCUDA
# from modules.dilate_loss_cache import dilate_loss
from modules.soft_dtw_cuda import SoftDTW
from modules.dtw_cuda import DTW
from modules.dpp_loss import dpp_loss
from modules.fft_ot import cal_wasserstein
from modules.fourier_koopman import fourier_loss
from modules.metrics import metric
from modules.metrics_torch import create_metric_collector, metric_torch
from modules.ot_dist import *
from modules.polynomial import (
    chebyshev_torch, 
    hermite_torch, 
    laguerre_torch, 
    leg_torch, 
    pca_torch, 
    Basis_Cache, 
    ica_torch, 
    robust_ica_torch, 
    robust_pca_torch, 
    svd_torch, 
    random_torch, 
    Random_Cache,
    fa_torch)
#%%
def initialize_cache(args, train_dataset):
    cache = None
    if args.auxi_mode == 'basis':
        if args.auxi_type == 'random':
            cache = Random_Cache(
                rank_ratio=args.rank_ratio, 
                pca_dim=args.pca_dim,
                pred_len=args.pred_len, 
                enc_in=args.enc_in,
                device=args.device
            )
        elif args.auxi_type == 'fa':
            cache = Basis_Cache(train_dataset.fa_components, train_dataset.initializer, mean=train_dataset.fa_mean, device=args.device)
        elif args.auxi_type == 'pca':
            cache = Basis_Cache(train_dataset.pca_components, train_dataset.initializer, weights=train_dataset.weights, device=args.device)
        elif args.auxi_type == 'robustpca':
            cache = Basis_Cache(train_dataset.pca_components, train_dataset.initializer, mean=train_dataset.rpca_mean, device=args.device)
        elif args.auxi_type == 'svd':
            cache = Basis_Cache(train_dataset.svd_components, train_dataset.initializer, device=args.device)
        elif args.auxi_type == 'ica':
            cache = Basis_Cache(train_dataset.ica_components, train_dataset.initializer, mean=train_dataset.ica_mean, whitening=train_dataset.whitening, device=args.device)
        elif args.auxi_type == 'robustica':
            cache = Basis_Cache(train_dataset.ica_components, train_dataset.initializer, device=args.device)
    return cache
    
def train_function(model, args, train_dataset, train_dataloader, valid_dataloader, optimizer, setting):
    cache = initialize_cache(args, train_dataset)
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)
    if args.add_noise and args.noise_amp > 0:
        seq_len = args.pred_len
        cutoff_freq_percentage = args.noise_freq_percentage
        cutoff_freq = int((seq_len // 2 + 1) * cutoff_freq_percentage)
        if args.auxi_mode == "rfft":
            low_pass_mask = torch.ones(seq_len // 2 + 1)
            low_pass_mask[-cutoff_freq:] = 0.
        else:
            raise NotImplementedError
        mask = low_pass_mask.reshape(1, -1, 1).to(args.device)
    else:
        mask = None

    time_now = time.time()

    train_steps = len(train_dataloader)
    model_state_last_effective = None
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    if args.auxi_mode == 'fourier_koopman':
        freqs = nn.Parameter(torch.tensor(train_dataset.freqs, device=args.device, dtype=torch.float32)) #### freqs checking !
        optimizer.add_param_group({'params': freqs, 'lr': args.learning_rate})
        
    scheduler = Scheduler(optimizer, args, train_steps)
    criterion = nn.MSELoss()
    if args.auxi_mode == 'soft_dtw':
        assert args.device != 'cpu' and args.device != torch.device('cpu'), "SoftDTW only supports GPU"
        sdtw = SoftDTW(use_cuda=True, gamma=0.1)
    elif args.auxi_mode == 'dtw':
        assert args.device != 'cpu' and args.device != torch.device('cpu'), "DTW only supports GPU"
        dtw = DTW(use_cuda=True, bandwidth=0.1)
    elif args.auxi_mode == 'dilate_cuda':
        assert args.device != 'cpu' and args.device != torch.device('cpu'), "DILATE only supports GPU"
        dilate_cuda = DilateLossCUDA(alpha=args.dilate_alpha, gamma=args.gamma, bandwidth=0)

    for epoch in range(args.train_epochs):
        logs = {
            'loss': [], 
            'recon': [],
            'auxi': [],
            'valid_loss':[],
            'cost_time':[],
        }
        has_nan_in_epoch = False
        train_loss = []

        lr_cur = scheduler.get_lr()
        lr_cur = lr_cur[0] if isinstance(lr_cur, list) else lr_cur

        model.train()
        epoch_time = time.time()
        
        iter_count = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_dataloader):
            iter_count += 1
            optimizer.zero_grad()

            outputs, batch_y, attn = model(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
            
            """1. reconstruction loss (MSE)"""
            loss = 0
            loss_ = []
            if args.rec_lambda:
                loss_rec = criterion(outputs, batch_y)
                loss += args.rec_lambda * loss_rec
            else:
                loss_rec = torch.tensor(1e4)

            loss_.append(('recon', loss_rec))
            
            if args.l1_weight and attn:
                loss += args.l1_weight * attn[0]
            
            """2. Distributional loss"""
            if args.auxi_lambda:
                if args.joint_forecast:  # joint distribution forecasting
                    outputs = torch.concat((batch_x.to(outputs.device), outputs), dim=1)  # [B, S+P, D]
                    batch_y = torch.concat((batch_x.to(batch_y.device), batch_y), dim=1)  # [B, S+P, D]

                if args.auxi_mode == "fft":
                    loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)  # shape: [B, P, D]

                elif args.auxi_mode == "rfft":
                    if args.auxi_type == 'complex':
                        loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)  # shape: [B, P//2+1, D]
                    elif args.auxi_type == 'complex-phase':
                        loss_auxi = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()  
                    elif args.auxi_type == 'complex-mag-phase':
                        loss_auxi_mag = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs()
                        loss_auxi_phase = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                        loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])  # shape: [2, B, P//2+1, D]
                    elif args.auxi_type == 'phase':
                        loss_auxi = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()  # shape: [B, P//2+1, D]
                    elif args.auxi_type == 'mag':
                        loss_auxi = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()  # shape: [B, P//2+1, D]
                    elif args.auxi_type == 'mag-phase':
                        loss_auxi_mag = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
                        loss_auxi_phase = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
                        loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])  # shape: [2, B, P//2+1, D]
                    else:
                        raise NotImplementedError

                elif args.auxi_mode == "rfft-D":
                    loss_auxi = torch.fft.rfft(outputs, dim=-1) - torch.fft.rfft(batch_y, dim=-1)  # shape: [B, P, D//2+1]

                elif args.auxi_mode == "rfft-2D":
                    loss_auxi = torch.fft.rfft2(outputs) - torch.fft.rfft2(batch_y)  # shape: [B, P, D//2+1]

                elif args.auxi_mode == "basis":
                    kwargs = {'degree': args.leg_degree, 'device': args.device}
                    if args.auxi_type == "legendre":
                        loss_auxi = leg_torch(outputs, **kwargs) - leg_torch(batch_y, **kwargs)  # shape: [B*D, degree+1]
                    elif args.auxi_type == "chebyshev":
                        loss_auxi = chebyshev_torch(outputs, **kwargs) - chebyshev_torch(batch_y, **kwargs)
                    elif args.auxi_type == "hermite":
                        loss_auxi = hermite_torch(outputs, **kwargs) - hermite_torch(batch_y, **kwargs)
                    elif args.auxi_type == "laguerre":
                        loss_auxi = laguerre_torch(outputs, **kwargs) - laguerre_torch(batch_y, **kwargs)
                    elif args.auxi_type == "random":
                        kwargs = {'pca_dim': args.pca_dim, 'random_cache': cache, 'device': args.device}
                        loss_auxi = random_torch(outputs, **kwargs) - random_torch(batch_y, **kwargs)
                    elif args.auxi_type == "fa":
                        kwargs = {'pca_dim': args.pca_dim, 'fa_cache': cache, 'reinit': args.reinit, 'device': args.device}
                        loss_auxi = fa_torch(outputs, **kwargs) - fa_torch(batch_y, **kwargs)
                    elif args.auxi_type == "pca":
                        kwargs = {
                            'pca_dim': args.pca_dim, 'pca_cache': cache, 'use_weights': args.use_weights, 
                            'reinit': args.reinit, 'device': args.device
                        }
                        # if prof is not None:
                            # with profiler.record_function("auxi_loss_forward_pass"):
                            # loss_auxi = pca_torch(outputs, **kwargs) - pca_torch(batch_y, **kwargs)
                        # else:
                        loss_auxi = pca_torch(outputs, **kwargs) - pca_torch(batch_y, **kwargs)
                    
                    elif args.auxi_type == "robustpca":
                        kwargs = {'pca_dim': args.pca_dim, 'pca_cache': cache, 'reinit': args.reinit, 'device': args.device}
                        loss_auxi = robust_pca_torch(outputs, **kwargs) - robust_pca_torch(batch_y, **kwargs)
                    elif args.auxi_type == "svd":
                        kwargs = {'pca_dim': args.pca_dim, 'svd_cache': cache, 'reinit': args.reinit, 'device': args.device}
                        loss_auxi = svd_torch(outputs, **kwargs) - svd_torch(batch_y, **kwargs)
                    elif args.auxi_type == "ica":
                        kwargs = {'pca_dim': args.pca_dim, 'ica_cache': cache, 'reinit': args.reinit, 'device': args.device}
                        loss_auxi = ica_torch(outputs, **kwargs) - ica_torch(batch_y, **kwargs)
                    elif args.auxi_type == "robustica":
                        kwargs = {'pca_dim': args.pca_dim, 'ica_cache': cache, 'reinit': args.reinit, 'device': args.device}
                        loss_auxi = robust_ica_torch(outputs, **kwargs) - robust_ica_torch(batch_y, **kwargs)
                    else:
                        raise NotImplementedError

                elif args.auxi_mode == "ot":
                    kwargs = {'dist_scale': args.dist_scale, 'device': args.device}
                    if args.auxi_type == "emd1d_t":
                        loss_auxi = emd_loss_1d_batched_align_t(outputs, batch_y, **kwargs)
                    elif args.auxi_type == "emd1d_h":
                        loss_auxi = emd_loss_1d_batched_align_h(outputs, batch_y, **kwargs)
                    elif args.auxi_type == "emd1d_all":
                        loss_auxi = emd_loss_1d_batched_align_all(outputs, batch_y, **kwargs)

                    elif args.auxi_type == "emd2d_h":
                        loss_auxi = emd_loss_2d_batched_align_h(outputs, batch_y, **kwargs)
                    elif args.auxi_type == "emd2d_t":
                        loss_auxi = emd_loss_2d_batched_align_t(outputs, batch_y, **kwargs)
                    elif args.auxi_type == "emd2d_all":
                        loss_auxi = emd_loss_2d_batched_align_all(outputs, batch_y, **kwargs)

                    elif args.auxi_type == "emd1d_h_learn_proj":
                        outputs_proj = model.project(outputs)
                        batch_y_proj = model.project(batch_y)
                        loss_auxi = emd_loss_1d_batched_align_h(outputs_proj, batch_y_proj, **kwargs)
                    elif args.auxi_type == "emd1d_t_learn_proj":
                        outputs_proj = model.project(outputs)
                        batch_y_proj = model.project(batch_y)
                        loss_auxi = emd_loss_1d_batched_align_t(outputs_proj, batch_y_proj, **kwargs)
                    elif args.auxi_type == "emd1d_all_learn_proj":
                        outputs_proj = model.project(outputs)
                        batch_y_proj = model.project(batch_y)
                        loss_auxi = emd_loss_1d_batched_align_all(outputs_proj, batch_y_proj, **kwargs)

                    elif args.auxi_type == "emd1d_h_pca_proj":
                        n_feats, rank_ratio = args.c_out, args.rank_ratio
                        low_rank = int(n_feats * rank_ratio)
                        outputs_proj = torch.matmul(outputs, torch.pca_lowrank(outputs.reshape(-1, n_feats), low_rank)[-1])
                        batch_y_proj = torch.matmul(batch_y, torch.pca_lowrank(batch_y.reshape(-1, n_feats), low_rank)[-1])
                        loss_auxi = emd_loss_1d_batched_align_h(outputs_proj, batch_y_proj, **kwargs)
                    elif args.auxi_type == "emd1d_t_pca_proj":
                        n_feats, rank_ratio = args.c_out, args.rank_ratio
                        low_rank = int(n_feats * rank_ratio)
                        outputs_proj = torch.matmul(outputs, torch.pca_lowrank(outputs.reshape(-1, n_feats), low_rank)[-1])
                        batch_y_proj = torch.matmul(batch_y, torch.pca_lowrank(batch_y.reshape(-1, n_feats), low_rank)[-1])
                        loss_auxi = emd_loss_1d_batched_align_t(outputs_proj, batch_y_proj, **kwargs)
                    elif args.auxi_type == "emd1d_all_pca_proj":
                        n_feats, rank_ratio = args.c_out, args.rank_ratio
                        low_rank = int(n_feats * rank_ratio)
                        outputs_proj = torch.matmul(outputs, torch.pca_lowrank(outputs.reshape(-1, n_feats), low_rank)[-1])
                        batch_y_proj = torch.matmul(batch_y, torch.pca_lowrank(batch_y.reshape(-1, n_feats), low_rank)[-1])
                        loss_auxi = emd_loss_1d_batched_align_all(outputs_proj, batch_y_proj, **kwargs)

                    else:
                        raise NotImplementedError

                elif args.auxi_mode == "fft_ot":
                    loss_auxi = cal_wasserstein(
                        outputs, batch_y, args.distance, ot_type=args.ot_type, normalize=args.normalize, 
                        mask_factor=args.mask_factor, reg_sk=args.reg_sk, stopThr=args.stopThr, numItermax=args.numItermax, 
                        var_weight=args.var_weight, mean_weight=args.mean_weight
                    )

                elif args.auxi_mode == "fourier_koopman":
                    loss_auxi = fourier_loss(outputs, batch_y, freqs, device=args.device)

                elif args.auxi_mode == "dilate":
                    loss_auxi, _, _ = dilate_loss(outputs, batch_y, args.alpha, args.gamma, args.device)

                elif args.auxi_mode == "dpp":
                    loss_auxi = dpp_loss(outputs, batch_y, args.alpha, args.gamma, args.device)

                elif args.auxi_mode == "soft_dtw":
                    loss_auxi = sdtw(outputs, batch_y)

                elif args.auxi_mode == "dtw":
                    loss_auxi = dtw(outputs, batch_y)[0].mean()

                elif args.auxi_mode == "dilate_cuda":
                    loss_auxi = dilate_cuda(outputs, batch_y)
                    
                elif args.auxi_mode == 'cramer':
                    B, T, D = outputs.shape
                    pred = outputs.reshape(B, T * D)
                    true = batch_y.reshape(B, T * D)
                    loss_auxi = cramer_wold_distance(pred, true)
                    
                elif args.auxi_mode == 'cramer_per_dim':
                    loss = 0.0
                    for d in range(outputs.shape[-1]):
                        loss += cramer_wold_distance(
                            outputs[..., d],   # (B, T)
                            batch_y[..., d]    # (B, T)
                        )
                    loss_auxi = loss / outputs.shape[-1]
                    
                elif args.auxi_mode == 'cramer_fft':
                    B, T, D = outputs.shape
                    Xp = torch.fft.rfft(outputs, dim=1).abs()
                    Xt = torch.fft.rfft(batch_y, dim=1).abs()

                    Xp = Xp.reshape(B, -1)
                    Xt = Xt.reshape(B, -1)

                    loss_auxi = cramer_wold_distance(Xp, Xt)

                else:
                    raise NotImplementedError

                if mask is not None:
                    loss_auxi *= mask

                if args.auxi_loss == "MAE":
                    loss_auxi = loss_auxi.abs().mean() if args.module_first else loss_auxi.mean().abs()  # check the dim of fft
                elif args.auxi_loss == "MSE":
                    loss_auxi = (loss_auxi.abs()**2).mean() if args.module_first else (loss_auxi**2).mean().abs()
                elif args.auxi_loss == "None":
                    pass
                else:
                    raise NotImplementedError
                
                loss_.append(('auxi', loss_auxi))
                
                loss += args.auxi_lambda * loss_auxi
                
            else:
                loss_auxi = torch.tensor(1e4)

            loss_.append(('auxi', loss_auxi))
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss is NaN or Inf, skipping epoch {epoch} step {i}")
                has_nan_in_epoch = True
                continue
            
            train_loss.append(loss.item())
        
            if (i + 1) % 100 == 0:
                print(
                    "\titers: {}, epoch: {} | loss_rec: {:.7f}, loss_auxi: {:.7f}, loss: {:.7f}".format(
                        i + 1, epoch, loss_rec.item(), loss_auxi.item(), loss.item()
                    )
                )
                cost_time = time.time() - time_now
                speed = cost_time / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; cost time: {:.4f}s; left time: {:.4f}s'.format(speed, cost_time, left_time))
                iter_count = 0
                time_now = time.time()
                model_state_last_effective = deepcopy(model.state_dict())  # save the last effective model state dict

            loss.backward()
            optimizer.step()
            
            loss_.append(('loss', loss))
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()]

            if args.lradj in ['TST']:
                scheduler.step(verbose=(i + 1 == train_steps))

        if model_state_last_effective is not None and has_nan_in_epoch:
            model.load_state_dict(model_state_last_effective)
        
        """validation"""
        valid_loss = validation(model, valid_dataloader, criterion)
        logs['valid_loss'] = logs.get('valid_loss') + [valid_loss]
        
        cost_time = time.time() - epoch_time
        logs['cost_time'] = logs.get('cost_time') + [cost_time]
        
        """print"""
        print_input = f"Epoch [{epoch+1:03d}/{args.train_epochs}]"
        print_input += "".join(
            [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
        )

        print(print_input)
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    
        early_stopping(valid_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if args.lradj not in ['TST']:
            scheduler.step(valid_loss, epoch)

    best_model_path = os.path.join(path, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_model_path))

    return model

#%%
def validation(model, valid_dataloader, criterion):
    total_loss = []
    model.eval()

    eval_time = time.time()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(valid_dataloader):
            outputs, batch_y, _ = model(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)

            total_loss.append(loss)

    print('Validation cost time: {}'.format(time.time() - eval_time))
    # total_loss = np.average(total_loss)
    total_loss = torch.mean(torch.stack(total_loss)).item()  # average loss
    model.train()
    return total_loss