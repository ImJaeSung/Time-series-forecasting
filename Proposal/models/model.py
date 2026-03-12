#%%
import os 
import yaml

import numpy as np 

import torch
import torch.nn as nn
from modules.metrics_torch import create_metric_collector, metric_torch

from modules.utils import visual
#%%
class CWDist(nn.Module):
    # def _build_model(self):
    def __init__(self, args, base):
        super(CWDist, self).__init__()
        # model = self.model_dict[self.args.model].Model(self.args).float()
        self.args = args
        self.device = args.device
        self.base = base
        # pretrain_model_path = args.pretrain_model_path
        
        # if pretrain_model_path and os.path.exists(pretrain_model_path):
        #     print(f'Loading pretrained model from {pretrain_model_path}')
        #     state_dict = torch.load(pretrain_model_path)
        #     model.load_state_dict(state_dict, strict=False)

        if args.use_multi_gpu and args.use_gpu:
            self.base = nn.DataParallel(self.base, device_ids=args.device_ids)

    def forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if ('PEMS' in self.args.data or 'SRU' in self.args.data) and self.args.model not in ['TiDE']:
            batch_x_mark = None
            batch_y_mark = None
        else:
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        model_args = [batch_x, batch_x_mark, dec_inp, batch_y_mark]
        if self.args.model in ['CycleNet', 'TQNet']:
            model_args.append(batch_cycle)
        if self.args.output_attention:
            outputs, attn = self.base(*model_args)
        else:
            outputs, attn = self.base(*model_args), None
            
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        
        return outputs, batch_y, attn
    
    
    def test(self, test_dataset, test_dataloader, setting, prof=None):
        # test_data, test_loader = self._get_data(flag='test')
        # if test:
        #     print('loading model')
        #     ckpt_dir = os.path.join(self.args.checkpoints, setting)
        #     self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))

        inputs, preds, trues = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        os.makedirs(folder_path, exist_ok=True)

        self.eval()
        # metric_collector = create_metric_collector(device=self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_dataloader):
                batch_x = batch_x.to(self.args.device)
                outputs, batch_y, _ = self(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                batch_x = batch_x.detach()
                outputs = outputs.detach()
                batch_y = batch_y.detach()

                if test_dataset.scale and self.args.inverse:
                    batch_x = batch_x.cpu().numpy()
                    in_shape = batch_x.shape
                    batch_x = test_dataset.inverse_transform(batch_x.reshape(-1, in_shape[-1])).reshape(in_shape)
                    batch_x = torch.from_numpy(batch_x).float().to(self.device)

                    outputs = outputs.cpu().numpy()
                    batch_y = batch_y.cpu().numpy()
                    out_shape = outputs.shape
                    outputs = test_dataset.inverse_transform(outputs.reshape(-1, out_shape[-1])).reshape(out_shape)
                    batch_y = test_dataset.inverse_transform(batch_y.reshape(-1, out_shape[-1])).reshape(out_shape)
                    outputs = torch.from_numpy(outputs).float().to(self.device)
                    batch_y = torch.from_numpy(batch_y).float().to(self.device)

                inputs.append(batch_x.cpu())
                preds.append(outputs.cpu())
                trues.append(batch_y.cpu())

                if i % 20 == 0 and self.args.output_vis:
                    gt = np.concatenate((batch_x[0, :, -1].cpu().numpy(), batch_y[0, :, -1].cpu().numpy()), axis=0)
                    pd = np.concatenate((batch_x[0, :, -1].cpu().numpy(), outputs[0, :, -1].cpu().numpy()), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                if prof is not None:
                    prof.step()

        inputs = torch.cat(inputs, dim=0)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        print('test shape:', preds.shape, trues.shape)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        # if self.writer is None:
        #     self.writer = self._create_writer(res_path)

        # m = metric_collector.compute()
        # mae, mse, rmse, mape, mspe, mre = m["mae"], m["mse"], m["rmse"], m["mape"], m["mspe"], m["mre"]
        mae, mse, rmse, mape, mspe, mre = metric_torch(preds, trues)
        print('{}\t| mse:{}, mae:{}'.format(self.args.pred_len, mse, mae))

        # self.writer.add_scalar(f'{self.pred_len}/test/mae', mae, self.epoch)
        # self.writer.add_scalar(f'{self.pred_len}/test/mse', mse, self.epoch)
        # self.writer.add_scalar(f'{self.pred_len}/test/rmse', rmse, self.epoch)
        # self.writer.add_scalar(f'{self.pred_len}/test/mape', mape, self.epoch)
        # self.writer.add_scalar(f'{self.pred_len}/test/mspe', mspe, self.epoch)
        # self.writer.add_scalar(f'{self.pred_len}/test/mre', mre, self.epoch)
        # self.writer.close()

        log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
        f = open(log_path, 'a')
        f.write(setting + "\n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, mre]))

        if self.args.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs.cpu().numpy())
            np.save(os.path.join(res_path, 'pred.npy'), preds.cpu().numpy())
            np.save(os.path.join(res_path, 'true.npy'), trues.cpu().numpy())
            if self.args.auxi_mode == 'basis' and self.args.auxi_type == 'pca':
                train_data, _ = self._get_data(flag='train')
                pca_components = train_data.pca_components
                np.save(os.path.join(res_path, 'pca_components.npy'), pca_components)

        if not os.path.exists(os.path.join(res_path, 'config.yaml')):
            print('save configs')
            args_dict = vars(self.args)
            with open(os.path.join(res_path, 'config.yaml'), 'w') as yaml_file:
                yaml.dump(args_dict, yaml_file, default_flow_style=False)

        return mae, mse, rmse, mape, mspe, mre