#%%
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self.weights = None
        self.biases = None

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter(
                '{}_weight_{}'.format(self._type, str(shape)), nn_param
            )
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter(
                '{}_biases_{}'.format(self._type, str(length)), biases
            )

        return self._biases_dict[length]


class DLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias, num_nodes):
        super(DLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_nodes = num_nodes
        self.rnn_units = hidden_size // num_nodes
        self._num_nodes = num_nodes
        self._memory_dim = 16
        self._bottleneck_dim = 4


        self._rnn_params = LayerParams(self, 'rnn_params')
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inputs, hx=None):

        if hx is None:
            hx = Variable(inputs.new_zeros(inputs.size(0), self.hidden_size))
            hx = (hx, hx)

        hx, cx = hx

        gates = self._fc_dynamic(inputs, hx, 4*self.rnn_units, bias_start=1.0, param_layer=self._rnn_params)
        gates = torch.reshape(gates, (-1, self._num_nodes, 4*self.rnn_units))
        # Get gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=2)
        input_gate = torch.reshape(input_gate, (-1, self._num_nodes * self.rnn_units))
        forget_gate = torch.reshape(forget_gate, (-1, self._num_nodes * self.rnn_units))
        cell_gate = torch.reshape(cell_gate, (-1, self._num_nodes * self.rnn_units))
        output_gate = torch.reshape(output_gate, (-1, self._num_nodes * self.rnn_units))
        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = cx * f_t + i_t * g_t

        hy = o_t * torch.tanh(cy)


        return (hy, cy)

    def _fc(self, inputs, state, output_size, bias_start=0.0, param_layer=None):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)

        input_size = inputs_and_state.shape[-1]
        weights = param_layer.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = param_layer.get_biases(output_size, bias_start)
        value = value + biases
        return value

    def _fc_dynamic(self, inputs, state, output_size, bias_start=0.0, param_layer=None, supports=None, full_input=None):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]

        memory = param_layer.get_weights((self._num_nodes, self._memory_dim))

        w1 = param_layer.get_weights((memory.shape[1], memory.shape[1]))
        b1 = param_layer.get_biases(memory.shape[1], bias_start)

        w2 = param_layer.get_weights((memory.shape[1], self._bottleneck_dim))
        b2 = param_layer.get_biases(self._bottleneck_dim, bias_start)

        w3 = param_layer.get_weights((self._bottleneck_dim, input_size * output_size))
        b3 = param_layer.get_biases(input_size * output_size, bias_start)

        mem = torch.tanh(torch.matmul(memory, w1) + b1)
        mem = torch.tanh(torch.matmul(mem, w2) + b2)

        weights = (torch.matmul(mem, w3) + b3).reshape([self._num_nodes, input_size, output_size])
        
        weights = weights.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        weights = weights.reshape([batch_size * self._num_nodes, input_size, output_size])

        b_out = param_layer.get_biases(output_size, bias_start)
        value = torch.sigmoid(torch.matmul(inputs_and_state.unsqueeze(1), weights).squeeze())
        value = value + b_out
        return value
    
#%%
class DLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_nodes, num_layers=1, bias=True, output_size=1):
        super(DLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.num_nodes = num_nodes
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(DLSTMCell(self.input_size,
                                            self.hidden_size,
                                            self.bias,
                                            self.num_nodes))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(DLSTMCell(self.hidden_size,
                                                self.hidden_size,
                                                self.bias,
                                                self.num_nodes))

    def forward(self, input, hx=None):

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        input[:, t, :],
                        (hidden[layer][0],hidden[layer][1])
                        )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                        )

                hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        if torch.cuda.is_available():
            out = torch.stack(outs, axis=1).cuda()
        else:
            out = torch.stack(outs, axis=1)

        return out, hidden
#%%
class HWNNLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, num_stock, K1=2, K2=2, approx=False, data=None):
        super(HWNNLayer, self).__init__()
        self.data = data
        self.input_size = input_size
        self.output_size = output_size
        self.num_stock = num_stock
        self.K1 = K1
        self.K2 = K2
        self.approx = approx
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_size, self.output_size))
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.num_stock))
        self.par = torch.nn.Parameter(torch.Tensor(self.K1 + self.K2))
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.99, 1.01)
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, features, snap_index, data):
        diagonal_weight_filter = torch.diag(self.diagonal_weight_filter)
        # Theta=self.data.Theta
        Theta = data.hypergraphsnapshot[snap_index]["Theta"]
        Theta_t = torch.transpose(Theta, 0, 1)

        if self.approx:
            poly = self.par[0] * torch.eye(self.num_stock)
            Theta_mul = torch.eye(self.num_stock)
            for ind in range(1, self.K1):
                Theta_mul = Theta_mul @ Theta
                poly = poly + self.par[ind] * Theta_mul

            poly_t = self.par[self.K1] * torch.eye(self.num_stock)
            Theta_mul = torch.eye(self.num_stock)
            for ind in range(self.K1 + 1, self.K1 + self.K2):
                Theta_mul = Theta_mul @ Theta_t  # 这里也可以使用Theta_transpose
                poly_t = poly_t + self.par[ind] * Theta_mul

            # poly=self.par[0]*torch.eye(self.num_stock)+self.par[1]*Theta+self.par[2]*Theta@Theta
            # poly_t = self.par[3] * torch.eye(self.num_stock) + self.par[4] * Theta_t + self.par[5] * Theta_t @ Theta_t
            # poly_t = self.par[3] * torch.eye(self.num_stock) + self.par[4] * Theta + self.par[
            #     5] * Theta @ Theta
            local_fea_1 = poly @ diagonal_weight_filter @ poly_t @ features @ self.weight_matrix
        else:
            wavelets = self.data.hypergraphsnapshot[snap_index]["wavelets"]
            wavelets_inverse = self.data.hypergraphsnapshot[snap_index]["wavelets_inv"]
            local_fea_1 = wavelets @ diagonal_weight_filter @ wavelets_inverse @ features @ self.weight_matrix

        localized_features = local_fea_1
        return localized_features
    
#%%
def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size(0), seq.size(1)
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.bool().bool().unsqueeze(0).expand(sz_b, -1, -1)

    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class TemporalAttention(nn.Module):
    ''' Temporal Attention module '''

    def __init__(self, n_head, rnn_unit, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(rnn_unit, n_head * d_k)
        self.w_ks = nn.Linear(rnn_unit, n_head * d_k)
        self.w_vs = nn.Linear(rnn_unit, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (rnn_unit + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (rnn_unit + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (rnn_unit + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(rnn_unit)

        self.fc = nn.Linear(n_head * d_v, rnn_unit)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)


        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        # output = output[:, -1, :]
        return output, attn