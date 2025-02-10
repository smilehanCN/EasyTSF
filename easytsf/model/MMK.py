import torch
import torch.nn as nn
import torch.nn.functional as F

from easytsf.layer.kanlayer import KANInterfaceV2
from easytsf.model.DenseRMoK import RevIN


class MoKLayer(nn.Module):
    def __init__(self, in_features, out_features, expert_config, res_con=False, with_bn=False, with_dropout=False):
        super(MoKLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_expert = len(expert_config)

        self.gate = nn.Linear(in_features, self.n_expert)
        self.softmax = nn.Softmax(dim=-1)
        self.experts = nn.ModuleList(
            [KANInterfaceV2(in_features, out_features, k[0], k[1]) for k in expert_config])

        self.res_con = res_con
        self.with_bn = with_bn
        self.with_dropout = with_dropout
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if self.with_dropout:
            self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        score = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.n_expert)], dim=-1)  # (BxN, Lo, E)
        # add BN here? Naming Expert Norm?
        y = torch.einsum("BLE,BE->BL", expert_outputs, score)

        if self.res_con:
            y = x + y
        if self.with_bn:
            y = self.bn(y)
        if self.with_dropout:
            y = self.dropout(y)
        return y


class MMK(nn.Module):
    """
    Multi-layer Mixture-of-KAN Network, MMK
    """

    def __init__(self, hist_len, pred_len, var_num, hidden_dim, layer_type, layer_hp, layer_num, use_norm=False):
        super(MMK, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.var_num = var_num
        self.use_norm = use_norm

        if layer_type == 'MoK':
            if layer_num == 1:
                self.model = MoKLayer(hist_len, pred_len, layer_hp)
            elif layer_num == 2:
                self.model = nn.Sequential(
                    MoKLayer(hist_len, hidden_dim, layer_hp),
                    MoKLayer(hidden_dim, pred_len, layer_hp)
                )
            elif layer_num == 3:
                self.model = nn.Sequential(
                    MoKLayer(hist_len, hidden_dim, layer_hp),
                    MoKLayer(hidden_dim, hidden_dim, layer_hp, res_con=use_norm, with_bn=use_norm, with_dropout=use_norm),
                    MoKLayer(hidden_dim, pred_len, layer_hp)
                )
            elif layer_num == 5:
                self.model = nn.Sequential(
                    MoKLayer(hist_len, hidden_dim, layer_hp),
                    MoKLayer(hidden_dim, hidden_dim, layer_hp, res_con=use_norm, with_bn=use_norm, with_dropout=use_norm),
                    MoKLayer(hidden_dim, hidden_dim, layer_hp, res_con=use_norm, with_bn=use_norm, with_dropout=use_norm),
                    MoKLayer(hidden_dim, hidden_dim, layer_hp, res_con=use_norm, with_bn=use_norm, with_dropout=use_norm),
                    MoKLayer(hidden_dim, pred_len, layer_hp)
                )
            else:
                raise NotImplementedError
        elif layer_type == 'Linear':
            if layer_num == 1:
                self.model = nn.Linear(hist_len, pred_len, bias=False)
            else:
                self.model = nn.Sequential(
                    nn.Linear(hist_len, hidden_dim, bias=False),
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.Linear(hidden_dim, pred_len, bias=False)
                )
        else:
            if layer_num == 1:
                self.model = KANInterfaceV2(hist_len, pred_len, layer_type=layer_type, hyperparam=layer_hp)
            else:
                self.model = nn.Sequential(
                    KANInterfaceV2(hist_len, hidden_dim, layer_type='TaylorKAN', hyperparam=layer_hp),
                    KANInterfaceV2(hidden_dim, hidden_dim, layer_type='TaylorKAN', hyperparam=layer_hp),
                    KANInterfaceV2(hidden_dim, pred_len, layer_type='TaylorKAN', hyperparam=layer_hp)
                )

        self.rev = RevIN(var_num, affine=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, var_x, marker_x):
        B, L, N = var_x.shape
        x = self.rev(var_x, 'norm')
        x = x.transpose(1, 2).reshape(B * N, L)
        prediction = self.model(x)
        prediction = prediction.reshape(B, N, -1).permute(0, 2, 1)
        prediction = self.rev(prediction, 'denorm')
        return prediction
