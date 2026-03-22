import torch
import torch.nn as nn


class VariableAwareLinear(nn.Module):
    def __init__(self, in_dim, out_dim, var_num, use_var_aware=True):
        super().__init__()
        self.use_var_aware = use_var_aware
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        if self.use_var_aware:
            self.var_bias = nn.Parameter(torch.zeros(1, var_num, 1, 1))
            self.var_scale = nn.Parameter(torch.ones(1, var_num, 1, 1))

    def forward(self, x):
        x = self.linear(x)
        if self.use_var_aware:
            x = x * self.var_scale + self.var_bias
        return x


class VarAwareMLPSimple(nn.Module):
    def __init__(self, in_dim, var_num, drop_rate=0.0, use_var_aware=True, ):
        super().__init__()
        hidden_dim = in_dim
        self.use_var_aware = use_var_aware

        self.up_linear = nn.Linear(in_dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        # self.inner_norm = nn.LayerNorm(hidden_dim)
        self.down_linear = nn.Linear(hidden_dim, in_dim, bias=False)
        if self.use_var_aware:
            self.var_bias = nn.Parameter(torch.zeros(1, var_num, 1))
            self.var_scale = nn.Parameter(torch.ones(1, var_num, 1))
        self.dropout = nn.Dropout(drop_rate)
        # self.out_norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        inputs = x

        x = self.up_linear(x)
        x = self.act(x)
        # x = self.inner_norm(x)
        x = self.down_linear(x)

        if self.use_var_aware:
            x = x * self.var_scale + self.var_bias
        # x = inputs + self.dropout(x)
        # x = self.out_norm(x)
        x = self.dropout(x)
        return x


class VarAwareMLP(nn.Module):
    def __init__(self, hidden_dim, var_num, drop_rate=0.0, use_var_aware=True):
        super().__init__()
        self.use_var_aware = use_var_aware

        self.up_linear = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.act = nn.GELU()
        self.inner_norm = nn.LayerNorm(hidden_dim * 2)
        self.down_linear = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        if self.use_var_aware:
            self.var_bias = nn.Parameter(torch.zeros(1, var_num, 1))
            self.var_scale = nn.Parameter(torch.ones(1, var_num, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        inputs = x

        x = self.up_linear(x)
        x = self.act(x)
        x = self.inner_norm(x)
        x = self.down_linear(x)

        if self.use_var_aware:
            x = x * self.var_scale + self.var_bias

        x = inputs + self.dropout(x)
        x = self.out_norm(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):

    def __init__(self, hist_len, pred_len, var_num, freq, use_norm, patch_size, patch_step, init_dim, dim_assign_alg,
                 use_tod=False, use_dow=False, head_drop=0.1, encoder_drop=0.0, use_tokenizer_var_aware=True,
                 use_encoder_var_aware=True, encoder_type=None):
        super(MLP, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.use_norm = use_norm
        self.dim_assign_alg = dim_assign_alg
        self.patch_size = patch_size
        patch_num = (hist_len - patch_size) // patch_step + 1

        dim_group, modified_dim = self._build_dim_group(patch_num, init_dim)
        tod_size = int((24 * 60) / freq)
        dow_size = 7
        self.tokenizer = nn.Linear(hist_len, modified_dim)
        if encoder_type == "mlp_1":
            self.encoder = VarAwareMLPSimple(modified_dim, var_num, encoder_drop, use_encoder_var_aware)
        elif encoder_type == "mlp_2":
            self.encoder = nn.Sequential(
                VarAwareMLP(modified_dim, var_num, encoder_drop, use_encoder_var_aware),
                VarAwareMLP(modified_dim, var_num, encoder_drop, use_encoder_var_aware),
            )
        else:
            raise NotImplementedError

        self.predictor = nn.Linear(modified_dim, pred_len)

    def forward(self, var_x, marker_x):
        if self.use_norm:
            seq_mean = torch.mean(var_x, dim=1, keepdim=True)
            seq_var = torch.var(var_x, dim=1, keepdim=True) + 1e-5
            var_x = (var_x - seq_mean) / torch.sqrt(seq_var)

        var_x = var_x.permute(0, 2, 1)  # (B, L, N) -> (B, N, L)

        tokens = self.tokenizer(var_x)

        tokens = self.encoder(tokens)

        pred = self.predictor(tokens).transpose(1, 2)

        if self.use_norm:
            pred = pred * torch.sqrt(seq_var) + seq_mean

        return pred

    def _build_dim_group(self, patch_num, hidden_dim):
        if self.dim_assign_alg == "linear":
            assign_rate = [[i, i + 1, i + 1] for i in range(patch_num)]
        elif self.dim_assign_alg == "uniform":
            assign_rate = [[0, patch_num, 1]]
        elif self.dim_assign_alg == "uniform_independent_weight":
            assign_rate = [[i, i + 1, 1] for i in range(patch_num)]
        elif self.dim_assign_alg == "step":
            head_tail_step = int(patch_num // 3)
            mid_step = patch_num - head_tail_step * 2
            assign_rate = [
                [0, head_tail_step, 1],
                [head_tail_step, head_tail_step + mid_step, 2],
                [head_tail_step + mid_step, patch_num, 3],
            ]
        elif self.dim_assign_alg == "step2":
            head_tail_step = int(patch_num // 2)
            assign_rate = [
                [0, head_tail_step, 1],
                [head_tail_step, patch_num, 2],
            ]
        elif self.dim_assign_alg == "step4":
            step_len = int(patch_num // 4)
            assign_rate = [
                [0, step_len, 1],
                [step_len, step_len*2, 2],
                [step_len*2, step_len * 3, 3],
                [step_len * 3, patch_num, 4],
            ]
        else:
            raise NotImplementedError

        total_rate = sum([rate * (ri - li) for li, ri, rate in assign_rate])
        basic_dim = hidden_dim // total_rate
        dim_group = [[li, ri, rate * basic_dim] for li, ri, rate in assign_rate]
        tokenizer_output_dim = sum([dim * (ri - li) for li, ri, dim in dim_group])

        print('basic_dim: {}, tokenizer_output_dim: {}'.format(basic_dim, tokenizer_output_dim))
        return dim_group, tokenizer_output_dim
