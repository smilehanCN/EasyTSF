import torch
import torch.nn as nn


def _normalize_feature_names(descriptions):
    return tuple(str(item).strip().lower() for item in (descriptions or ()))


class VarAwareAffine(nn.Module):
    def __init__(self, var_num, enabled=True):
        super().__init__()
        self.enabled = bool(enabled)
        if self.enabled:
            self.var_scale = nn.Parameter(torch.ones(int(var_num)))
            self.var_bias = nn.Parameter(torch.zeros(int(var_num)))

    def forward(self, x):
        if not self.enabled:
            return x
        broadcast_shape = (1, -1) + (1,) * (x.ndim - 2)
        scale = self.var_scale.view(*broadcast_shape)
        bias = self.var_bias.view(*broadcast_shape)
        return x * scale + bias


class PatchLinear(nn.Module):
    def __init__(self, in_dim, out_dim, var_num, use_var_aware=True):
        super().__init__()
        self.linear = nn.Linear(int(in_dim), int(out_dim), bias=False)
        self.var_affine = VarAwareAffine(var_num, enabled=use_var_aware)

    def forward(self, x):
        x = self.linear(x)
        return self.var_affine(x)


class EfficientTokenizer(nn.Module):
    def __init__(
        self,
        input_len,
        patch_size,
        patch_step,
        dim_group,
        var_num,
        use_var_aware,
        tokenizer_drop,
        use_tod,
        use_dow,
        tod_size,
        dow_size,
        tod_idx=None,
        dow_idx=None,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.patch_step = int(patch_step)
        self.dim_group = list(dim_group)
        self.use_tod = bool(use_tod)
        self.use_dow = bool(use_dow)
        self.tod_idx = tod_idx
        self.dow_idx = dow_idx
        self.patch_num = (int(input_len) - self.patch_size) // self.patch_step + 1

        self.tokenizer_group = nn.ModuleList(
            [
                PatchLinear(self.patch_size, dim, var_num, use_var_aware)
                for _, _, dim in self.dim_group
            ]
        )

        total_hidden_dim = sum(dim for _, _, dim in self.dim_group)
        self.tod_pe = None
        self.dow_pe = None
        if self.use_tod:
            self.tod_pe = nn.Parameter(torch.empty(int(tod_size), total_hidden_dim))
            nn.init.xavier_uniform_(self.tod_pe)
        if self.use_dow:
            self.dow_pe = nn.Parameter(torch.empty(int(dow_size), total_hidden_dim))
            nn.init.xavier_uniform_(self.dow_pe)

        self.dropout = nn.Dropout(float(tokenizer_drop))

    def _extract_marker_index(self, marker_x, feature_idx, embedding_size):
        marker = marker_x[:, :, feature_idx].unfold(dimension=-1, size=self.patch_size, step=self.patch_step)
        marker = marker[..., 0].long().to(marker_x.device)
        return marker % int(embedding_size)

    def forward(self, x, marker_x):
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_step)
        tod = None
        dow = None
        if self.use_tod:
            tod = self._extract_marker_index(marker_x, self.tod_idx, self.tod_pe.shape[0])
        if self.use_dow:
            dow = self._extract_marker_index(marker_x, self.dow_idx, self.dow_pe.shape[0])

        out = []
        start_dim = 0
        var_num = x.shape[1]
        for group_idx, (left_idx, right_idx, dim) in enumerate(self.dim_group):
            patch_group_tokens = self.tokenizer_group[group_idx](x[:, :, left_idx:right_idx, :])

            if self.use_tod:
                tod_pe = self.tod_pe[tod[:, left_idx:right_idx]].unsqueeze(1).repeat(1, var_num, 1, 1)
                patch_group_tokens = patch_group_tokens + tod_pe[..., start_dim:start_dim + dim]
            if self.use_dow:
                dow_pe = self.dow_pe[dow[:, left_idx:right_idx]].unsqueeze(1).repeat(1, var_num, 1, 1)
                patch_group_tokens = patch_group_tokens + dow_pe[..., start_dim:start_dim + dim]

            out.append(torch.flatten(patch_group_tokens, start_dim=2, end_dim=3))
            start_dim += dim

        return self.dropout(torch.cat(out, dim=-1))


class EncoderMLP(nn.Module):
    def __init__(self, in_dim, var_num, drop_rate=0.0, use_var_aware=True):
        super().__init__()
        hidden_dim = int(in_dim)
        self.up_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.down_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.var_affine = VarAwareAffine(var_num, enabled=use_var_aware)
        self.dropout = nn.Dropout(float(drop_rate))

    def forward(self, x):
        x = self.up_linear(x)
        x = self.act(x)
        x = self.down_linear(x)
        x = self.var_affine(x)
        return self.dropout(x)


class Model(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        var_num,
        freq=60,
        use_norm=True,
        patch_size=16,
        patch_step=None,
        init_dim=256,
        dim_assign_alg="step2",
        use_tod=None,
        use_dow=None,
        head_drop=0.1,
        encoder_drop=0.0,
        use_tokenizer_var_aware=True,
        use_encoder_var_aware=True,
        time_feature_descriptions=(),
    ):
        super().__init__()
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.var_num = int(var_num)
        self.freq = int(freq)
        self.use_norm = bool(use_norm)
        self.patch_size = int(patch_size)
        self.patch_step = self.patch_size if patch_step is None else int(patch_step)
        self.init_dim = int(init_dim)
        self.dim_assign_alg = str(dim_assign_alg)
        normalized_descriptions = _normalize_feature_names(time_feature_descriptions)
        self.use_tod = ("time of day" in normalized_descriptions) if use_tod is None else bool(use_tod)
        self.use_dow = ("day of week" in normalized_descriptions) if use_dow is None else bool(use_dow)
        self.head_drop = float(head_drop)
        self.encoder_drop = float(encoder_drop)
        self.use_tokenizer_var_aware = bool(use_tokenizer_var_aware)
        self.use_encoder_var_aware = bool(use_encoder_var_aware)
        self.tod_idx = normalized_descriptions.index("time of day") if "time of day" in normalized_descriptions else None
        self.dow_idx = normalized_descriptions.index("day of week") if "day of week" in normalized_descriptions else None

        patch_num = (self.hist_len - self.patch_size) // self.patch_step + 1
        dim_group, modified_dim = self._build_dim_group(patch_num, self.init_dim)
        tod_size = int((24 * 60) / self.freq)
        dow_size = 7

        self.tokenizer = EfficientTokenizer(
            input_len=self.hist_len,
            patch_size=self.patch_size,
            patch_step=self.patch_step,
            dim_group=dim_group,
            var_num=self.var_num,
            use_var_aware=self.use_tokenizer_var_aware,
            tokenizer_drop=self.head_drop,
            use_tod=self.use_tod,
            use_dow=self.use_dow,
            tod_size=tod_size,
            dow_size=dow_size,
            tod_idx=self.tod_idx,
            dow_idx=self.dow_idx,
        )
        self.encoder = EncoderMLP(
            modified_dim,
            self.var_num,
            self.encoder_drop,
            self.use_encoder_var_aware,
        )

        self.predictor = nn.Linear(modified_dim, self.pred_len)

    def forward(self, var_x, marker_x, marker_y):

        if self.use_norm:
            seq_mean = torch.mean(var_x, dim=1, keepdim=True)
            seq_var = torch.var(var_x, dim=1, keepdim=True, unbiased=False) + 1e-5
            var_x = (var_x - seq_mean) / torch.sqrt(seq_var)

        tokens = self.tokenizer(var_x.permute(0, 2, 1), marker_x)
        tokens = self.encoder(tokens)
        prediction = self.predictor(tokens).transpose(1, 2)

        if self.use_norm:
            prediction = prediction * torch.sqrt(seq_var) + seq_mean
        return prediction

    def _build_dim_group(self, patch_num, hidden_dim):
        if self.dim_assign_alg == "linear":
            assign_rate = [[idx, idx + 1, idx + 1] for idx in range(patch_num)]
        elif self.dim_assign_alg == "uniform":
            assign_rate = [[0, patch_num, 1]]
        elif self.dim_assign_alg == "uniform_independent_weight":
            assign_rate = [[idx, idx + 1, 1] for idx in range(patch_num)]
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
                [step_len, step_len * 2, 2],
                [step_len * 2, step_len * 3, 3],
                [step_len * 3, patch_num, 4],
            ]
        else:
            raise ValueError(
                "PCMLP only supports dim_assign_alg in {'linear', 'uniform', 'uniform_independent_weight', 'step', 'step2', 'step4'}, but received '{}'".format(
                    self.dim_assign_alg
                )
            )

        total_rate = sum(rate * (right_idx - left_idx) for left_idx, right_idx, rate in assign_rate)
        basic_dim = hidden_dim // total_rate
        if basic_dim <= 0:
            raise ValueError(
                "PCMLP requires init_dim {} to be large enough for dim_assign_alg '{}' with patch_num {}".format(
                    hidden_dim,
                    self.dim_assign_alg,
                    patch_num,
                )
            )
        dim_group = [[left_idx, right_idx, rate * basic_dim] for left_idx, right_idx, rate in assign_rate]
        tokenizer_output_dim = sum(dim * (right_idx - left_idx) for left_idx, right_idx, dim in dim_group)
        return dim_group, tokenizer_output_dim
