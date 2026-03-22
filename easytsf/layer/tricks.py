import torch
import torch.nn as nn


class EfficientTokenizer(nn.Module):

    def __init__(self, input_len, patch_size, patch_step, dim_group, var_num, use_var_transform, tokenizer_drop):
        super(EfficientTokenizer, self).__init__()
        assert (input_len - patch_size) % patch_step == 0
        self.patch_size = patch_size
        self.patch_step = patch_step
        self.dim_group = dim_group # example: [[left, right, dim], [0, 4, 32], [4, 9, 64], [9, 13, 128]]
        self.patch_num = (input_len - patch_size) // patch_step + 1

        self.tokenizer_group = nn.ModuleList(
            [VariableAwareLinear(patch_size, dim, var_num, use_var_transform) for _, _, dim in self.dim_group])
        self.dropout = nn.Dropout(tokenizer_drop)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_step)  # (B, N, L) -> (B, N, P, S)

        out = []
        for i, (li, ri, _) in enumerate(self.dim_group):
            patch_group_tokens = self.tokenizer_group[i](x[:, :, li:ri, :]) # (B, N, P, S) -> (B, N, P, di)
            out.append(torch.flatten(patch_group_tokens, start_dim=2, end_dim=3))
        out = torch.cat(out, dim=-1) # (B, N, D)

        return self.dropout(out)


class VariableAwareLinear(nn.Module):
    def __init__(self, in_dim, out_dim, var_num, use_var_transform=True):
        super().__init__()
        self.use_var_transform = use_var_transform
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        if self.use_var_transform:
            self.var_bias = nn.Parameter(torch.zeros(1, var_num, 1))
            self.var_scale = nn.Parameter(torch.ones(1, var_num, 1))

    def forward(self, x):
        x = self.linear(x)
        if self.use_var_transform:
            x = x * self.var_scale + self.var_bias
        return x
