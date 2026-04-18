import math

import torch
import torch.nn as nn


def calculate_orthogonal_loss(matrix):
    gram_matrix = torch.matmul(matrix.transpose(-2, -1), matrix)
    diagonal = torch.diag_embed(torch.diagonal(gram_matrix, dim1=-2, dim2=-1))
    off_diagonal = gram_matrix - diagonal
    return torch.norm(off_diagonal, dim=(-2, -1)).mean()


class Model(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        var_num,
        period_len,
        basis_num,
        use_period_norm=True,
        use_orthogonal=True,
        individual=False,
    ):
        super().__init__()
        self.seq_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(var_num)
        self.period_len = int(period_len)
        self.basis_num = int(basis_num)
        self.use_period_norm = bool(use_period_norm)
        self.use_orthogonal = bool(use_orthogonal)
        self.individual = bool(individual)

        if self.seq_len <= 0:
            raise ValueError("TimeBase requires hist_len > 0, but received {}".format(self.seq_len))
        if self.pred_len <= 0:
            raise ValueError("TimeBase requires pred_len > 0, but received {}".format(self.pred_len))
        if self.enc_in <= 0:
            raise ValueError("TimeBase requires var_num > 0, but received {}".format(self.enc_in))
        if self.period_len <= 0:
            raise ValueError("TimeBase requires period_len > 0, but received {}".format(self.period_len))
        if self.basis_num <= 0:
            raise ValueError("TimeBase requires basis_num > 0, but received {}".format(self.basis_num))

        self.seg_num_x = math.ceil(self.seq_len / self.period_len)
        self.seg_num_y = math.ceil(self.pred_len / self.period_len)
        self.pad_seq_len = self.seg_num_x * self.period_len - self.seq_len
        self._aux_loss = None

        if self.individual:
            self.ts2basis = nn.ModuleList(
                [nn.Linear(self.seg_num_x, self.basis_num) for _ in range(self.enc_in)]
            )
            self.basis2ts = nn.ModuleList(
                [nn.Linear(self.basis_num, self.seg_num_y) for _ in range(self.enc_in)]
            )
        else:
            self.ts2basis = nn.Linear(self.seg_num_x, self.basis_num)
            self.basis2ts = nn.Linear(self.basis_num, self.seg_num_y)

    def _pad_input(self, x):
        if self.pad_seq_len <= 0:
            return x

        pad_start = (self.seg_num_x - 1) * self.period_len
        padding = x[:, :, max(0, pad_start - self.pad_seq_len):pad_start]
        if padding.shape[-1] == 0:
            padding = x
        if padding.shape[-1] < self.pad_seq_len:
            repeat_count = math.ceil(self.pad_seq_len / padding.shape[-1])
            padding = padding.repeat(1, 1, repeat_count)
        padding = padding[:, :, -self.pad_seq_len:]
        return torch.cat([x, padding], dim=-1)

    def _normalize_input(self, x, batch_size, var_num):
        if self.use_period_norm:
            period_mean = torch.mean(x, dim=-1, keepdim=True)
            return x - period_mean, {"period_mean": period_mean}

        x = x.reshape(batch_size, var_num, -1)
        seq_mean = torch.mean(x, dim=-1, keepdim=True)
        x = x - seq_mean
        x = x.reshape(-1, self.period_len, self.seg_num_x)
        return x, {"mean": seq_mean}

    def _denormalize_output(self, x, norm_stats, batch_size, var_num):
        if self.use_period_norm:
            return x + norm_stats["period_mean"]

        x = x.reshape(batch_size, var_num, -1)
        x = x + norm_stats["mean"]
        return x.reshape(-1, self.period_len, self.seg_num_y)

    def get_aux_loss(self):
        return self._aux_loss

    def forward(self, var_x, marker_x, marker_y):
        if var_x.ndim != 3:
            raise ValueError("TimeBase expects var_x as [B, hist_len, N], but received shape {}".format(tuple(var_x.shape)))
        batch_size, hist_len, var_num = var_x.shape
        if hist_len != self.seq_len:
            raise ValueError(
                "TimeBase requires hist_len {} at runtime, but received {}".format(self.seq_len, hist_len)
            )
        if var_num != self.enc_in:
            raise ValueError("TimeBase requires var_num {} at runtime, but received {}".format(self.enc_in, var_num))

        x = var_x.permute(0, 2, 1)
        x = self._pad_input(x)
        x = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.period_len)
        x = x.permute(0, 1, 3, 2).reshape(-1, self.period_len, self.seg_num_x)
        x, norm_stats = self._normalize_input(x, batch_size, var_num)

        if self.individual:
            x = x.reshape(batch_size, var_num, self.period_len, self.seg_num_x)
            basis_list = []
            prediction_list = []
            for channel_idx in range(self.enc_in):
                basis = self.ts2basis[channel_idx](x[:, channel_idx, :, :])
                prediction = self.basis2ts[channel_idx](basis)
                basis_list.append(basis.unsqueeze(1))
                prediction_list.append(prediction.unsqueeze(1))
            x_basis = torch.cat(basis_list, dim=1)
            x = torch.cat(prediction_list, dim=1)
            x_basis = x_basis.reshape(-1, self.period_len, self.basis_num)
            x = x.reshape(-1, self.period_len, self.seg_num_y)
        else:
            x_basis = self.ts2basis(x)
            x = self.basis2ts(x_basis)

        x = self._denormalize_output(x, norm_stats, batch_size, var_num)
        prediction = x.reshape(batch_size, self.enc_in, self.period_len, self.seg_num_y).permute(0, 1, 3, 2)
        prediction = prediction.reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)

        if self.use_orthogonal:
            self._aux_loss = calculate_orthogonal_loss(x_basis)
        else:
            self._aux_loss = None
        return prediction[:, : self.pred_len, :]
