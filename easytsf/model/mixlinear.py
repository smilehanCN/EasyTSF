import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        if bias:
            raise ValueError("ComplexLinear only supports bias=False")

        weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight.to(torch.cfloat))

    def forward(self, x):
        return torch.matmul(x, self.weight.transpose(-1, -2))


class Model(nn.Module):
    def __init__(self, hist_len, pred_len, var_num, period_len, lpf, alpha):
        super().__init__()
        if period_len <= 0:
            raise ValueError("MixLinear requires period_len > 0, but received {}".format(period_len))
        if period_len % 2 != 0:
            raise ValueError("MixLinear requires even period_len, but received {}".format(period_len))
        if hist_len % period_len != 0:
            raise ValueError(
                "MixLinear requires hist_len {} to be divisible by period_len {}".format(hist_len, period_len)
            )

        seg_num_x = hist_len // period_len
        if lpf < 1 or lpf > seg_num_x:
            raise ValueError(
                "MixLinear requires lpf in [1, {}] for hist_len {} and period_len {}, but received {}".format(
                    seg_num_x,
                    hist_len,
                    period_len,
                    lpf,
                )
            )

        self.seq_len = hist_len
        self.pred_len = pred_len
        self.enc_in = var_num
        self.period_len = period_len
        self.lpf = lpf
        self.alpha = alpha

        self.seg_num_x = seg_num_x
        self.seg_num_y = math.ceil(self.pred_len / self.period_len)
        self.sqrt_seg_num_x = math.ceil(math.sqrt(self.seg_num_x))
        self.sqrt_seg_num_y = math.ceil(math.sqrt(self.seg_num_y))

        self.tlinear1 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)
        self.tlinear2 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.period_len + 1,
            stride=1,
            padding=self.period_len // 2,
            padding_mode="zeros",
            bias=False,
        )

        self.flinear1 = ComplexLinear(self.lpf, 2, bias=False)
        self.flinear2 = ComplexLinear(2, self.seg_num_y, bias=False)

    def forward(self, var_x, marker_x, marker_y):
        del marker_x, marker_y
        batch_size = var_x.shape[0]

        seq_mean = torch.mean(var_x, dim=1, keepdim=True)
        x = (var_x - seq_mean).permute(0, 2, 1)
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(batch_size, self.enc_in, self.seq_len) + x
        x = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.period_len).permute(0, 1, 3, 2)

        x_time = F.pad(x, (0, self.sqrt_seg_num_x ** 2 - x.shape[-1], 0, 0, 0, 0))
        x_time = x_time.reshape(
            batch_size,
            self.enc_in,
            self.period_len,
            self.sqrt_seg_num_x,
            self.sqrt_seg_num_x,
        )
        x_time = self.tlinear1(x_time).permute(0, 1, 2, 4, 3)
        x_time = self.tlinear2(x_time).permute(0, 1, 2, 4, 3)
        x_time = x_time.reshape(batch_size, self.enc_in, self.period_len, -1)
        x_time = x_time.permute(0, 1, 3, 2).reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)

        x_freq = torch.fft.fft(x, dim=3)[:, :, :, : self.lpf]
        x_freq = self.flinear1(x_freq)
        x_freq = self.flinear2(x_freq).reshape(batch_size, self.enc_in, self.period_len, -1)
        x_freq = torch.fft.ifft(x_freq, dim=3).real
        x_freq = x_freq.permute(0, 1, 3, 2).reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)

        prediction = (
            x_time[:, : self.pred_len, :] * self.alpha
            + seq_mean
            + x_freq[:, : self.pred_len, :] * (1 - self.alpha)
        )
        return prediction[:, : self.pred_len, :]
