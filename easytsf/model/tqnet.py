import torch
import torch.nn as nn


def _normalize_feature_name(name):
    return str(name).strip().lower()


class Model(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        var_num,
        cycle,
        cycle_feature_name="time of day",
        time_feature_descriptions=(),
        d_model=512,
        dropout=0.0,
        use_revin=True,
        use_tq=True,
        channel_aggre=True,
        channel_aggre_heads=4,
    ):
        super().__init__()
        self.seq_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(var_num)
        self.cycle_len = int(cycle)
        self.cycle_feature_name = str(cycle_feature_name)
        self.d_model = int(d_model)
        self.dropout = float(dropout)
        self.use_revin = bool(use_revin)
        self.use_tq = bool(use_tq)
        self.channel_aggre = bool(channel_aggre)
        self.channel_aggre_heads = int(channel_aggre_heads)

        if self.cycle_len <= 0:
            raise ValueError("TQNet requires cycle > 0, but received {}".format(self.cycle_len))
        if self.channel_aggre_heads <= 0:
            raise ValueError("TQNet requires channel_aggre_heads > 0, but received {}".format(self.channel_aggre_heads))
        if self.channel_aggre and self.seq_len % self.channel_aggre_heads != 0:
            raise ValueError(
                "TQNet requires hist_len {} to be divisible by channel_aggre_heads {}".format(
                    self.seq_len,
                    self.channel_aggre_heads,
                )
            )

        normalized_descriptions = tuple(_normalize_feature_name(item) for item in (time_feature_descriptions or ()))
        normalized_cycle_feature = _normalize_feature_name(self.cycle_feature_name)
        if normalized_cycle_feature not in normalized_descriptions:
            raise ValueError(
                "TQNet requires time_feature_descriptions to include '{}', but received {}".format(
                    self.cycle_feature_name,
                    list(time_feature_descriptions or ()),
                )
            )
        self.cycle_feature_idx = normalized_descriptions.index(normalized_cycle_feature)

        if self.use_tq:
            self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)

        if self.channel_aggre:
            self.channelAggregator = nn.MultiheadAttention(
                embed_dim=self.seq_len,
                num_heads=self.channel_aggre_heads,
                batch_first=True,
                dropout=0.5,
            )

        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len),
        )

    def _extract_cycle_index(self, marker_y):
        if marker_y is None:
            raise ValueError("TQNet requires marker_y in forward(var_x, marker_x, marker_y) to derive cycle_index")
        if marker_y.ndim != 3:
            raise ValueError("TQNet expects marker_y as [B, pred_len, T], but received shape {}".format(tuple(marker_y.shape)))
        if marker_y.shape[1] <= 0:
            raise ValueError("TQNet requires marker_y to contain at least one future step")
        if marker_y.shape[2] <= self.cycle_feature_idx:
            raise ValueError(
                "TQNet requires marker_y feature dimension > {}, but received shape {}".format(
                    self.cycle_feature_idx,
                    tuple(marker_y.shape),
                )
            )
        cycle_index = marker_y[:, 0, self.cycle_feature_idx].round().long()
        return cycle_index % self.cycle_len

    def _forecast(self, x, cycle_index):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x_input = x.permute(0, 2, 1)

        if self.use_tq:
            gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
            query_input = self.temporalQuery[gather_index].permute(0, 2, 1)
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=query_input, key=x_input, value=x_input)[0]
            else:
                channel_information = query_input
        else:
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=x_input, key=x_input, value=x_input)[0]
            else:
                channel_information = torch.zeros_like(x_input)

        model_input = self.input_proj(x_input + channel_information)
        hidden = self.model(model_input)
        output = self.output_proj(hidden + model_input).permute(0, 2, 1)

        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean
        return output

    def forward(self, var_x, marker_x, marker_y):
        del marker_x
        cycle_index = self._extract_cycle_index(marker_y)
        output = self._forecast(var_x, cycle_index)
        return output[:, -self.pred_len:, :]
