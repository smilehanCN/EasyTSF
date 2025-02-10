import torch
import torch.nn as nn


class ResMLP(nn.Module):
    def __init__(self, x_dim, h_dim, bias=True, drop_rate=0.15):
        """
        具有残差连接的多层感知机
        Args:
            x_dim: 该模块的输入和输出维度，残差结构限制输入维度等于输出维度
        """
        super().__init__()
        self.fc1 = nn.Conv1d(x_dim, h_dim, kernel_size=1, bias=bias)
        self.fc2 = nn.Conv1d(h_dim, x_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, input_data):
        output = self.fc2(self.drop(self.act(self.fc1(input_data))))
        output = output + input_data
        return output


class STID(nn.Module):
    def __init__(self, hist_len, pred_len, var_num, block_num, ts_emb_dim, node_emb_dim, tod_emb_dim, dow_emb_dim,
                 freq):
        super(STID, self).__init__()
        self.in_len = hist_len
        self.out_len = pred_len
        self.var_num = var_num
        self.block_num = block_num
        self.ts_emb_dim = ts_emb_dim
        self.node_emb_dim = node_emb_dim
        self.tod_emb_dim = tod_emb_dim
        self.dow_emb_dim = dow_emb_dim
        self.freq = freq

        self.tod_size = int((24 * 60) / self.freq)
        self.dow_size = 7
        self.hidden_dim = self.ts_emb_dim + self.node_emb_dim + self.tod_emb_dim + self.dow_emb_dim

        self.node_emb = nn.Parameter(torch.empty(self.node_emb_dim, self.var_num))
        self.tod_emb = nn.Parameter(torch.empty(self.tod_size, self.tod_emb_dim))
        self.dow_emb = nn.Parameter(torch.empty(self.dow_size, self.dow_emb_dim))
        self._init_embeddings()

        self.ts_emb_layer = nn.Conv1d(self.in_len, self.ts_emb_dim, kernel_size=1, bias=True)
        self.encoder = nn.Sequential(*[ResMLP(self.hidden_dim, self.hidden_dim) for _ in range(self.block_num)])
        self.predictor = nn.Conv1d(self.hidden_dim, self.out_len, kernel_size=1, bias=True)

    def _init_embeddings(self):
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.tod_emb)
        nn.init.xavier_uniform_(self.dow_emb)

    def forward(self, var_x, marker_x):
        B, L, N = var_x.shape
        hidden = self.ts_emb_layer(var_x)

        if self.node_emb_dim:
            node_emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)
            hidden = torch.cat((hidden, node_emb), dim=1)

        if self.tod_emb_dim:
            tod = marker_x[:, -1, 0].type(torch.LongTensor)
            tod_emb = self.tod_emb[tod].unsqueeze(-1).repeat(1, 1, N)  # (B, tod_emb_dim, N)
            hidden = torch.cat((hidden, tod_emb), dim=1)

        if self.dow_emb_dim:
            dow = marker_x[:, -1, 1].type(torch.LongTensor)
            dow_emb = self.dow_emb[dow].unsqueeze(-1).repeat(1, 1, N)
            hidden = torch.cat((hidden, dow_emb), dim=1)

        hidden = self.encoder(hidden)
        prediction = self.predictor(hidden)
        return prediction
