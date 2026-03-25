import torch
import torch.nn as nn

from ._moment_utils import MomentForecastBackbone


class PredictionHead(nn.Module):
    def __init__(self, n_vars, hidden, forecast_len, head_dropout=0.2):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(hidden, forecast_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x.transpose(2, 1)


class ProjectBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.weight_layer = nn.Linear(d_model, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden):
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.dropout1(torch.nn.functional.gelu(self.fc1(hidden)))
        hidden = self.fc2(hidden)
        avg = hidden.mean(dim=-2)
        hidden = self.dropout2(hidden)
        weight = self.softmax(self.weight_layer(avg).squeeze(-1))
        hidden = hidden * weight.unsqueeze(-1).unsqueeze(-1)
        return hidden + residual


class ChannelContrastive(nn.Module):
    def __init__(self, n_vars, model_dim, m_dim=None, k_order=3, de=4, thresold=0.3):
        super().__init__()
        self.n_vars = n_vars
        self.m_dim = m_dim if m_dim is not None else max(1, n_vars // 10 + 1)
        self.k_order = k_order
        self.thresold = thresold
        self.q = nn.Parameter(torch.randn(n_vars, self.m_dim))
        self.v1 = nn.Parameter(torch.randn(self.m_dim, de))
        self.v2 = nn.Parameter(torch.randn(self.m_dim, de))
        self.f = nn.Linear(model_dim, self.k_order)
        self.a = None

    def polynomial(self, embedding):
        embedding = embedding.permute(0, 2, 1, 3).reshape(-1, embedding.shape[1], embedding.shape[-1])
        q_power = self.q
        coeff = self.f(embedding).unsqueeze(-2).expand(-1, -1, self.m_dim, -1)
        q_mixture = coeff[..., 0] * q_power
        for index in range(1, self.k_order):
            q_power = q_power * self.q
            q_mixture = coeff[..., index] * q_power + q_mixture
        return q_mixture

    def composition(self, ts, q_mixture):
        v_matrix = torch.mm(self.v1, self.v2.transpose(0, 1)).unsqueeze(0).expand(q_mixture.size(0), -1, -1)
        corr = torch.sigmoid(torch.bmm(torch.bmm(q_mixture, v_matrix), q_mixture.transpose(1, 2)))
        return (self.cal_pearson_corr(ts) + corr) / 2

    def cal_corr(self, ts, embedding):
        q_mixture = self.polynomial(embedding)
        self.a = self.composition(ts, q_mixture)

    def forward(self, features, polarity):
        if self.a is None:
            raise RuntimeError("correlation matrix must be computed before contrastive loss")

        adjacency = self.a
        if polarity == "neg":
            adjacency = adjacency * ((adjacency < -1 * self.thresold).float() * -1) + adjacency * (adjacency == 1).float()
        else:
            adjacency = adjacency * (adjacency > self.thresold).float()

        distance = self.get_feature_dis(features)
        return self.cal_loss(distance, adjacency), adjacency

    @staticmethod
    def cal_loss(distance, adjacency):
        distance = torch.exp(distance)
        distance_sum = torch.sum(distance, dim=-1)
        distance_sum_pos = torch.sum(distance * adjacency, dim=-1)
        return -torch.log(distance_sum_pos * distance_sum.pow(-1) + 1e-8).mean()

    @staticmethod
    def cal_pearson_corr(ts):
        _, _, series_length = ts.shape
        mean = ts.mean(dim=2, keepdim=True)
        std = ts.std(dim=2, unbiased=False, keepdim=True)
        centered = ts - mean
        cov = torch.matmul(centered, centered.transpose(1, 2)) / series_length
        std_outer = torch.matmul(std, std.transpose(1, 2))
        std_outer = torch.where(std_outer == 0, torch.tensor(1e-8, device=ts.device), std_outer)
        corr = cov / std_outer
        eye = torch.eye(corr.size(1), device=ts.device).unsqueeze(0).expand(corr.size(0), -1, -1)
        return corr * (1 - eye) + eye

    @staticmethod
    def get_feature_dis(features):
        distance = torch.matmul(features, features.transpose(-2, -1))
        mask = torch.eye(distance.shape[-1], device=features.device).unsqueeze(0)
        norm = torch.sum(features ** 2, dim=2, keepdim=True).sqrt()
        norm = torch.matmul(norm, norm.transpose(-2, -1)) + 1e-8
        distance = distance / norm
        return (1 - mask) * distance


class _MomentBackbone(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        moment_model_name_or_path,
        freeze_encoder=True,
        freeze_embedder=True,
        freeze_head=False,
    ):
        super().__init__()
        self.backbone = MomentForecastBackbone(
            hist_len=hist_len,
            pred_len=pred_len,
            moment_model_name_or_path=moment_model_name_or_path,
            freeze_encoder=freeze_encoder,
            freeze_embedder=freeze_embedder,
            freeze_head=freeze_head,
            error_prefix="CoRA support",
        )
        self.model = self.backbone.model
        self.patch_size = self.backbone.patch_size
        self.stride = self.backbone.stride
        self.model_dim = self.backbone.model_dim
        self.patch_num = self.backbone.patch_num

    def get_settings(self):
        return self.model_dim, self.patch_size, self.stride, self.patch_num

    def forecast_for_plugin(self, inputs):
        return self.backbone.forecast_for_plugin(inputs)

    def denorm_for_plugin(self, inputs):
        return self.backbone.denorm(inputs)


class Model(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        var_num,
        foundation_model,
        moment_model_name_or_path,
        plugin_dim,
        num_before,
        num_after,
        beta,
        dropout,
        head_dropout,
        plugin_lr=None,
        backbone_lr=None,
        freeze_encoder=True,
        freeze_embedder=True,
        freeze_head=False,
        gama=0.001,
        K=3,
        de=4,
        thresold=0.3,
    ):
        super().__init__()
        if foundation_model.upper() != "MOMENT":
            raise ValueError("CoRA compatibility currently supports only foundation_model='MOMENT'")

        self.fm = _MomentBackbone(
            hist_len=hist_len,
            pred_len=pred_len,
            moment_model_name_or_path=moment_model_name_or_path,
            freeze_encoder=freeze_encoder,
            freeze_embedder=freeze_embedder,
            freeze_head=freeze_head,
        )
        model_dim, self.patch_size, self.stride, self.patch_num = self.fm.get_settings()
        self.n_vars = var_num
        self.pred_len = pred_len
        self.plugin_lr = plugin_lr
        self.backbone_lr = backbone_lr
        self.gama = gama
        self.dropout = nn.Dropout(dropout)
        self.adapter = nn.ModuleDict(
            {
                "pos": nn.Sequential(
                    nn.Linear(model_dim, model_dim),
                    nn.Dropout(dropout),
                    nn.GELU(),
                    nn.Linear(model_dim, plugin_dim),
                ),
                "neg": nn.Sequential(
                    nn.Linear(model_dim, model_dim),
                    nn.Dropout(dropout),
                    nn.GELU(),
                    nn.Linear(model_dim, plugin_dim),
                ),
            }
        )
        self.projections_before = nn.ModuleDict(
            {
                "pos": nn.ModuleList([ProjectBlock(plugin_dim, dropout=dropout) for _ in range(num_before)]),
                "neg": nn.ModuleList([ProjectBlock(plugin_dim, dropout=dropout) for _ in range(num_before)]),
            }
        )
        self.projections_after = nn.ModuleDict(
            {
                "pos": nn.ModuleList([ProjectBlock(plugin_dim, dropout=dropout) for _ in range(num_after)]),
                "neg": nn.ModuleList([ProjectBlock(plugin_dim, dropout=dropout) for _ in range(num_after)]),
            }
        )
        self.contrastive = ChannelContrastive(
            n_vars=var_num,
            model_dim=model_dim,
            m_dim=max(1, var_num // 10 + 1),
            k_order=K,
            de=de,
            thresold=thresold,
        )
        self.head = nn.ModuleDict(
            {
                "pos": PredictionHead(var_num, self.patch_num * plugin_dim, pred_len, head_dropout=head_dropout),
                "neg": PredictionHead(var_num, self.patch_num * plugin_dim, pred_len, head_dropout=head_dropout),
            }
        )
        self.norm = nn.LayerNorm(plugin_dim)
        self.beta = nn.Parameter(torch.tensor([beta] * var_num, dtype=torch.float32))
        self._last_aux_loss = None

    def get_param_groups(self, default_lr):
        plugin_params = []
        backbone_param_ids = {id(parameter) for parameter in self.fm.parameters()}
        for parameter in self.parameters():
            if id(parameter) not in backbone_param_ids and parameter.requires_grad:
                plugin_params.append(parameter)

        param_groups = []
        backbone_params = [parameter for parameter in self.fm.parameters() if parameter.requires_grad]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.backbone_lr or default_lr})
        if plugin_params:
            param_groups.append({"params": plugin_params, "lr": self.plugin_lr or default_lr})
        return param_groups

    def forward(self, var_x, marker_x):
        del marker_x
        output, embedding = self.fm.forecast_for_plugin(var_x)
        embedding = self.dropout(embedding)
        ts_patch = var_x.unfold(1, self.patch_size, self.stride)
        ts_patch = ts_patch.permute(0, 1, 2, 3).reshape(-1, self.n_vars, self.patch_size)
        self.contrastive.cal_corr(ts_patch, embedding)

        losses = {}
        enhanced = {}
        for polarity in ("pos", "neg"):
            x_embed = self.adapter[polarity](embedding)
            x_mixer = x_embed
            for mixer in self.projections_before[polarity]:
                x_mixer = mixer(x_mixer)

            if self.training:
                features = x_mixer.permute(0, 2, 1, 3).reshape(-1, self.n_vars, x_mixer.shape[-1])
                losses[polarity], _ = self.contrastive(features, polarity)

            for mixer in self.projections_after[polarity]:
                x_mixer = mixer(x_mixer)

            x_mixer = self.norm(x_mixer + x_embed)
            enhanced[polarity] = x_mixer

        enhanced = enhanced["neg"] + enhanced["pos"]
        enhanced = self.head["neg"](enhanced)
        output_plugin = self.fm.denorm_for_plugin(enhanced)
        gate = self.beta.view(1, 1, -1)
        output = output_plugin * gate + output * (1 - gate)

        if self.training:
            self._last_aux_loss = self.gama * (losses["neg"] + losses["pos"])
        else:
            self._last_aux_loss = None

        return output

    def get_aux_loss(self):
        return self._last_aux_loss
