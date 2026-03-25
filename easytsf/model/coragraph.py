import torch
import torch.nn as nn

from ._moment_utils import MomentForecastBackbone
from .spatial_cora import GraphProjectBlock, SpatialContrastive, TemporalPredictionHead, build_graph_prior


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
        structure_prior_weight=0.5,
        neighbor_order=1,
    ):
        super().__init__()
        if foundation_model.upper() != "MOMENT":
            raise ValueError("CoRAGraph currently supports only foundation_model='MOMENT'")

        self.fm = MomentForecastBackbone(
            hist_len=hist_len,
            pred_len=pred_len,
            moment_model_name_or_path=moment_model_name_or_path,
            freeze_encoder=freeze_encoder,
            freeze_embedder=freeze_embedder,
            freeze_head=freeze_head,
            error_prefix="CoRAGraph support",
        )
        model_dim, self.patch_size, self.stride, self.patch_num = self.fm.get_settings()
        self.num_nodes = var_num
        self.pred_len = pred_len
        self.plugin_lr = plugin_lr
        self.backbone_lr = backbone_lr
        self.gama = gama
        self.neighbor_order = neighbor_order
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
                "pos": nn.ModuleList([GraphProjectBlock(plugin_dim, dropout=dropout) for _ in range(num_before)]),
                "neg": nn.ModuleList([GraphProjectBlock(plugin_dim, dropout=dropout) for _ in range(num_before)]),
            }
        )
        self.projections_after = nn.ModuleDict(
            {
                "pos": nn.ModuleList([GraphProjectBlock(plugin_dim, dropout=dropout) for _ in range(num_after)]),
                "neg": nn.ModuleList([GraphProjectBlock(plugin_dim, dropout=dropout) for _ in range(num_after)]),
            }
        )
        self.contrastive = SpatialContrastive(
            num_tokens=var_num,
            model_dim=model_dim,
            m_dim=max(1, var_num // 10 + 1),
            k_order=K,
            de=de,
            threshold=thresold,
            structure_prior_weight=structure_prior_weight,
        )
        self.head = nn.ModuleDict(
            {
                "pos": TemporalPredictionHead(self.patch_num * plugin_dim, pred_len, head_dropout=head_dropout),
                "neg": TemporalPredictionHead(self.patch_num * plugin_dim, pred_len, head_dropout=head_dropout),
            }
        )
        self.norm = nn.LayerNorm(plugin_dim)
        self.beta = nn.Parameter(torch.tensor(float(beta), dtype=torch.float32))
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

    def forward(self, var_x, marker_x, graph):
        del marker_x
        batch_size, _, num_nodes = var_x.shape
        if num_nodes != self.num_nodes:
            raise ValueError("CoRAGraph expected {} nodes but received {}".format(self.num_nodes, num_nodes))

        node_inputs = var_x.transpose(1, 2).reshape(batch_size * num_nodes, var_x.shape[1], 1)
        output, embedding = self.fm.forecast_for_plugin(node_inputs)
        output = output.view(batch_size, num_nodes, self.pred_len).transpose(1, 2).contiguous()
        embedding = embedding.view(batch_size, num_nodes, 1, self.patch_num, -1).mean(dim=2)
        embedding = self.dropout(embedding)

        ts_patch = var_x.transpose(1, 2).unfold(-1, self.patch_size, self.stride)
        ts_patch = ts_patch.permute(0, 2, 1, 3).reshape(-1, num_nodes, self.patch_size)
        support = build_graph_prior(graph, self.neighbor_order)
        self.contrastive.cal_corr(ts_patch, embedding, support)

        losses = {}
        enhanced = {}
        for polarity in ("pos", "neg"):
            x_embed = self.adapter[polarity](embedding)
            x_mixer = x_embed
            for mixer in self.projections_before[polarity]:
                x_mixer = mixer(x_mixer, support)

            if self.training:
                features = x_mixer.permute(0, 2, 1, 3).reshape(-1, num_nodes, x_mixer.shape[-1])
                losses[polarity], _ = self.contrastive(features, polarity)

            for mixer in self.projections_after[polarity]:
                x_mixer = mixer(x_mixer, support)

            x_mixer = self.norm(x_mixer + x_embed)
            enhanced[polarity] = x_mixer

        enhanced = enhanced["neg"] + enhanced["pos"]
        enhanced = self.head["neg"](enhanced)
        enhanced = enhanced.transpose(1, 2).reshape(batch_size * num_nodes, self.pred_len, 1)
        output_plugin = self.fm.denorm(enhanced).view(batch_size, num_nodes, self.pred_len).transpose(1, 2).contiguous()
        output = output_plugin * self.beta + output * (1 - self.beta)

        if self.training:
            self._last_aux_loss = self.gama * (losses["neg"] + losses["pos"])
        else:
            self._last_aux_loss = None
        return output

    def get_aux_loss(self):
        return self._last_aux_loss
