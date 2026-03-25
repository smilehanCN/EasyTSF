import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._moment_utils import MomentForecastBackbone
from .spatial_cora import GridProjectBlock, SpatialContrastive, TokenPredictionHead, build_grid_prior


def _normalize_spatial_shape(spatial_shape):
    if spatial_shape is None:
        raise ValueError("CoRAGrid requires spatial_shape from the grid datamodule")
    shape = tuple(int(size) for size in spatial_shape)
    if len(shape) not in {2, 3}:
        raise ValueError("CoRAGrid supports only 2D or 3D grids")
    return shape


def _normalize_patch_size(spatial_patch_size, spatial_ndim):
    if isinstance(spatial_patch_size, int):
        return (int(spatial_patch_size),) * spatial_ndim
    if not isinstance(spatial_patch_size, (list, tuple)):
        raise ValueError("spatial_patch_size must be an int or a sequence")
    patch_size = tuple(int(size) for size in spatial_patch_size)
    if len(patch_size) != spatial_ndim:
        raise ValueError("spatial_patch_size {} does not match spatial ndim {}".format(patch_size, spatial_ndim))
    return patch_size


class Model(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        channel_num,
        spatial_shape,
        foundation_model,
        moment_model_name_or_path,
        plugin_dim,
        num_before,
        num_after,
        beta,
        dropout,
        head_dropout,
        spatial_patch_size,
        spatial_mixer_type="conv",
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
            raise ValueError("CoRAGrid currently supports only foundation_model='MOMENT'")
        if spatial_mixer_type != "conv":
            raise NotImplementedError("CoRAGrid v1 supports only spatial_mixer_type='conv'")

        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.spatial_shape = _normalize_spatial_shape(spatial_shape)
        self.spatial_ndim = len(self.spatial_shape)
        self.spatial_patch_size = _normalize_patch_size(spatial_patch_size, self.spatial_ndim)
        self.padded_shape = tuple(
            int(math.ceil(size / patch_size) * patch_size)
            for size, patch_size in zip(self.spatial_shape, self.spatial_patch_size)
        )
        self.token_grid_shape = tuple(
            padded // patch_size for padded, patch_size in zip(self.padded_shape, self.spatial_patch_size)
        )
        self.num_tokens = math.prod(self.token_grid_shape)
        self.channel_num = int(channel_num)
        self.token_dim = self.channel_num * math.prod(self.spatial_patch_size)
        self.neighbor_order = neighbor_order
        self.plugin_lr = plugin_lr
        self.backbone_lr = backbone_lr
        self.gama = gama

        self.fm = MomentForecastBackbone(
            hist_len=hist_len,
            pred_len=pred_len,
            moment_model_name_or_path=moment_model_name_or_path,
            freeze_encoder=freeze_encoder,
            freeze_embedder=freeze_embedder,
            freeze_head=freeze_head,
            error_prefix="CoRAGrid support",
        )
        model_dim, self.patch_size, self.stride, self.patch_num = self.fm.get_settings()
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
                "pos": nn.ModuleList(
                    [GridProjectBlock(plugin_dim, self.spatial_ndim, dropout=dropout) for _ in range(num_before)]
                ),
                "neg": nn.ModuleList(
                    [GridProjectBlock(plugin_dim, self.spatial_ndim, dropout=dropout) for _ in range(num_before)]
                ),
            }
        )
        self.projections_after = nn.ModuleDict(
            {
                "pos": nn.ModuleList(
                    [GridProjectBlock(plugin_dim, self.spatial_ndim, dropout=dropout) for _ in range(num_after)]
                ),
                "neg": nn.ModuleList(
                    [GridProjectBlock(plugin_dim, self.spatial_ndim, dropout=dropout) for _ in range(num_after)]
                ),
            }
        )
        self.contrastive = SpatialContrastive(
            num_tokens=self.num_tokens,
            model_dim=model_dim,
            m_dim=max(1, self.num_tokens // 10 + 1),
            k_order=K,
            de=de,
            threshold=thresold,
            structure_prior_weight=structure_prior_weight,
        )
        self.head = nn.ModuleDict(
            {
                "pos": TokenPredictionHead(self.patch_num * plugin_dim, pred_len, self.token_dim, head_dropout=head_dropout),
                "neg": TokenPredictionHead(self.patch_num * plugin_dim, pred_len, self.token_dim, head_dropout=head_dropout),
            }
        )
        self.norm = nn.LayerNorm(plugin_dim)
        self.beta = nn.Parameter(torch.tensor(float(beta), dtype=torch.float32))
        self._last_aux_loss = None
        self.register_buffer(
            "grid_prior",
            build_grid_prior(self.token_grid_shape, self.neighbor_order, dtype=torch.float32),
            persistent=False,
        )

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

    def _pad_input(self, var_x):
        pad_sizes = []
        for current, target in zip(reversed(var_x.shape[-self.spatial_ndim:]), reversed(self.padded_shape)):
            pad_sizes.extend([0, target - current])
        if any(pad_sizes):
            var_x = F.pad(var_x, tuple(pad_sizes))
        return var_x

    def _patchify(self, var_x):
        if tuple(var_x.shape[-self.spatial_ndim:]) != self.spatial_shape:
            raise ValueError(
                "CoRAGrid expected spatial shape {} but received {}".format(
                    self.spatial_shape,
                    tuple(var_x.shape[-self.spatial_ndim:]),
                )
            )

        var_x = self._pad_input(var_x)
        if self.spatial_ndim == 2:
            batch_size, seq_len, channels, height, width = var_x.shape
            patch_h, patch_w = self.spatial_patch_size
            grid_h, grid_w = self.token_grid_shape
            tokens = var_x.view(batch_size, seq_len, channels, grid_h, patch_h, grid_w, patch_w)
            tokens = tokens.permute(0, 1, 3, 5, 2, 4, 6).reshape(batch_size, seq_len, self.num_tokens, self.token_dim)
            return tokens

        batch_size, seq_len, channels, size_x, size_y, size_z = var_x.shape
        patch_x, patch_y, patch_z = self.spatial_patch_size
        grid_x, grid_y, grid_z = self.token_grid_shape
        tokens = var_x.view(batch_size, seq_len, channels, grid_x, patch_x, grid_y, patch_y, grid_z, patch_z)
        tokens = tokens.permute(0, 1, 3, 5, 7, 2, 4, 6, 8).reshape(batch_size, seq_len, self.num_tokens, self.token_dim)
        return tokens

    def _unpatchify(self, tokens):
        batch_size, seq_len, num_tokens, token_dim = tokens.shape
        if num_tokens != self.num_tokens or token_dim != self.token_dim:
            raise ValueError("unexpected token tensor shape for unpatchify: {}".format(tokens.shape))

        if self.spatial_ndim == 2:
            grid_h, grid_w = self.token_grid_shape
            patch_h, patch_w = self.spatial_patch_size
            tokens = tokens.view(batch_size, seq_len, grid_h, grid_w, self.channel_num, patch_h, patch_w)
            tokens = tokens.permute(0, 1, 4, 2, 5, 3, 6).reshape(
                batch_size,
                seq_len,
                self.channel_num,
                grid_h * patch_h,
                grid_w * patch_w,
            )
            return tokens[..., :self.spatial_shape[0], :self.spatial_shape[1]]

        grid_x, grid_y, grid_z = self.token_grid_shape
        patch_x, patch_y, patch_z = self.spatial_patch_size
        tokens = tokens.view(batch_size, seq_len, grid_x, grid_y, grid_z, self.channel_num, patch_x, patch_y, patch_z)
        tokens = tokens.permute(0, 1, 5, 2, 6, 3, 7, 4, 8).reshape(
            batch_size,
            seq_len,
            self.channel_num,
            grid_x * patch_x,
            grid_y * patch_y,
            grid_z * patch_z,
        )
        return tokens[..., :self.spatial_shape[0], :self.spatial_shape[1], :self.spatial_shape[2]]

    def forward(self, var_x, marker_x, grid_mask=None, coord=None):
        del marker_x, grid_mask, coord
        batch_size, _, channel_num = var_x.shape[:3]
        if channel_num != self.channel_num:
            raise ValueError("CoRAGrid expected {} channels but received {}".format(self.channel_num, channel_num))

        tokens = self._patchify(var_x)
        token_inputs = tokens.reshape(batch_size * self.num_tokens, var_x.shape[1], self.token_dim)
        output, embedding = self.fm.forecast_for_plugin(token_inputs)
        output = output.view(batch_size, self.num_tokens, self.pred_len, self.token_dim).permute(0, 2, 1, 3).contiguous()
        embedding = embedding.view(batch_size, self.num_tokens, self.token_dim, self.patch_num, -1).mean(dim=2)
        embedding = self.dropout(embedding)

        token_series = tokens.mean(dim=-1).transpose(1, 2)
        ts_patch = token_series.unfold(-1, self.patch_size, self.stride)
        ts_patch = ts_patch.permute(0, 2, 1, 3).reshape(-1, self.num_tokens, self.patch_size)
        self.contrastive.cal_corr(ts_patch, embedding, self.grid_prior)

        losses = {}
        enhanced = {}
        for polarity in ("pos", "neg"):
            x_embed = self.adapter[polarity](embedding)
            x_mixer = x_embed
            for mixer in self.projections_before[polarity]:
                x_mixer = mixer(x_mixer, self.token_grid_shape)

            if self.training:
                features = x_mixer.permute(0, 2, 1, 3).reshape(-1, self.num_tokens, x_mixer.shape[-1])
                losses[polarity], _ = self.contrastive(features, polarity)

            for mixer in self.projections_after[polarity]:
                x_mixer = mixer(x_mixer, self.token_grid_shape)

            x_mixer = self.norm(x_mixer + x_embed)
            enhanced[polarity] = x_mixer

        enhanced = enhanced["neg"] + enhanced["pos"]
        enhanced = self.head["neg"](enhanced)
        enhanced = enhanced.reshape(batch_size * self.num_tokens, self.pred_len, self.token_dim)
        output_plugin = self.fm.denorm(enhanced)
        output_plugin = output_plugin.view(batch_size, self.num_tokens, self.pred_len, self.token_dim)
        output_plugin = output_plugin.permute(0, 2, 1, 3).contiguous()

        output = self._unpatchify(output)
        output_plugin = self._unpatchify(output_plugin)
        output = output_plugin * self.beta + output * (1 - self.beta)

        if self.training:
            self._last_aux_loss = self.gama * (losses["neg"] + losses["pos"])
        else:
            self._last_aux_loss = None
        return output

    def get_aux_loss(self):
        return self._last_aux_loss
