from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from .grid3d_forecasting import Grid3DForecastingTask


_HEAD_NAMES = ("shear_x", "shear_y", "shear_z", "speed_cls")


class Grid3DRiskPredictionTask(Grid3DForecastingTask):
    def _setup_task_state(self):
        super()._setup_task_state()
        self.output_mode = str(getattr(self.hparams, "output_mode", "classification"))
        if self.output_mode != "classification":
            raise ValueError("grid3d_risk_prediction requires output_mode='classification'")

        self.risk_num_classes = int(getattr(self.hparams, "risk_num_classes", 3))
        self.risk_num_heads = int(getattr(self.hparams, "risk_num_heads", 4))
        if self.risk_num_classes < 2:
            raise ValueError("grid3d_risk_prediction expects risk_num_classes>=2")
        if self.risk_num_heads != len(_HEAD_NAMES):
            raise ValueError(
                "grid3d_risk_prediction expects risk_num_heads={}, got {}".format(
                    len(_HEAD_NAMES),
                    self.risk_num_heads,
                )
            )

        self.risk_bins = self._parse_risk_bins(getattr(self.hparams, "risk_bins", None))
        self.grid_spacing_m = self._resolve_grid_spacing(getattr(self.hparams, "grid_spacing_m", None))
        self.head_loss_weights = self._parse_head_loss_weights(getattr(self.hparams, "head_loss_weights", None))
        self.risk_class_weights = self._parse_risk_class_weights(getattr(self.hparams, "risk_class_weights", None))

        self.high_risk_class_index = int(getattr(self.hparams, "high_risk_class_index", self.risk_num_classes - 1))
        if self.high_risk_class_index < 0 or self.high_risk_class_index >= self.risk_num_classes:
            raise ValueError(
                "high_risk_class_index {} is out of range for risk_num_classes={}".format(
                    self.high_risk_class_index,
                    self.risk_num_classes,
                )
            )

        self.val_confusion = None
        self.val_head_confusion = None
        self.test_confusion = None
        self.test_head_confusion = None

    def _get_model_derived_args(self):
        derived = dict(super()._get_model_derived_args())
        derived.update(
            {
                "output_mode": "classification",
                "risk_num_classes": self.risk_num_classes,
                "risk_num_heads": self.risk_num_heads,
            }
        )
        return derived

    def _build_loss_function(self):
        return self._compute_risk_loss

    def _parse_risk_bins(self, raw_bins):
        if not isinstance(raw_bins, Mapping):
            raise ValueError("grid3d_risk_prediction requires risk_bins mapping")

        def parse_thresholds(key, values):
            expected_bin_count = self.risk_num_classes - 1
            if (
                not isinstance(values, Sequence)
                or isinstance(values, (str, bytes))
                or len(values) != expected_bin_count
            ):
                raise ValueError(
                    "risk_bins['{}'] must be a {}-item list/tuple for risk_num_classes={}".format(
                        key,
                        expected_bin_count,
                        self.risk_num_classes,
                    )
                )
            parsed_values = tuple(float(value) for value in values)
            if any(left >= right for left, right in zip(parsed_values, parsed_values[1:])):
                raise ValueError(
                    "risk_bins['{}'] must be strictly increasing, got {}".format(key, list(parsed_values))
                )
            return parsed_values

        parsed = {}
        if "speed" not in raw_bins:
            raise ValueError("risk_bins must define 'speed'")
        parsed["speed"] = parse_thresholds("speed", raw_bins["speed"])

        if "shear" not in raw_bins:
            raise ValueError("risk_bins must define unified 'shear'")
        parsed_shear = parse_thresholds("shear", raw_bins["shear"])
        for key in ("shear_x", "shear_y", "shear_z"):
            parsed[key] = parsed_shear
        return parsed

    def _parse_grid_spacing(self, raw_spacing):
        if (
            not isinstance(raw_spacing, Sequence)
            or isinstance(raw_spacing, (str, bytes))
            or len(raw_spacing) != 3
        ):
            raise ValueError("grid_spacing_m must be a length-3 list/tuple in [dy, dx, dz] order")
        spacing = tuple(float(value) for value in raw_spacing)
        if any(value <= 0.0 for value in spacing):
            raise ValueError("grid_spacing_m values must be > 0")
        return spacing

    def _resolve_grid_spacing(self, raw_spacing):
        meta_spacing = self.meta.get("grid_spacing_m")
        if not isinstance(meta_spacing, list) or len(meta_spacing) != 3:
            raise ValueError(
                "grid3d_risk_prediction requires datasets imported with grid_spacing_m in meta.json"
            )
        parsed_meta_spacing = self._parse_grid_spacing(meta_spacing)
        if raw_spacing is None:
            return parsed_meta_spacing
        parsed_runtime_spacing = self._parse_grid_spacing(raw_spacing)
        if parsed_runtime_spacing != parsed_meta_spacing:
            raise ValueError(
                "runtime grid_spacing_m {} does not match dataset meta grid_spacing_m {}".format(
                    parsed_runtime_spacing,
                    parsed_meta_spacing,
                )
            )
        return parsed_meta_spacing

    def _parse_head_loss_weights(self, raw_weights):
        if raw_weights is None:
            return (1.0,) * self.risk_num_heads
        if (
            not isinstance(raw_weights, Sequence)
            or isinstance(raw_weights, (str, bytes))
            or len(raw_weights) != self.risk_num_heads
        ):
            raise ValueError("head_loss_weights must be a length-{} list/tuple".format(self.risk_num_heads))
        weights = tuple(float(weight) for weight in raw_weights)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("head_loss_weights values must be >= 0")
        if sum(weights) <= 0.0:
            raise ValueError("head_loss_weights must contain at least one positive value")
        return weights

    def _parse_risk_class_weights(self, raw_weights):
        if raw_weights is None:
            return None
        if (
            not isinstance(raw_weights, Sequence)
            or isinstance(raw_weights, (str, bytes))
            or len(raw_weights) != self.risk_num_classes
        ):
            raise ValueError("risk_class_weights must be a length-{} list/tuple".format(self.risk_num_classes))
        weights = tuple(float(weight) for weight in raw_weights)
        if any(weight <= 0.0 for weight in weights):
            raise ValueError("risk_class_weights values must be > 0")
        return weights

    def _bucketize_classes(self, values: torch.Tensor, thresholds: tuple[float, ...]) -> torch.Tensor:
        boundaries = values.new_tensor(thresholds)
        return torch.bucketize(values, boundaries=boundaries, right=True).long()

    def _compute_vector_shear(
        self,
        physical_targets: torch.Tensor,
        spatial_dim: int,
        spacing_m: float,
    ) -> torch.Tensor:
        axis_length = int(physical_targets.size(spatial_dim))
        out = physical_targets.new_zeros(
            physical_targets.size(0),
            physical_targets.size(1),
            physical_targets.size(3),
            physical_targets.size(4),
            physical_targets.size(5),
        )
        if axis_length <= 1:
            return out

        lhs_index = [slice(None)] * 6
        rhs_index = [slice(None)] * 6
        lhs_index[spatial_dim] = slice(1, None)
        rhs_index[spatial_dim] = slice(0, -1)
        vector_delta = physical_targets[tuple(lhs_index)] - physical_targets[tuple(rhs_index)]
        magnitude = torch.linalg.vector_norm(vector_delta, ord=2, dim=2) / float(spacing_m)

        out_spatial_dim = spatial_dim - 1
        fill_index = [slice(None)] * 5
        fill_index[out_spatial_dim] = slice(0, -1)
        out[tuple(fill_index)] = magnitude

        last_index = [slice(None)] * 5
        last_index[out_spatial_dim] = -1
        out[tuple(last_index)] = magnitude[tuple(last_index)]
        return out

    def _build_risk_labels(self, normalized_targets: torch.Tensor) -> torch.Tensor:
        physical_targets = self.scaler.inverse_transform(normalized_targets)
        u_component = physical_targets[:, :, 0, ...]
        v_component = physical_targets[:, :, 1, ...]
        w_component = physical_targets[:, :, 2, ...]

        speed = torch.sqrt(torch.clamp(u_component.square() + v_component.square() + w_component.square(), min=0.0))
        dy_m, dx_m, dz_m = self.grid_spacing_m
        shear_x = self._compute_vector_shear(physical_targets, spatial_dim=4, spacing_m=dx_m)
        shear_y = self._compute_vector_shear(physical_targets, spatial_dim=3, spacing_m=dy_m)
        shear_z = self._compute_vector_shear(physical_targets, spatial_dim=5, spacing_m=dz_m)

        labels = torch.stack(
            [
                self._bucketize_classes(shear_x, self.risk_bins["shear_x"]),
                self._bucketize_classes(shear_y, self.risk_bins["shear_y"]),
                self._bucketize_classes(shear_z, self.risk_bins["shear_z"]),
                self._bucketize_classes(speed, self.risk_bins["speed"]),
            ],
            dim=2,
        )
        return labels

    def _reshape_logits(self, prediction: torch.Tensor) -> torch.Tensor:
        expected_channels = self.risk_num_heads * self.risk_num_classes
        if prediction.ndim != 6:
            raise ValueError(
                "grid3d_risk_prediction expects logits with 6 dims [B,T,C,Y,X,Z], got {}".format(
                    tuple(prediction.shape)
                )
            )
        if int(prediction.size(2)) != expected_channels:
            raise ValueError(
                "grid3d_risk_prediction expects {} output channels, got {}".format(
                    expected_channels,
                    int(prediction.size(2)),
                )
            )
        return prediction.reshape(
            prediction.size(0),
            prediction.size(1),
            self.risk_num_heads,
            self.risk_num_classes,
            prediction.size(3),
            prediction.size(4),
            prediction.size(5),
        )

    def _compute_risk_loss(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        logits = self._reshape_logits(prediction)
        if tuple(label.shape[:3]) != (prediction.size(0), prediction.size(1), self.risk_num_heads):
            raise ValueError(
                "grid3d_risk_prediction expects labels [B,T,{},Y,X,Z], got {}".format(
                    self.risk_num_heads,
                    tuple(label.shape),
                )
            )

        class_weights = None
        if self.risk_class_weights is not None:
            class_weights = prediction.new_tensor(self.risk_class_weights)
        head_weights = prediction.new_tensor(self.head_loss_weights)

        head_losses = []
        for head_index in range(self.risk_num_heads):
            head_logits = logits[:, :, head_index, ...]
            flattened_logits = head_logits.permute(0, 1, 3, 4, 5, 2).reshape(-1, self.risk_num_classes)
            flattened_labels = label[:, :, head_index, ...].reshape(-1)
            head_loss = F.cross_entropy(
                flattened_logits,
                flattened_labels,
                weight=class_weights,
                reduction="mean",
            )
            head_losses.append(head_loss)

        stacked_losses = torch.stack(head_losses, dim=0)
        return (stacked_losses * head_weights).sum() / torch.clamp(head_weights.sum(), min=1e-12)

    def postprocess_outputs(self, prediction, label, targets_mask=None):
        del targets_mask
        return prediction, label

    def _forward(self, batch):
        var_x, var_y, coords = self.preprocess_batch(batch)
        prediction = self.model(var_x, coords=coords)
        label = self._build_risk_labels(var_y)
        return prediction, label

    def _compute_validation_loss(self, batch, prediction, label):
        loss_sum = prediction.new_zeros(())
        value_count = 0
        for pred_crop, label_crop in self._iter_eval_crops(batch, prediction, label):
            crop_loss = self._compute_risk_loss(pred_crop, label_crop)
            crop_values = int(label_crop.numel())
            loss_sum = loss_sum + crop_loss * crop_values
            value_count += crop_values
        loss = loss_sum / max(value_count, 1)
        return loss, {"batch_size": value_count}

    def _new_confusion_buffers(self):
        return (
            torch.zeros(
                self.risk_num_classes,
                self.risk_num_classes,
                dtype=torch.long,
                device=self.device,
            ),
            torch.zeros(
                self.risk_num_heads,
                self.risk_num_classes,
                self.risk_num_classes,
                dtype=torch.long,
                device=self.device,
            ),
        )

    def on_validation_epoch_start(self):
        self.val_confusion, self.val_head_confusion = self._new_confusion_buffers()

    def on_test_epoch_start(self):
        self.test_confusion, self.test_head_confusion = self._new_confusion_buffers()

    def _update_confusion(self, predictions: torch.Tensor, labels: torch.Tensor, confusion: torch.Tensor, head_confusion: torch.Tensor):
        for head_index in range(self.risk_num_heads):
            head_predictions = predictions[:, :, head_index, ...].reshape(-1).to(dtype=torch.long)
            head_labels = labels[:, :, head_index, ...].reshape(-1).to(dtype=torch.long)
            packed = head_labels * self.risk_num_classes + head_predictions
            current_confusion = torch.bincount(
                packed,
                minlength=self.risk_num_classes * self.risk_num_classes,
            ).reshape(self.risk_num_classes, self.risk_num_classes)
            confusion = confusion + current_confusion
            head_confusion[head_index] = head_confusion[head_index] + current_confusion
        return confusion, head_confusion

    def validation_step(self, batch, batch_idx):
        del batch_idx
        if self.val_confusion is None or self.val_head_confusion is None:
            self.val_confusion, self.val_head_confusion = self._new_confusion_buffers()
        prediction, label = self._forward(batch)
        loss, log_kwargs = self._compute_validation_loss(batch, prediction, label)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, **dict(log_kwargs))
        for pred_crop, label_crop in self._iter_eval_crops(batch, prediction, label):
            logits = self._reshape_logits(pred_crop)
            predicted_classes = logits.argmax(dim=3)
            self.val_confusion, self.val_head_confusion = self._update_confusion(
                predicted_classes,
                label_crop,
                self.val_confusion,
                self.val_head_confusion,
            )
        return loss

    def test_step(self, batch, batch_idx):
        del batch_idx
        if self.test_confusion is None or self.test_head_confusion is None:
            self.test_confusion, self.test_head_confusion = self._new_confusion_buffers()
        prediction, label = self._forward(batch)
        for pred_crop, label_crop in self._iter_eval_crops(batch, prediction, label):
            logits = self._reshape_logits(pred_crop)
            predicted_classes = logits.argmax(dim=3)
            self.test_confusion, self.test_head_confusion = self._update_confusion(
                predicted_classes,
                label_crop,
                self.test_confusion,
                self.test_head_confusion,
            )

    def _gather_confusion(self, confusion: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_trainer", None) is None:
            return confusion
        gathered = self.all_gather(confusion)
        if gathered.ndim == confusion.ndim + 1:
            return gathered.sum(dim=0)
        return gathered

    def _compute_confusion_metrics(self, confusion: torch.Tensor) -> dict[str, torch.Tensor]:
        confusion = confusion.to(dtype=torch.float32)
        true_positive = confusion.diag()
        predicted_positive = confusion.sum(dim=0)
        actual_positive = confusion.sum(dim=1)

        precision = torch.where(predicted_positive > 0, true_positive / predicted_positive, torch.zeros_like(true_positive))
        recall = torch.where(actual_positive > 0, true_positive / actual_positive, torch.zeros_like(true_positive))
        f1 = torch.where(precision + recall > 0, 2.0 * precision * recall / (precision + recall), torch.zeros_like(precision))
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": f1.mean(),
            "high_risk_recall": recall[self.high_risk_class_index],
        }

    def _log_confusion_metrics(self, prefix: str, confusion: torch.Tensor, head_confusion: torch.Tensor, *, log_class_metrics: bool) -> None:
        metrics = self._compute_confusion_metrics(confusion)
        self.log("{}/macro_f1".format(prefix), metrics["macro_f1"], on_step=False, on_epoch=True, sync_dist=False)
        self.log(
            "{}/high_risk_recall".format(prefix),
            metrics["high_risk_recall"],
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

        for head_index, head_name in enumerate(_HEAD_NAMES):
            head_metrics = self._compute_confusion_metrics(head_confusion[head_index])
            self.log(
                "{}/{}_macro_f1".format(prefix, head_name),
                head_metrics["macro_f1"],
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                "{}/{}_high_risk_recall".format(prefix, head_name),
                head_metrics["high_risk_recall"],
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

        if log_class_metrics:
            for class_index in range(self.risk_num_classes):
                self.log(
                    "{}/class_{}_precision".format(prefix, class_index),
                    metrics["precision"][class_index],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
                self.log(
                    "{}/class_{}_recall".format(prefix, class_index),
                    metrics["recall"][class_index],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )

    def on_validation_epoch_end(self):
        if self.val_confusion is None or self.val_head_confusion is None:
            return
        total_confusion = self._gather_confusion(self.val_confusion)
        per_head_confusion = self._gather_confusion(self.val_head_confusion)
        self._log_confusion_metrics("val", total_confusion, per_head_confusion, log_class_metrics=False)

    def on_test_epoch_end(self):
        total_confusion = self._gather_confusion(self.test_confusion)
        per_head_confusion = self._gather_confusion(self.test_head_confusion)
        self._log_confusion_metrics("test", total_confusion, per_head_confusion, log_class_metrics=True)
