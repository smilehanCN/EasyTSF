from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from .grid3d_forecasting import Grid3DForecastingTask


_HEAD_NAMES = ("shear_x", "shear_y", "shear_z", "speed_cls")
_RISK_BIN_KEYS = ("shear_x", "shear_y", "shear_z", "speed")


class Grid3DRiskPredictionTask(Grid3DForecastingTask):
    def _setup_task_state(self):
        super()._setup_task_state()
        self.output_mode = str(getattr(self.hparams, "output_mode", "classification"))
        if self.output_mode != "classification":
            raise ValueError("grid3d_risk_prediction requires output_mode='classification'")

        self.risk_num_classes = int(getattr(self.hparams, "risk_num_classes", 3))
        self.risk_num_heads = int(getattr(self.hparams, "risk_num_heads", 4))
        if self.risk_num_classes != 3:
            raise ValueError("grid3d_risk_prediction expects risk_num_classes=3")
        if self.risk_num_heads != len(_HEAD_NAMES):
            raise ValueError(
                "grid3d_risk_prediction expects risk_num_heads={}, got {}".format(
                    len(_HEAD_NAMES),
                    self.risk_num_heads,
                )
            )

        self.risk_bins = self._parse_risk_bins(getattr(self.hparams, "risk_bins", None))
        self.head_loss_weights = self._parse_head_loss_weights(getattr(self.hparams, "head_loss_weights", None))
        self.risk_class_weights = self._parse_risk_class_weights(getattr(self.hparams, "risk_class_weights", None))

        self.high_risk_class_index = int(getattr(self.hparams, "high_risk_class_index", 2))
        if self.high_risk_class_index < 0 or self.high_risk_class_index >= self.risk_num_classes:
            raise ValueError(
                "high_risk_class_index {} is out of range for risk_num_classes={}".format(
                    self.high_risk_class_index,
                    self.risk_num_classes,
                )
            )

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
        parsed = {}
        for key in _RISK_BIN_KEYS:
            if key not in raw_bins:
                raise ValueError("risk_bins must define '{}'".format(key))
            values = raw_bins[key]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 2:
                raise ValueError("risk_bins['{}'] must be a 2-item list/tuple".format(key))
            first, second = float(values[0]), float(values[1])
            if not first < second:
                raise ValueError(
                    "risk_bins['{}'] must be strictly increasing, got [{}, {}]".format(key, first, second)
                )
            parsed[key] = (first, second)
        return parsed

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

    def _bucketize_three_classes(self, values: torch.Tensor, thresholds: tuple[float, float]) -> torch.Tensor:
        boundaries = values.new_tensor(thresholds)
        return torch.bucketize(values, boundaries=boundaries, right=False).long()

    def _compute_vector_shear(self, physical_targets: torch.Tensor, spatial_dim: int) -> torch.Tensor:
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
        magnitude = torch.linalg.vector_norm(vector_delta, ord=2, dim=2)

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
        shear_x = self._compute_vector_shear(physical_targets, spatial_dim=4)
        shear_y = self._compute_vector_shear(physical_targets, spatial_dim=3)
        shear_z = self._compute_vector_shear(physical_targets, spatial_dim=5)

        labels = torch.stack(
            [
                self._bucketize_three_classes(shear_x, self.risk_bins["shear_x"]),
                self._bucketize_three_classes(shear_y, self.risk_bins["shear_y"]),
                self._bucketize_three_classes(shear_z, self.risk_bins["shear_z"]),
                self._bucketize_three_classes(speed, self.risk_bins["speed"]),
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

    def on_test_epoch_start(self):
        self.test_confusion = torch.zeros(
            self.risk_num_classes,
            self.risk_num_classes,
            dtype=torch.long,
            device=self.device,
        )
        self.test_head_confusion = torch.zeros(
            self.risk_num_heads,
            self.risk_num_classes,
            self.risk_num_classes,
            dtype=torch.long,
            device=self.device,
        )

    def _update_confusion(self, predictions: torch.Tensor, labels: torch.Tensor):
        for head_index in range(self.risk_num_heads):
            head_predictions = predictions[:, :, head_index, ...].reshape(-1).to(dtype=torch.long)
            head_labels = labels[:, :, head_index, ...].reshape(-1).to(dtype=torch.long)
            packed = head_labels * self.risk_num_classes + head_predictions
            confusion = torch.bincount(
                packed,
                minlength=self.risk_num_classes * self.risk_num_classes,
            ).reshape(self.risk_num_classes, self.risk_num_classes)
            self.test_confusion = self.test_confusion + confusion
            self.test_head_confusion[head_index] = self.test_head_confusion[head_index] + confusion

    def test_step(self, batch, batch_idx):
        del batch_idx
        prediction, label = self._forward(batch)
        for pred_crop, label_crop in self._iter_eval_crops(batch, prediction, label):
            logits = self._reshape_logits(pred_crop)
            predicted_classes = logits.argmax(dim=3)
            self._update_confusion(predicted_classes, label_crop)

    def _gather_confusion(self, confusion: torch.Tensor) -> torch.Tensor:
        gathered = self.all_gather(confusion)
        if gathered.ndim == confusion.ndim + 1:
            return gathered.sum(dim=0)
        return gathered

    def _compute_confusion_metrics(self, confusion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        confusion = confusion.to(dtype=torch.float32)
        true_positive = confusion.diag()
        predicted_positive = confusion.sum(dim=0)
        actual_positive = confusion.sum(dim=1)

        precision = torch.where(predicted_positive > 0, true_positive / predicted_positive, torch.zeros_like(true_positive))
        recall = torch.where(actual_positive > 0, true_positive / actual_positive, torch.zeros_like(true_positive))
        f1 = torch.where(precision + recall > 0, 2.0 * precision * recall / (precision + recall), torch.zeros_like(precision))
        macro_f1 = f1.mean()
        high_risk_recall = recall[self.high_risk_class_index]
        return macro_f1, high_risk_recall

    def on_test_epoch_end(self):
        total_confusion = self._gather_confusion(self.test_confusion)
        per_head_confusion = self._gather_confusion(self.test_head_confusion)

        macro_f1, high_risk_recall = self._compute_confusion_metrics(total_confusion)
        self.log("test/macro_f1", macro_f1, on_step=False, on_epoch=True, sync_dist=False)
        self.log("test/high_risk_recall", high_risk_recall, on_step=False, on_epoch=True, sync_dist=False)

        for head_index, head_name in enumerate(_HEAD_NAMES):
            head_macro_f1, _ = self._compute_confusion_metrics(per_head_confusion[head_index])
            self.log(
                "test/{}_macro_f1".format(head_name),
                head_macro_f1,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
