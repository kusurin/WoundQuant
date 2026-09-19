"""Dataset-level metrics for soft semantic segmentation."""

from __future__ import annotations

import json
import math
from typing import Any, Sequence

import torch


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _macro(values: Sequence[float]) -> float | None:
    valid = [value for value in values if math.isfinite(value)]
    return sum(valid) / len(valid) if valid else None


class SoftSegmentationMetrics:
    """Accumulate sufficient statistics and compute once per dataset."""

    def __init__(
        self,
        num_classes: int,
        *,
        epsilon: float = 1e-6,
        accumulator_device: torch.device | str = "cpu",
        accumulator_dtype: torch.dtype | None = None,
    ) -> None:
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.accumulator_device = torch.device(accumulator_device)
        self.accumulator_dtype = accumulator_dtype or (
            torch.float32 if self.accumulator_device.type == "cuda" else torch.float64
        )
        if not self.accumulator_dtype.is_floating_point:
            raise ValueError("accumulator_dtype must be floating point")
        self.reset()

    def reset(self) -> None:
        shape = (self.num_classes,)
        options = {"device": self.accumulator_device, "dtype": self.accumulator_dtype}
        self.intersection = torch.zeros(shape, **options)
        self.prediction_square = torch.zeros(shape, **options)
        self.target_square = torch.zeros(shape, **options)
        self.prediction_mass = torch.zeros(shape, **options)
        self.target_mass = torch.zeros(shape, **options)
        self.per_image_rae_sum = torch.zeros(shape, **options)
        self.per_image_rae_count = torch.zeros(
            shape, device=self.accumulator_device, dtype=torch.int64
        )
        self.per_image_area_smape_sum = torch.zeros(shape, **options)
        self.per_image_area_smape_count = torch.zeros(
            shape, device=self.accumulator_device, dtype=torch.int64
        )
        self.kl_sum = torch.zeros((), **options)
        self.brier_sum = torch.zeros((), **options)
        self.pixel_count = 0
        self.image_count = 0

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError("prediction and target must have shape (B, K, H, W)")
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: "
                f"{prediction.shape} vs {target.shape}"
            )
        if prediction.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} channels, got {prediction.shape[1]}"
            )
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("prediction and target must be finite")
        tolerance = 1e-4
        if torch.any(prediction < -tolerance) or torch.any(prediction > 1 + tolerance):
            raise ValueError("prediction must be probabilities in [0, 1]")
        if torch.any(target < -tolerance) or torch.any(target > 1 + tolerance):
            raise ValueError("target must be probabilities in [0, 1]")
        if not torch.allclose(
            prediction.sum(dim=1),
            torch.ones_like(prediction[:, 0]),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("prediction is not normalized across channels")
        if not torch.allclose(
            target.sum(dim=1),
            torch.ones_like(target[:, 0]),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("target is not normalized across channels")

        prediction = prediction.detach().to(
            device=self.accumulator_device, dtype=self.accumulator_dtype
        )
        target = target.detach().to(
            device=self.accumulator_device, dtype=self.accumulator_dtype
        )
        dimensions = (0, 2, 3)
        self.intersection += (prediction * target).sum(dim=dimensions)
        self.prediction_square += prediction.square().sum(dim=dimensions)
        self.target_square += target.square().sum(dim=dimensions)
        self.prediction_mass += prediction.sum(dim=dimensions)
        self.target_mass += target.sum(dim=dimensions)

        safe_prediction = prediction.clamp_min(self.epsilon)
        safe_target = target.clamp_min(self.epsilon)
        kl_terms = torch.where(
            target > 0,
            target * (safe_target.log() - safe_prediction.log()),
            torch.zeros_like(target),
        )
        self.kl_sum += kl_terms.sum()
        self.brier_sum += (prediction - target).square().sum()

        true_area = target.sum(dim=(2, 3))
        predicted_area = prediction.sum(dim=(2, 3))
        valid = true_area > self.epsilon
        per_image_rae = (
            (predicted_area - true_area).abs()
            / true_area.clamp_min(self.epsilon)
            * 100.0
        )
        self.per_image_rae_sum += torch.where(
            valid, per_image_rae, torch.zeros_like(per_image_rae)
        ).sum(dim=0)
        self.per_image_rae_count += valid.sum(dim=0)

        area_smape_denominator = predicted_area.abs() + true_area.abs()
        valid_area_smape = area_smape_denominator > self.epsilon
        per_image_area_smape = (
            200.0
            * (predicted_area - true_area).abs()
            / area_smape_denominator.clamp_min(self.epsilon)
        )
        self.per_image_area_smape_sum += torch.where(
            valid_area_smape,
            per_image_area_smape,
            torch.zeros_like(per_image_area_smape),
        ).sum(dim=0)
        self.per_image_area_smape_count += valid_area_smape.sum(dim=0)

        batch, _, height, width = prediction.shape
        self.pixel_count += batch * height * width
        self.image_count += batch

    def compute(self) -> dict[str, Any]:
        if self.pixel_count == 0:
            raise RuntimeError("No samples have been accumulated")

        dice_denominator = self.prediction_square + self.target_square
        iou_denominator = (
            self.prediction_mass + self.target_mass - self.intersection
        )
        present = (self.prediction_mass + self.target_mass) > self.epsilon

        options = {"device": self.accumulator_device, "dtype": self.accumulator_dtype}
        dice = torch.full((self.num_classes,), float("nan"), **options)
        iou = torch.full((self.num_classes,), float("nan"), **options)
        dice[present] = (
            2.0 * self.intersection[present] + self.epsilon
        ) / (dice_denominator[present] + self.epsilon)
        iou[present] = (
            self.intersection[present] + self.epsilon
        ) / (iou_denominator[present] + self.epsilon)

        area_absolute_error = (self.prediction_mass - self.target_mass).abs()
        area_relative_error = (
            area_absolute_error / (self.target_mass + self.epsilon) * 100.0
        )
        per_image_rae_mean = torch.full((self.num_classes,), float("nan"), **options)
        valid_rae = self.per_image_rae_count > 0
        per_image_rae_mean[valid_rae] = (
            self.per_image_rae_sum[valid_rae]
            / self.per_image_rae_count[valid_rae]
        )
        per_image_area_smape_mean = torch.full(
            (self.num_classes,), float("nan"), **options
        )
        valid_area_smape = self.per_image_area_smape_count > 0
        per_image_area_smape_mean[valid_area_smape] = (
            self.per_image_area_smape_sum[valid_area_smape]
            / self.per_image_area_smape_count[valid_area_smape]
        )

        dice_values = dice.tolist()
        iou_values = iou.tolist()
        return {
            "soft_dice_per_class": [_finite_or_none(value) for value in dice_values],
            "soft_iou_per_class": [_finite_or_none(value) for value in iou_values],
            "soft_dice_background": _finite_or_none(dice_values[0]),
            "soft_iou_background": _finite_or_none(iou_values[0]),
            "soft_dice_foreground_macro": _macro(dice_values[1:]),
            "soft_iou_foreground_macro": _macro(iou_values[1:]),
            "mkl": float(self.kl_sum) / self.pixel_count,
            "brier": float(self.brier_sum) / self.pixel_count,
            "brier_mse": float(self.brier_sum) / (self.pixel_count * self.num_classes),
            "area_true_per_class": self.target_mass.tolist(),
            "area_pred_per_class": self.prediction_mass.tolist(),
            "aae_per_class": area_absolute_error.tolist(),
            "rae_dataset_per_class_percent": area_relative_error.tolist(),
            "rae_per_image_mean_percent": [
                _finite_or_none(value) for value in per_image_rae_mean.tolist()
            ],
            "rae_per_image_valid_count": self.per_image_rae_count.tolist(),
            "area_smape_per_image_mean_percent": [
                _finite_or_none(value)
                for value in per_image_area_smape_mean.tolist()
            ],
            "area_smape_per_image_valid_count": (
                self.per_image_area_smape_count.tolist()
            ),
            "pixel_count": self.pixel_count,
            "image_count": self.image_count,
        }


def metrics_with_class_names(
    metrics: dict[str, Any], class_names: Sequence[str]
) -> dict[str, Any]:
    """Add readable per-class records without discarding raw aggregates."""

    per_class = []
    for index, name in enumerate(class_names):
        per_class.append(
            {
                "channel": index,
                "name": name,
                "soft_dice": metrics["soft_dice_per_class"][index],
                "soft_iou": metrics["soft_iou_per_class"][index],
                "area_true": metrics["area_true_per_class"][index],
                "area_pred": metrics["area_pred_per_class"][index],
                "aae": metrics["aae_per_class"][index],
                "rae_dataset_percent": metrics[
                    "rae_dataset_per_class_percent"
                ][index],
                "rae_per_image_mean_percent": metrics[
                    "rae_per_image_mean_percent"
                ][index],
                "rae_per_image_valid_count": metrics[
                    "rae_per_image_valid_count"
                ][index],
                "area_smape_per_image_mean_percent": metrics[
                    "area_smape_per_image_mean_percent"
                ][index],
                "area_smape_per_image_valid_count": metrics[
                    "area_smape_per_image_valid_count"
                ][index],
            }
        )
        if "probability_smape_per_class_percent" in metrics:
            per_class[-1]["probability_smape_percent"] = metrics[
                "probability_smape_per_class_percent"
            ][index]
    result = dict(metrics)
    result["per_class"] = per_class
    return result


@torch.no_grad()
def per_image_class_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    class_names: Sequence[str],
    image_ids: Sequence[int],
    file_names: Sequence[str],
    epsilon: float = 1e-6,
) -> list[dict[str, Any]]:
    """Return one serializable metric row per image and semantic class."""

    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("prediction and target must have shape (B, K, H, W)")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    batch_size, num_classes, height, width = prediction.shape
    if len(class_names) != num_classes:
        raise ValueError("class_names length must match channel count")
    if len(image_ids) != batch_size or len(file_names) != batch_size:
        raise ValueError("image ids and file names must match batch size")

    prediction = prediction.detach().to(device="cpu", dtype=torch.float64)
    target = target.detach().to(device="cpu", dtype=torch.float64)
    rows: list[dict[str, Any]] = []
    for batch_index in range(batch_size):
        predicted_image = prediction[batch_index]
        target_image = target[batch_index]
        safe_prediction = predicted_image.clamp_min(epsilon)
        safe_target = target_image.clamp_min(epsilon)
        kl = torch.where(
            target_image > 0,
            target_image * (safe_target.log() - safe_prediction.log()),
            torch.zeros_like(target_image),
        ).sum() / (height * width)
        brier = (
            (predicted_image - target_image).square().sum() / (height * width)
        )

        for channel, class_name in enumerate(class_names):
            predicted_class = predicted_image[channel]
            target_class = target_image[channel]
            intersection = (predicted_class * target_class).sum()
            predicted_mass = predicted_class.sum()
            target_mass = target_class.sum()
            dice_denominator = (
                predicted_class.square().sum() + target_class.square().sum()
            )
            iou_denominator = predicted_mass + target_mass - intersection
            present = float(predicted_mass + target_mass) > epsilon
            soft_dice = (
                float((2.0 * intersection + epsilon) / (dice_denominator + epsilon))
                if present
                else None
            )
            soft_iou = (
                float((intersection + epsilon) / (iou_denominator + epsilon))
                if present
                else None
            )
            absolute_error = float((predicted_mass - target_mass).abs())
            relative_error = (
                absolute_error / (float(target_mass) + epsilon) * 100.0
            )
            area_smape_denominator = float(predicted_mass + target_mass)
            area_smape = (
                200.0 * absolute_error / area_smape_denominator
                if area_smape_denominator > epsilon
                else None
            )
            rows.append(
                {
                    "image_id": int(image_ids[batch_index]),
                    "file_name": str(file_names[batch_index]),
                    "height": height,
                    "width": width,
                    "channel": channel,
                    "class_name": str(class_name),
                    "soft_dice": soft_dice,
                    "soft_iou": soft_iou,
                    "area_true": float(target_mass),
                    "area_pred": float(predicted_mass),
                    "aae": absolute_error,
                    "rae_percent": relative_error,
                    "area_smape_percent": area_smape,
                    "image_mkl": float(kl),
                    "image_brier": float(brier),
                }
            )
    return rows


def format_metrics(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)
