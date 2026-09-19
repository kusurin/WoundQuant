"""Pixel-weighted metrics on hard, uniformly distributed class sets."""

import torch

from metrics import SoftSegmentationMetrics, per_image_class_metrics


class DiscreteSegmentationMetrics(SoftSegmentationMetrics):
    def reset(self):
        super().reset()
        options = {"device": self.accumulator_device, "dtype": self.accumulator_dtype}
        count_options = {"device": self.accumulator_device, "dtype": torch.int64}
        self.absolute_error = torch.zeros(self.num_classes, **options)
        self.symmetric_percentage_error = torch.zeros(self.num_classes, **options)
        self.exact_count = torch.zeros((), **count_options)
        self.overlap_count = torch.zeros((), **count_options)
        self.overlap_absolute_error = torch.zeros((), **options)
        self.overlap_symmetric_percentage_error = torch.zeros((), **options)
        self.overlap_exact_count = torch.zeros((), **count_options)
        self.foreground_count = torch.zeros((), **count_options)
        self.foreground_absolute_error = torch.zeros((), **options)
        self.foreground_symmetric_percentage_error = torch.zeros((), **options)
        self.member_intersection = torch.zeros(self.num_classes, **options)
        self.member_denominator = torch.zeros(self.num_classes, **options)

    @torch.no_grad()
    def update(self, prediction, target):
        super().update(prediction, target)
        p = prediction.detach().to(
            device=self.accumulator_device, dtype=self.accumulator_dtype
        )
        y = target.detach().to(
            device=self.accumulator_device, dtype=self.accumulator_dtype
        )
        error = (p - y).abs()
        denominator = p.abs() + y.abs()
        symmetric_percentage_error = torch.where(
            denominator > self.epsilon,
            200.0 * error / denominator.clamp_min(self.epsilon),
            torch.zeros_like(error),
        )
        self.absolute_error += error.sum((0, 2, 3))
        self.symmetric_percentage_error += symmetric_percentage_error.sum(
            (0, 2, 3)
        )
        pm, ym = p > 0, y > 0
        exact = (pm == ym).all(1)
        overlap = ym.sum(1) > 1
        foreground = ym[:, 1:].any(1)
        self.exact_count += exact.sum()
        self.overlap_count += overlap.sum()
        self.overlap_exact_count += (exact & overlap).sum()
        self.overlap_absolute_error += error.sum(1)[overlap].sum()
        self.overlap_symmetric_percentage_error += (
            symmetric_percentage_error.sum(1)[overlap].sum()
        )
        self.foreground_count += foreground.sum()
        self.foreground_absolute_error += error.sum(1)[foreground].sum()
        self.foreground_symmetric_percentage_error += (
            symmetric_percentage_error.sum(1)[foreground].sum()
        )
        self.member_intersection += (pm & ym).sum((0, 2, 3))
        self.member_denominator += pm.sum((0, 2, 3)) + ym.sum((0, 2, 3))

    def compute(self):
        result = super().compute()
        result["clipped_mkl"] = result.pop("mkl")
        overlap_count = int(self.overlap_count)
        foreground_count = int(self.foreground_count)
        membership_dice = torch.where(
            self.member_denominator > 0,
            2 * self.member_intersection / self.member_denominator.clamp_min(1),
            torch.full_like(self.member_denominator, float("nan")),
        ).tolist()
        result.update({
            "probability_mae": float(self.absolute_error.sum()) / (self.pixel_count * self.num_classes),
            "probability_mae_per_class": (self.absolute_error / self.pixel_count).tolist(),
            "probability_smape_percent": float(
                self.symmetric_percentage_error.sum()
            ) / (self.pixel_count * self.num_classes),
            "probability_smape_per_class_percent": (
                self.symmetric_percentage_error / self.pixel_count
            ).tolist(),
            "set_exact_match": float(self.exact_count) / self.pixel_count,
            "overlap_pixel_count": overlap_count,
            "overlap_probability_mae": float(self.overlap_absolute_error) / (overlap_count * self.num_classes) if overlap_count else None,
            "overlap_probability_smape_percent": (
                float(self.overlap_symmetric_percentage_error)
                / (overlap_count * self.num_classes)
                if overlap_count
                else None
            ),
            "overlap_set_exact_match": float(self.overlap_exact_count) / overlap_count if overlap_count else None,
            "foreground_pixel_count": foreground_count,
            "foreground_probability_mae": float(self.foreground_absolute_error) / (foreground_count * self.num_classes) if foreground_count else None,
            "foreground_probability_smape_percent": (
                float(self.foreground_symmetric_percentage_error)
                / (foreground_count * self.num_classes)
                if foreground_count
                else None
            ),
            "membership_dice_per_class": [None if value != value else value for value in membership_dice],
        })
        return result


def per_image_discrete_metrics(prediction, target, **kwargs):
    rows = per_image_class_metrics(prediction, target, **kwargs)
    num_classes = prediction.shape[1]
    for index in range(len(prediction)):
        accumulator = DiscreteSegmentationMetrics(num_classes)
        accumulator.update(prediction[index:index + 1], target[index:index + 1])
        metrics = accumulator.compute()
        for channel in range(num_classes):
            row = rows[index * num_classes + channel]
            row["image_clipped_mkl"] = row.pop("image_mkl")
            row["probability_mae"] = metrics["probability_mae_per_class"][channel]
            row["probability_smape_percent"] = metrics[
                "probability_smape_per_class_percent"
            ][channel]
            row["membership_dice"] = metrics["membership_dice_per_class"][channel]
            for key in (
                "probability_mae",
                "probability_smape_percent",
                "set_exact_match",
                "overlap_probability_mae",
                "overlap_probability_smape_percent",
                "overlap_set_exact_match",
                "overlap_pixel_count",
                "foreground_probability_mae",
                "foreground_probability_smape_percent",
            ):
                row[f"image_{key}"] = metrics[key]
    return rows
