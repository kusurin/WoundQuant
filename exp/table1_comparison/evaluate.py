"""Export Table 1 predictions once and derive reusable area-error tables.

Predictions and ground truth are cached as full-resolution bit-coded PNG masks.
Use ``--summary-only`` to rebuild the combined CSV tables without loading models,
running inference, or decoding the COCO annotations again.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


from _common import settings as s
ROOT = s.REPO
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import metrics_with_class_names
from independent_sigmoid.train import (
    load_inference_model as load_independent_sigmoid_model,
    predict_image as predict_independent_sigmoid_image,
)
from table1_fcn import load_table1_fcn, predict_table1_fcn
from seg_discrete_overlap.discrete_metrics import DiscreteSegmentationMetrics
from seg_discrete_overlap.train_eval import (
    CocoFullResolutionSegmentationEvaluationDataset,
    load_inference_model,
    normalise_image,
    predict_image,
    resolve_device,
)


DEFAULT_ANNOTATIONS = s.data_path("annotations")
DEFAULT_IMAGES = s.data_path("images")
RULER_GT = s.data_path("ruler")
RULER_PREDICTIONS = s.output() / "ruler" / "predictions.csv"
TABLE_OUTPUT = s.output()
GROUND_TRUTH_CODE_CACHE = (
    s.output() / "cache" / "reference_codes"
)
OURS_OUTPUT = s.output() / "evaluation" / "ours"
INDEPENDENT_SIGMOID_OUTPUT = s.output() / "evaluation" / "independent_sigmoid"
FCN_OUTPUTS = {
    "fcn_single_label": s.output() / "evaluation" / "fcn_single_label",
    "fcn_lookup_multilabel": s.output() / "evaluation" / "fcn_lookup_multilabel",
}
ABLATIONS = {
    "without_expected_l1": s.output() / "evaluation" / "without_expected_l1",
    "without_dice": s.output() / "evaluation" / "without_dice",
    "without_focal": s.output() / "evaluation" / "without_focal",
}
BASELINE_CHECKPOINT = s.path(s.current()["models"]["single_label_unetplusplus"]["checkpoint"])


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fixed_test_ids() -> list[int]:
    split = json.loads(s.data_path("split").read_text(encoding="utf-8"))
    return [int(value) for value in split["splits"]["test"]["image_ids"]]


def code_cache_path(root: Path, image_id: int) -> Path:
    return root / f"{image_id:06d}.png"


def save_probability_code(probability: torch.Tensor, path: Path) -> None:
    array = probability.detach().cpu().numpy()[0]
    code = np.zeros(array.shape[1:], dtype=np.uint8)
    for channel in range(array.shape[0]):
        code |= ((array[channel] > 0).astype(np.uint8) << channel)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(code).save(path, optimize=True)


def load_probability_code(path: Path, num_classes: int = 4) -> torch.Tensor:
    code = np.asarray(Image.open(path), dtype=np.uint8)
    members = np.stack(
        [((code >> channel) & 1).astype(np.float32) for channel in range(num_classes)]
    )
    count = members.sum(axis=0, keepdims=True)
    if np.any(count == 0):
        raise RuntimeError(f"Invalid zero-member prediction code in {path}")
    return torch.from_numpy(members / count).unsqueeze(0)


def cached_or_save_target(sample: dict) -> None:
    path = code_cache_path(GROUND_TRUTH_CODE_CACHE, int(sample["image_id"]))
    save_probability_code(sample["target"].unsqueeze(0), path)


@lru_cache(maxsize=1)
def area_scales() -> tuple[dict[str, float], dict[int, float]]:
    manual_frame = pd.read_csv(RULER_GT, sep="\t")
    manual = {
        str(row["img"]).casefold(): float(row["ruler pixel/cm"])
        for _, row in manual_frame.iterrows()
    }
    predicted_frame = pd.read_csv(RULER_PREDICTIONS)
    predicted = {
        int(row.image_id): float(row.predicted_pixel_per_cm)
        for _, row in predicted_frame.iterrows()
        if row.status == "valid" and float(row.predicted_pixel_per_cm) > 0
    }
    return manual, predicted


def image_area_rows(method: str, sample: dict, prediction: torch.Tensor) -> list[dict]:
    manual, predicted = area_scales()
    image_id = int(sample["image_id"])
    file_name = str(sample["file_name"])
    manual_scale = manual.get(file_name.casefold())
    predicted_scale = predicted.get(image_id)
    if manual_scale is None or manual_scale <= 0 or predicted_scale is None:
        raise RuntimeError(f"Missing valid area scale for test image {image_id}: {file_name}")
    prediction_array = prediction.detach().cpu().numpy()[0]
    target_array = sample["target"].detach().cpu().numpy()
    names = list(sample.get("class_names", ("normal", "eschar", "granulation", "suppuration")))
    rows = []
    selections = [(channel, names[channel]) for channel in range(1, len(names))]
    selections.append((-1, "total_foreground"))
    for channel, class_name in selections:
        if channel < 0:
            predicted_pixels = float(prediction_array[1:].sum())
            target_pixels = float(target_array[1:].sum())
        else:
            predicted_pixels = float(prediction_array[channel].sum())
            target_pixels = float(target_array[channel].sum())
        predicted_area = predicted_pixels / predicted_scale**2
        target_area = target_pixels / manual_scale**2
        difference = predicted_area - target_area
        absolute = abs(difference)
        denominator = predicted_area + target_area
        relative_defined = target_area > 0
        rows.append({
            "method": method,
            "image_id": image_id,
            "file_name": file_name,
            "channel": channel,
            "class_name": class_name,
            "target_effective_pixels": target_pixels,
            "prediction_effective_pixels": predicted_pixels,
            "manual_pixel_per_cm": manual_scale,
            "predicted_pixel_per_cm": predicted_scale,
            "gt_area_cm2": target_area,
            "predicted_area_cm2": predicted_area,
            "signed_error_cm2": difference,
            "absolute_error_cm2": absolute,
            "signed_relative_error_percent": 100 * difference / target_area if relative_defined else 0.0,
            "absolute_relative_error_percent": 100 * absolute / target_area if relative_defined else 0.0,
            "relative_error_status": "defined" if relative_defined else "gt_zero_defined_as_zero",
            "smape_percent": 200 * absolute / denominator if denominator > 0 else 0.0,
            "smape_status": "defined" if denominator > 0 else "both_zero_defined_as_zero",
        })
    return rows


def save_area_rows(output: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "table1_per_image_class_area_errors.csv", index=False, encoding="utf-8-sig")


def completed_training_epochs(method: str, output: Path) -> int | None:
    configured = s.current()["models"][method].get("training_history")
    history_path = s.path(configured) if configured else (
        s.output() / "training" / method /
        ("history.json" if method == "single_label_unetplusplus" else "segmenter/history.json")
    )
    if not history_path.is_file():
        return None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"Invalid training history: {history_path}")
    return len(history)


def table1_dice(output: Path) -> dict[str, float]:
    """Read the common total/macro and tissue-specific Dice columns."""
    path = output / "table1_test_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Table 1 Dice metrics: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    by_name = {item["name"]: item["soft_dice"] for item in metrics["per_class"]}
    required = ("eschar", "granulation", "suppuration")
    missing = [name for name in required if by_name.get(name) is None]
    if missing:
        raise RuntimeError(f"Missing tissue Dice metrics in {path}: {missing}")
    return {
        "foreground_macro_soft_dice": float(metrics["soft_dice_foreground_macro"]),
        **{f"{name}_soft_dice": float(by_name[name]) for name in required},
    }


def bootstrap_test_smape(combined: pd.DataFrame) -> dict[str, dict]:
    """Percentile CI for mean SMAPE, resampling fixed test images (not pixels).

    All models use identical image draws; model fitting is held fixed.
    """
    ids = sorted(fixed_test_ids())
    n_resamples = int(s.current()["bootstrap"])
    seed = int(s.current()["bootstrap_seed"])
    indices = np.random.default_rng(seed).integers(0, len(ids), (n_resamples, len(ids)))
    results = {}
    foreground = combined[combined.class_name == "total_foreground"]
    for method, group in foreground.groupby("method", sort=False):
        if group.image_id.duplicated().any() or set(group.image_id) != set(ids):
            raise ValueError(f"{method}: bootstrap requires exactly one row per fixed test image")
        values = group.set_index("image_id").loc[ids, "smape_percent"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{method}: non-finite test SMAPE")
        means = values[indices].mean(axis=1)
        lower, upper = np.percentile(means, [2.5, 97.5])
        results[method] = {
            "smape_percent_ci95_lower": float(lower),
            "smape_percent_ci95_upper": float(upper),
            "smape_bootstrap_n_images": len(ids),
            "smape_bootstrap_n_resamples": n_resamples,
            "smape_bootstrap_seed": seed,
        }
    return results


def combine_area_outputs() -> None:
    outputs = {
        "single_label_unetplusplus": s.output() / "evaluation" / "single_label_unetplusplus",
        **FCN_OUTPUTS,
        "independent_sigmoid": INDEPENDENT_SIGMOID_OUTPUT,
        **ABLATIONS,
        "ours": OURS_OUTPUT,
    }
    outputs = {
        method: output
        for method, output in outputs.items()
        if (output / "table1_per_image_class_area_errors.csv").is_file()
    }
    frames = []
    for output in outputs.values():
        path = output / "table1_per_image_class_area_errors.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No evaluation tables; run this group with evaluate first.")
    combined = pd.concat(frames, ignore_index=True)
    gt_zero = combined.gt_area_cm2 == 0
    combined.loc[gt_zero, "signed_relative_error_percent"] = 0.0
    combined.loc[gt_zero, "absolute_relative_error_percent"] = 0.0
    combined.loc[gt_zero, "relative_error_status"] = "gt_zero_defined_as_zero"
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        combined.to_csv(
            TABLE_OUTPUT / "06.02_table1_per_image_class_area_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )
    except PermissionError:
        print("Warning: 06.02 detail CSV is open; skipped replacing it.")
    summaries = []
    for (method, class_name), group in combined.groupby(["method", "class_name"], sort=False):
        gt_sum = float(group.gt_area_cm2.sum())
        prediction_sum = float(group.predicted_area_cm2.sum())
        summaries.append({
            "method": method,
            "class_name": class_name,
            "n_images": int(len(group)),
            "n_relative_error_defined": int(len(group)),
            "n_gt_zero": int((group.gt_area_cm2 == 0).sum()),
            "n_both_zero": int((group.smape_status == "both_zero_defined_as_zero").sum()),
            "total_gt_area_cm2": gt_sum,
            "total_predicted_area_cm2": prediction_sum,
            "aggregate_signed_relative_error_percent": 100 * (prediction_sum - gt_sum) / gt_sum if gt_sum > 0 else np.nan,
            "aggregate_absolute_relative_error_percent": 100 * abs(prediction_sum - gt_sum) / gt_sum if gt_sum > 0 else np.nan,
            "mean_absolute_error_cm2": float(group.absolute_error_cm2.mean()),
            "median_absolute_error_cm2": float(group.absolute_error_cm2.median()),
            "mean_absolute_relative_error_percent": float(group.absolute_relative_error_percent.mean()),
            "median_absolute_relative_error_percent": float(group.absolute_relative_error_percent.median()),
            "mean_smape_percent": float(group.smape_percent.mean()),
            "median_smape_percent": float(group.smape_percent.median()),
        })
    summary_frame = pd.DataFrame(summaries)
    try:
        summary_frame.to_csv(
            TABLE_OUTPUT / "06.03_table1_area_error_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    except PermissionError:
        print("Warning: 06.03 summary CSV is open; skipped replacing it.")

    # One manuscript-facing row per model. RAE is the mean per-image absolute
    # relative area error (GT-zero images count as zero); AAE is the mean per-image
    # absolute physical-area error. SMAPE is averaged over all test images.
    metric_columns = {
        "mean_rae_per_image_percent": "mean_absolute_relative_error_percent",
        "mean_aae_per_image_cm2": "mean_absolute_error_cm2",
        "mean_smape_per_image_percent": "mean_smape_percent",
    }
    bootstrap = bootstrap_test_smape(combined)
    bootstrap_columns = list(next(iter(bootstrap.values())))
    compact_rows = []
    class_names = ["total_foreground", "eschar", "granulation", "suppuration"]
    for method, output in outputs.items():
        method_summary = summary_frame[summary_frame.method == method].set_index("class_name")
        missing = [name for name in class_names if name not in method_summary.index]
        if missing:
            raise RuntimeError(f"Missing area summaries for {method}: {missing}")
        row = {
            "method": method,
            "total_training_epochs": completed_training_epochs(method, output),
            **table1_dice(output),
            **bootstrap[method],
        }
        for class_name in class_names:
            for output_name, source_name in metric_columns.items():
                row[f"{class_name}_{output_name}"] = float(
                    method_summary.loc[class_name, source_name]
                )
        tissue_rows = method_summary.loc[["eschar", "granulation", "suppuration"]]
        for output_name, source_name in metric_columns.items():
            row[f"tissue_macro_{output_name}"] = float(tissue_rows[source_name].mean())
        compact_rows.append(row)
    compact_columns = [
        "method",
        "total_training_epochs",
        "foreground_macro_soft_dice",
        "eschar_soft_dice",
        "granulation_soft_dice",
        "suppuration_soft_dice",
        *[
            f"{prefix}_{metric}"
            for prefix in [
                "total_foreground",
                "tissue_macro",
                "eschar",
                "granulation",
                "suppuration",
            ]
            for metric in metric_columns
        ],
    ]
    compact_columns.extend(bootstrap_columns)
    compact_frame = pd.DataFrame(compact_rows)[compact_columns]
    compact_frame.to_csv(
        TABLE_OUTPUT / "06.04_table1_foreground_area_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    display_names = {
        "single_label_unetplusplus": "Single-label UNet++",
        "fcn_single_label": "Dice-Focal FCN (single-label)",
        "fcn_lookup_multilabel": "Dice-Focal FCN (lookup multilabel)",
        "independent_sigmoid": "Independent Sigmoid",
        "without_expected_l1": "Without expected L1",
        "without_dice": "Without Dice",
        "without_focal": "Without focal loss",
        "ours": "Ours",
    }
    table1 = compact_frame[
        [
            "method",
            "total_training_epochs",
            "foreground_macro_soft_dice",
            "eschar_soft_dice",
            "granulation_soft_dice",
            "suppuration_soft_dice",
            "total_foreground_mean_smape_per_image_percent",
            *bootstrap_columns,
        ]
    ].copy()
    table1["method"] = table1["method"].map(display_names)
    table1 = table1.rename(
        columns={
            "total_foreground_mean_smape_per_image_percent": "smape_percent",
            "foreground_macro_soft_dice": "mean_dice",
            "eschar_soft_dice": "eschar_dice",
            "granulation_soft_dice": "granulation_dice",
            "suppuration_soft_dice": "suppuration_dice",
        }
    )
    table_path = TABLE_OUTPUT / "06.01_table1_model_comparison.csv"
    try:
        table1.to_csv(table_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        pending = table_path.with_name(f"{table_path.stem}.pending.csv")
        table1.to_csv(pending, index=False, encoding="utf-8-sig")
        print(f"Warning: {table_path.name} is open; wrote {pending.name} instead.")


@torch.no_grad()
def evaluate_overlap_checkpoint(name: str, output: Path, device: torch.device) -> dict:
    checkpoint = s.path(s.current()["models"][name]["checkpoint"])
    model, payload = load_inference_model(checkpoint, device)
    expected_ids = fixed_test_ids()
    saved_ids = [int(value) for value in payload["data_split"]["splits"]["test"]["image_ids"]]
    if saved_ids != expected_ids:
        raise RuntimeError(f"{name} does not use the fixed Table 1 test split")
    dataset = CocoFullResolutionSegmentationEvaluationDataset(
        DEFAULT_ANNOTATIONS,
        DEFAULT_IMAGES,
        image_ids=expected_ids,
        normal_name="normal",
    )
    accumulator = DiscreteSegmentationMetrics(dataset.mapping.num_classes)
    area_rows = []
    image_size = int(payload["args"]["image_size"])
    prediction_cache = output / "table1_prediction_codes"
    for sample in tqdm(dataset, desc=f"Table 1: {name}", leave=False):
        prediction_path = code_cache_path(prediction_cache, int(sample["image_id"]))
        if s.current().get("reuse_predictions", False) and prediction_path.is_file():
            prediction = load_probability_code(prediction_path)
        else:
            prediction = torch.from_numpy(
                predict_image(model, sample["image"], device, image_size)
            ).unsqueeze(0)
            save_probability_code(prediction, prediction_path)
        cached_or_save_target(sample)
        accumulator.update(prediction, sample["target"].unsqueeze(0))
        sample["class_names"] = dataset.mapping.channel_to_name
        area_rows.extend(image_area_rows(name, sample, prediction))
    metrics = metrics_with_class_names(accumulator.compute(), dataset.mapping.channel_to_name)
    result = {
        "method": name,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "test_image_count": len(dataset),
        "test_image_ids": expected_ids,
        "prediction_representation": "hard discrete overlap with uniform membership mass",
        "metrics": metrics,
    }
    save_json(output / "table1_test_metrics.json", result)
    save_area_rows(output, area_rows)
    return result


@torch.no_grad()
def evaluate_independent_sigmoid(device: torch.device) -> dict:
    checkpoint = s.path(s.current()["models"]["independent_sigmoid"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Independent-sigmoid checkpoint is missing; run "
            "python -m independent_sigmoid.train"
        )
    model, payload = load_independent_sigmoid_model(checkpoint, device)
    expected_ids = fixed_test_ids()
    saved_ids = [
        int(value)
        for value in payload["data_split"]["splits"]["test"]["image_ids"]
    ]
    if saved_ids != expected_ids:
        raise RuntimeError("independent_sigmoid does not use the fixed Table 1 test split")
    dataset = CocoFullResolutionSegmentationEvaluationDataset(
        DEFAULT_ANNOTATIONS,
        DEFAULT_IMAGES,
        image_ids=expected_ids,
        normal_name="normal",
    )
    accumulator = DiscreteSegmentationMetrics(dataset.mapping.num_classes)
    area_rows = []
    image_size = int(payload["args"]["image_size"])
    prediction_cache = INDEPENDENT_SIGMOID_OUTPUT / "table1_prediction_codes"
    for sample in tqdm(dataset, desc="Table 1: independent_sigmoid", leave=False):
        prediction_path = code_cache_path(prediction_cache, int(sample["image_id"]))
        if s.current().get("reuse_predictions", False) and prediction_path.is_file():
            prediction = load_probability_code(prediction_path)
        else:
            prediction = torch.from_numpy(
                predict_independent_sigmoid_image(
                    model, sample["image"], device, image_size
                )
            ).unsqueeze(0)
            save_probability_code(prediction, prediction_path)
        cached_or_save_target(sample)
        accumulator.update(prediction, sample["target"].unsqueeze(0))
        sample["class_names"] = dataset.mapping.channel_to_name
        area_rows.extend(image_area_rows("independent_sigmoid", sample, prediction))
    metrics = metrics_with_class_names(
        accumulator.compute(), dataset.mapping.channel_to_name
    )
    result = {
        "method": "independent_sigmoid",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "test_image_count": len(dataset),
        "test_image_ids": expected_ids,
        "prediction_representation": (
            "four independent sigmoid memberships, thresholded then uniformly "
            "allocated across active classes"
        ),
        "metrics": metrics,
    }
    save_json(INDEPENDENT_SIGMOID_OUTPUT / "table1_test_metrics.json", result)
    save_area_rows(INDEPENDENT_SIGMOID_OUTPUT, area_rows)
    return result


@torch.no_grad()
def evaluate_fcn(name: str, mode: str, output: Path, device: torch.device) -> dict:
    checkpoint = s.path(s.current()["models"][name]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"FCN checkpoint is missing; configure models.{name}.checkpoint or train this model with run.py"
        )
    model, payload = load_table1_fcn(checkpoint, device)
    expected_ids = fixed_test_ids()
    saved_ids = [
        int(value)
        for value in payload["data_split"]["splits"]["test"]["image_ids"]
    ]
    if payload["mode"] != mode or saved_ids != expected_ids:
        raise RuntimeError(f"{name} mode or fixed Table 1 split is incompatible")
    dataset = CocoFullResolutionSegmentationEvaluationDataset(
        DEFAULT_ANNOTATIONS,
        DEFAULT_IMAGES,
        image_ids=expected_ids,
        normal_name="normal",
    )
    accumulator = DiscreteSegmentationMetrics(dataset.mapping.num_classes)
    area_rows = []
    prediction_cache = output / "table1_prediction_codes"
    for sample in tqdm(dataset, desc=f"Table 1: {name}", leave=False):
        prediction_path = code_cache_path(prediction_cache, int(sample["image_id"]))
        if s.current().get("reuse_predictions", False) and prediction_path.is_file():
            prediction = load_probability_code(prediction_path)
        else:
            prediction = torch.from_numpy(
                predict_table1_fcn(
                    model, sample["image"], device, int(payload["image_size"])
                )
            ).unsqueeze(0)
            save_probability_code(prediction, prediction_path)
        cached_or_save_target(sample)
        accumulator.update(prediction.cpu(), sample["target"].unsqueeze(0))
        sample["class_names"] = dataset.mapping.channel_to_name
        area_rows.extend(image_area_rows(name, sample, prediction))
    metrics = metrics_with_class_names(
        accumulator.compute(), dataset.mapping.channel_to_name
    )
    result = {
        "method": name,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "test_image_count": len(dataset),
        "test_image_ids": expected_ids,
        "architecture": "torchvision FCN-ResNet50",
        "prediction_representation": (
            "exclusive hard label"
            if mode == "single_label"
            else "hard discrete overlap with uniform membership mass"
        ),
        "metrics": metrics,
    }
    save_json(output / "table1_test_metrics.json", result)
    save_area_rows(output, area_rows)
    return result


def baseline_model(checkpoint: Path, device: torch.device):
    import segmentation_models_pytorch as smp

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload["model_state_dict"]
    model = smp.UnetPlusPlus(
        encoder_name=str(payload.get("encoder_name", "efficientnet-b4")),
        encoder_weights=None,
        in_channels=3,
        classes=5,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, payload


@torch.no_grad()
def evaluate_single_label_baseline(device: torch.device) -> dict:
    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "Single-label checkpoint is missing; configure its path or run run.py train --models single_label_unetplusplus"
        )
    model, payload = baseline_model(BASELINE_CHECKPOINT, device)
    ids = fixed_test_ids()
    dataset = CocoFullResolutionSegmentationEvaluationDataset(
        DEFAULT_ANNOTATIONS,
        DEFAULT_IMAGES,
        image_ids=ids,
        normal_name="normal",
    )
    accumulator = DiscreteSegmentationMetrics(dataset.mapping.num_classes)
    area_rows = []
    prediction_cache = s.output() / "evaluation" / "single_label_unetplusplus" / "table1_prediction_codes"
    for sample in tqdm(dataset, desc="Table 1: single-label baseline", leave=False):
        prediction_path = code_cache_path(prediction_cache, int(sample["image_id"]))
        if s.current().get("reuse_predictions", False) and prediction_path.is_file():
            prediction = load_probability_code(prediction_path)
        else:
            image = sample["image"]
            resized = np.asarray(
                Image.fromarray(image).resize(
                    (int(payload.get("image_size", 512)),) * 2, Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )
            logits = model(normalise_image(resized).unsqueeze(0).to(device))
            logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
            label = logits.argmax(dim=1)
            prediction = torch.zeros(
                (1, 4, image.shape[0], image.shape[1]), dtype=torch.float32, device=device
            )
            # Exclusive classes: slight, granulation, eschar, deep, suppuration.
            # Classes absent from the overlap taxonomy map to normal.
            prediction[:, 0] = ((label == 0) | (label == 3)).float()
            prediction[:, 1] = (label == 2).float()  # eschar
            prediction[:, 2] = (label == 1).float()  # granulation
            prediction[:, 3] = (label == 4).float()  # suppuration
            save_probability_code(prediction, prediction_path)
        cached_or_save_target(sample)
        accumulator.update(prediction.cpu(), sample["target"].unsqueeze(0))
        sample["class_names"] = dataset.mapping.channel_to_name
        area_rows.extend(image_area_rows("single_label_unetplusplus", sample, prediction))
    metrics = metrics_with_class_names(accumulator.compute(), dataset.mapping.channel_to_name)
    result = {
        "method": "single_label_unetplusplus",
        "checkpoint": str(BASELINE_CHECKPOINT.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "test_image_count": len(dataset),
        "test_image_ids": ids,
        "architecture": "UNet++ with EfficientNet-B4 encoder",
        "prediction_representation": "exclusive hard label",
        "class_mapping": {
            "slight": "normal",
            "deep": "normal",
            "eschar": "eschar",
            "granulation": "granulation",
            "suppuration": "suppuration",
        },
        "metrics": metrics,
    }
    save_json(s.output() / "evaluation" / "single_label_unetplusplus" / "table1_test_metrics.json", result)
    save_area_rows(s.output() / "evaluation" / "single_label_unetplusplus", area_rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        default=None,
        metavar="MODEL",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild combined area summaries from per-model CSV caches only.",
    )
    args = parser.parse_args()
    if args.summary_only:
        combine_area_outputs()
        print(f"Rebuilt cached area tables in {TABLE_OUTPUT}")
        return 0
    device = resolve_device(args.device)
    results = []
    model_names = args.models or ["baseline", "ours", *ABLATIONS]
    valid_models = {
        "baseline", "independent_sigmoid", "ours", *FCN_OUTPUTS, *ABLATIONS
    }
    invalid_models = [name for name in model_names if name not in valid_models]
    if invalid_models:
        parser.error(f"unknown model(s): {', '.join(invalid_models)}")
    for name in model_names:
        if name == "baseline":
            results.append(evaluate_single_label_baseline(device))
        elif name == "fcn_single_label":
            results.append(evaluate_fcn(name, "single_label", FCN_OUTPUTS[name], device))
        elif name == "fcn_lookup_multilabel":
            results.append(
                evaluate_fcn(name, "lookup_multilabel", FCN_OUTPUTS[name], device)
            )
        elif name == "independent_sigmoid":
            results.append(evaluate_independent_sigmoid(device))
        elif name == "ours":
            results.append(evaluate_overlap_checkpoint(name, OURS_OUTPUT, device))
        else:
            results.append(evaluate_overlap_checkpoint(name, ABLATIONS[name], device))
    for result in results:
        print(
            f"{result['method']}: "
            f"macro Soft Dice={result['metrics']['soft_dice_foreground_macro']:.6f}"
        )
    combine_area_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
