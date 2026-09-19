"""Segmentation confidence intervals and physical-area error tables."""
from __future__ import annotations
import numpy as np
import pandas as pd
from _common import settings as s

FOREGROUND = ["eschar", "granulation", "suppuration"]
TEST_EXPORT = s.output("fig4_performance") / "segmentation"
RULER_GT = s.data_path("ruler")
SEED = int(s.current()["seed"])
BOOTSTRAP_REPLICATES = int(s.current()["bootstrap"])

def _load_sufficient_statistics():
    return pd.read_csv(TEST_EXPORT / "per_image_sufficient_statistics.csv")


def _model_summary_with_ci() -> pd.DataFrame:
    data = _load_sufficient_statistics()
    data = data[data.class_name.isin(FOREGROUND)].copy()
    image_ids = np.sort(data.image_id.unique())
    if len(image_ids) != 54:
        raise RuntimeError(f"Expected 54 test images, found {len(image_ids)}")
    arrays = {}
    for column in ("intersection", "prediction_square", "target_square"):
        pivot = data.pivot(index="image_id", columns="class_name", values=column).reindex(index=image_ids, columns=FOREGROUND)
        arrays[column] = pivot.to_numpy(dtype=float)

    def calculate(indices: np.ndarray):
        intersection = arrays["intersection"][indices].sum(axis=0)
        denominator = arrays["prediction_square"][indices].sum(axis=0) + arrays["target_square"][indices].sum(axis=0)
        dice = np.divide(2 * intersection, denominator, out=np.full(3, np.nan), where=denominator > 0)
        return np.r_[dice, np.nanmean(dice)]

    point = calculate(np.arange(len(image_ids)))
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty((BOOTSTRAP_REPLICATES, 4), dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        bootstrap[replicate] = calculate(rng.integers(0, len(image_ids), len(image_ids)))
    lower, upper = np.nanpercentile(bootstrap, [2.5, 97.5], axis=0)
    groups = [*FOREGROUND, "foreground macro"]
    rows = []
    for group_index, group in enumerate(groups):
        rows.append(
            {
                "metric": "soft_dice",
                "unit": "fraction",
                "group": group,
                "estimate": point[group_index],
                "ci95_lower_image_bootstrap": lower[group_index],
                "ci95_upper_image_bootstrap": upper[group_index],
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "n_test_images": len(image_ids),
                "seed": SEED,
            }
        )
    return pd.DataFrame(rows)

def _manual_ruler_map() -> dict[str, float]:
    path = RULER_GT
    if not path.is_file():
        return {}
    try:
        data = pd.read_csv(path, sep="\t")
        if {"img", "ruler pixel/cm"}.issubset(data.columns):
            return {str(row["img"]).casefold(): float(row["ruler pixel/cm"]) for _, row in data.iterrows()}
    except Exception:
        pass
    data = pd.read_csv(path)
    candidates = {
        "file_name": next((name for name in data.columns if name.casefold() in {"file_name", "img", "image"}), None),
        "scale": next((name for name in data.columns if name.casefold().replace(" ", "_") in {"ruler_pixel/cm", "manual_pixel_per_cm", "pixel_per_cm"}), None),
    }
    if not all(candidates.values()):
        raise ValueError("ruler.csv must contain image name and manual pixel-per-centimetre columns")
    return {str(row[candidates["file_name"]]).casefold(): float(row[candidates["scale"]]) for _, row in data.iterrows()}

def build_end_to_end_area_records() -> tuple[pd.DataFrame, bool, int]:
    stats = _load_sufficient_statistics()
    stats = stats[stats.class_name.isin(FOREGROUND)].copy()
    manual = _manual_ruler_map()
    ruler_path = s.output("fig4_performance") / "ruler" / "predictions.csv"
    ruler = pd.read_csv(ruler_path) if ruler_path.is_file() else pd.DataFrame(columns=["image_id", "predicted_pixel_per_cm", "status"])
    ruler_by_id = {int(row.image_id): row for _, row in ruler.iterrows()}
    test_files = stats[["image_id", "file_name"]].drop_duplicates()
    missing_manual_count = sum(
        (
            str(row.file_name).casefold() not in manual
            or not np.isfinite(manual.get(str(row.file_name).casefold(), np.nan))
            or manual.get(str(row.file_name).casefold(), np.nan) <= 0
        )
        for _, row in test_files.iterrows()
    )
    rows = []
    for _, row in stats.iterrows():
        manual_scale = manual.get(str(row.file_name).casefold())
        prediction_row = ruler_by_id.get(int(row.image_id))
        predicted_scale = None if prediction_row is None else prediction_row.predicted_pixel_per_cm
        if manual_scale is None or not np.isfinite(manual_scale) or manual_scale <= 0:
            status = "missing_manual_scale"
        elif prediction_row is None:
            status = "missing_rulernet_prediction"
        elif prediction_row.status != "valid" or pd.isna(predicted_scale) or float(predicted_scale) <= 0:
            status = "invalid_rulernet_prediction"
        else:
            status = "valid"
        record = row.to_dict()
        record.update(
            {
                "manual_pixel_per_cm": manual_scale,
                "predicted_pixel_per_cm": predicted_scale,
                "gt_area_cm2": np.nan,
                "predicted_area_cm2": np.nan,
                "absolute_error_cm2": np.nan,
                "area_smape_cm2_percent": np.nan,
                "analysis_status": status,
            }
        )
        if status == "valid":
            gt_area = float(row.target_effective_pixels) / float(manual_scale) ** 2
            pred_area = float(row.prediction_effective_pixels) / float(predicted_scale) ** 2
            absolute = abs(pred_area - gt_area)
            denominator = pred_area + gt_area
            record.update(
                {
                    "gt_area_cm2": gt_area,
                    "predicted_area_cm2": pred_area,
                    "absolute_error_cm2": absolute,
                    "area_smape_cm2_percent": 200 * absolute / denominator if denominator > 0 else 0.0,
                }
            )
        rows.append(record)
    frame = pd.DataFrame(rows)
    valid_rows = frame[frame.analysis_status == "valid"]
    totals = (
        valid_rows.groupby(["image_id", "file_name"], as_index=False)[["gt_area_cm2", "predicted_area_cm2"]]
        .sum()
        .rename(
            columns={
                "gt_area_cm2": "total_gt_area_cm2",
                "predicted_area_cm2": "total_predicted_area_cm2",
            }
        )
    )
    total_denominator = totals.total_gt_area_cm2 + totals.total_predicted_area_cm2
    totals["total_area_smape_percent"] = np.divide(
        200 * (totals.total_predicted_area_cm2 - totals.total_gt_area_cm2).abs(),
        total_denominator,
        out=np.zeros(len(totals), dtype=float),
        where=total_denominator.to_numpy(dtype=float) > 0,
    )
    frame = frame.merge(totals, on=["image_id", "file_name"], how="left")
    complete = frame.image_id.nunique() == 54 and frame.analysis_status.eq("valid").all()
    return frame, complete, missing_manual_count

def main():
    output = s.output()
    output.mkdir(parents=True, exist_ok=True)
    _model_summary_with_ci().to_csv(output / "segmentation_summary.csv", index=False)
    records, complete, missing = build_end_to_end_area_records()
    records.to_csv(output / "area_errors.csv", index=False)
    rows = []
    for name in FOREGROUND:
        valid = records[(records.class_name == name) & (records.analysis_status == "valid")]
        for metric in ["area_smape_cm2_percent", "absolute_error_cm2"]:
            values = valid[metric].to_numpy(dtype=float)
            if len(values):
                rng = np.random.default_rng(SEED)
                bootstrap = np.array([rng.choice(values, size=len(values), replace=True).mean()
                                      for _ in range(BOOTSTRAP_REPLICATES)])
                lower, upper = np.percentile(bootstrap, [2.5, 97.5])
            else:
                lower, upper = None, None
            rows.append({"class_name": name, "metric": metric, "n": len(values),
                         "mean": float(values.mean()) if len(values) else None,
                         "ci95_lower_image_bootstrap": lower,
                         "ci95_upper_image_bootstrap": upper,
                         "bootstrap_replicates": BOOTSTRAP_REPLICATES, "seed": SEED,
                         "median": float(np.median(values)) if len(values) else None,
                         "q25": float(np.percentile(values, 25)) if len(values) else None,
                         "q75": float(np.percentile(values, 75)) if len(values) else None})
    pd.DataFrame(rows).to_csv(output / "area_summary.csv", index=False)
    print(f"Area measurements complete: {complete}; missing reference scales: {missing}")
