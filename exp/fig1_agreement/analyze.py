"""Compute rater agreement for hard/soft wound-area annotations.

The script maps COCO polygons drawn on 1024 x 1024 thumbnails back to the
matching original-image coordinate system, rasterizes them at original
resolution, and converts original pixels to physical area with the ruler
calibration. It then reports ICC(A,1) for the two commonly observed wound
classes and fits the prespecified mixed-effects model to pairwise symmetric
relative differences among the raters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.stats import norm, t
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


PRIMARY_CATEGORIES = ("granulation", "suppuration")
NORMAL_NAME = "normal"
METHOD_ORDER = ("hard", "soft")
RATER_ORDER = ("doc1", "doc2", "doc3")
FILE_PATTERN = re.compile(r"^annotations_(doc\d+)_(hard|soft)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icc-dir", type=Path, default=Path("dataset/icc"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/fig1_agreement/statistics"))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--test-tail",
        choices=("one-sided", "two-sided"),
        default="one-sided",
        help=(
            "Tail for the primary soft-vs-hard tests. The one-sided alternatives "
            "are ICC_soft > ICC_hard and mixed-model soft - hard < 0."
        ),
    )
    return parser.parse_args()


def load_rulers(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["img", "ruler pixel/cm"]:
            raise ValueError(f"Unexpected ruler columns: {reader.fieldnames}")
        rulers = {row["img"]: float(row["ruler pixel/cm"]) for row in reader}
    if not rulers or any(not math.isfinite(value) or value <= 0 for value in rulers.values()):
        raise ValueError("Every ruler calibration must be finite and positive")
    return rulers


def decode_segmentation(
    segmentation: Any,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    if isinstance(segmentation, list):
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        scaled_polygons = [
            [
                float(value) * (scale_x if index % 2 == 0 else scale_y)
                for index, value in enumerate(polygon)
            ]
            for polygon in segmentation
        ]
        encoded = mask_utils.merge(
            mask_utils.frPyObjects(scaled_polygons, target_height, target_width)
        )
        decoded = mask_utils.decode(encoded)
    elif isinstance(segmentation, dict):
        encoded = dict(segmentation)
        if isinstance(encoded.get("counts"), list):
            encoded = mask_utils.frPyObjects(encoded, source_height, source_width)
        elif isinstance(encoded.get("counts"), str):
            encoded["counts"] = encoded["counts"].encode("ascii")
        decoded = mask_utils.decode(encoded)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        if (target_height, target_width) != (source_height, source_width):
            decoded = np.asarray(
                Image.fromarray(np.asarray(decoded, dtype=np.uint8)).resize(
                    (target_width, target_height), resample=Image.Resampling.NEAREST
                )
            )
    else:
        raise TypeError("COCO segmentation must be polygons or RLE")
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    if decoded.shape != (target_height, target_width):
        raise ValueError(
            f"Decoded mask shape {decoded.shape} != {(target_height, target_width)}"
        )
    return np.asarray(decoded, dtype=bool)


def build_category_masks(
    image: dict[str, Any],
    annotations: Iterable[dict[str, Any]],
    foreground_categories: list[tuple[int, str]],
    target_height: int,
    target_width: int,
) -> np.ndarray:
    source_height, source_width = int(image["height"]), int(image["width"])
    masks = np.zeros(
        (len(foreground_categories), target_height, target_width), dtype=bool
    )
    category_to_index = {
        category_id: index
        for index, (category_id, _) in enumerate(foreground_categories)
    }
    for annotation in annotations:
        category_id = int(annotation["category_id"])
        if category_id not in category_to_index:
            continue
        masks[category_to_index[category_id]] |= decode_segmentation(
            annotation["segmentation"],
            source_height,
            source_width,
            target_height,
            target_width,
        )
    return masks


def resolve_areas(
    masks: np.ndarray, category_ids: np.ndarray, method: str
) -> np.ndarray:
    if method == "hard":
        labels = np.max(
            np.where(masks, category_ids[:, None, None], 0), axis=0
        )
        return np.asarray([(labels == category_id).sum() for category_id in category_ids])
    if method == "soft":
        active_count = masks.sum(axis=0)
        return np.asarray(
            [
                np.divide(
                    masks[index],
                    active_count,
                    out=np.zeros(active_count.shape, dtype=np.float64),
                    where=active_count > 0,
                ).sum()
                for index in range(len(category_ids))
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unknown method: {method}")


def load_annotation_files(annotation_dir: Path) -> list[tuple[Path, str, str]]:
    files: list[tuple[Path, str, str]] = []
    for path in sorted(annotation_dir.glob("annotations_*.json")):
        match = FILE_PATTERN.match(path.name)
        if match:
            files.append((path, match.group(1), match.group(2)))
    expected = {(rater, method) for rater in RATER_ORDER for method in METHOD_ORDER}
    found = {(rater, method) for _, rater, method in files}
    if found != expected:
        raise ValueError(f"Expected annotation combinations {sorted(expected)}, found {sorted(found)}")
    return files


def collect_areas(
    icc_dir: Path,
    annotation_files: list[tuple[Path, str, str]],
    rulers: dict[str, float],
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    thumbnail_dir = icc_dir / "samples"
    original_dir = thumbnail_dir / "ori"
    reference_names: set[str] | None = None
    reference_categories: list[tuple[int, str]] | None = None
    datasets: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for json_path, rater, method in annotation_files:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_names = {image["file_name"] for image in data["images"]}
        categories = sorted(
            (int(category["id"]), str(category["name"]))
            for category in data["categories"]
            if str(category["name"]).casefold() != NORMAL_NAME
        )
        if reference_names is None:
            reference_names = image_names
            reference_categories = categories
        elif image_names != reference_names or categories != reference_categories:
            raise ValueError(
                "All rater/method COCO files must contain identical images and categories"
            )

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        datasets[(rater, method)] = {
            "data": data,
            "annotations_by_image": annotations_by_image,
            "categories": categories,
        }

        category_ids = np.asarray([item[0] for item in categories], dtype=np.int64)
        for image in data["images"]:
            sample = str(image["file_name"])
            thumbnail_path = thumbnail_dir / sample
            original_path = original_dir / sample
            if sample not in rulers:
                raise ValueError(f"Missing ruler for {sample}")
            dimension_path = icc_dir / "image_dimensions.json"
            if dimension_path.is_file():
                dimensions = json.loads(dimension_path.read_text(encoding="utf-8"))[sample]
                thumbnail_width, thumbnail_height = dimensions["thumbnail"]
                original_width, original_height = dimensions["original"]
            else:
                with Image.open(thumbnail_path) as thumbnail:
                    thumbnail_width, thumbnail_height = thumbnail.size
                with Image.open(original_path) as original:
                    original_width, original_height = original.size
            if (thumbnail_width, thumbnail_height) != (
                int(image["width"]),
                int(image["height"]),
            ):
                raise ValueError(f"COCO dimensions do not match thumbnail for {sample}")

            masks = build_category_masks(
                image,
                annotations_by_image[int(image["id"])],
                categories,
                original_height,
                original_width,
            )
            pixel_areas = resolve_areas(masks, category_ids, method)
            scale_x = original_width / thumbnail_width
            scale_y = original_height / thumbnail_height
            cm2_per_original_pixel = 1.0 / rulers[sample] ** 2
            for (category_id, category), pixel_area in zip(categories, pixel_areas):
                rows.append(
                    {
                        "sample": sample,
                        "rater": rater,
                        "method": method,
                        "category_id": category_id,
                        "category": category,
                        "weighted_area_original_px": float(pixel_area),
                        "area_cm2": float(pixel_area * cm2_per_original_pixel),
                        "thumbnail_width_px": thumbnail_width,
                        "thumbnail_height_px": thumbnail_height,
                        "original_width_px": original_width,
                        "original_height_px": original_height,
                        "thumbnail_to_original_scale_x": scale_x,
                        "thumbnail_to_original_scale_y": scale_y,
                        "ruler_px_per_cm": rulers[sample],
                        "cm2_per_original_pixel": cm2_per_original_pixel,
                    }
                )

    area_table = pd.DataFrame(rows).sort_values(
        ["sample", "method", "rater", "category_id"]
    )
    return area_table, datasets


def icc_absolute_agreement_single(values: np.ndarray) -> float:
    """Return two-way random, absolute-agreement, single-measure ICC(A,1)."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return float("nan")
    n, k = matrix.shape
    grand_mean = matrix.mean()
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    ss_rows = k * np.square(row_means - grand_mean).sum()
    ss_columns = n * np.square(column_means - grand_mean).sum()
    residual = matrix - row_means[:, None] - column_means[None, :] + grand_mean
    ss_error = np.square(residual).sum()
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = (
        ms_rows
        + (k - 1) * ms_error
        + k * (ms_columns - ms_error) / n
    )
    if np.isclose(denominator, 0.0):
        return float("nan")
    return float((ms_rows - ms_error) / denominator)


def bootstrap_icc(
    values: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float, np.ndarray]:
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, values.shape[0], values.shape[0])
        estimate = icc_absolute_agreement_single(values[indices])
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return float("nan"), float("nan"), np.asarray([], dtype=np.float64)
    estimate_array = np.asarray(estimates, dtype=np.float64)
    lower, upper = np.percentile(estimate_array, [2.5, 97.5])
    return float(lower), float(upper), estimate_array


def calculate_icc_table(
    areas: pd.DataFrame, n_bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = areas[areas["category"].isin(PRIMARY_CATEGORIES)]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for category in PRIMARY_CATEGORIES:
            subset = selected[
                (selected["method"] == method)
                & (selected["category"] == category)
            ]
            matrix = subset.pivot(index="sample", columns="rater", values="area_cm2")
            matrix = matrix.loc[:, list(RATER_ORDER)].sort_index()
            if matrix.isna().any().any():
                raise ValueError(f"Incomplete rater data for {method}/{category}")
            values = matrix.to_numpy(dtype=np.float64)
            estimate = icc_absolute_agreement_single(values)
            lower, upper, bootstrap_estimates = bootstrap_icc(
                values, n_bootstrap, rng
            )
            rows.append(
                {
                    "method": method,
                    "category": category,
                    "icc_type": "ICC(A,1): two-way random, absolute agreement, single measurement",
                    "icc": estimate,
                    "ci95_lower_cluster_bootstrap": lower,
                    "ci95_upper_cluster_bootstrap": upper,
                    "n_samples": values.shape[0],
                    "n_raters": values.shape[1],
                    "bootstrap_replicates_requested": n_bootstrap,
                    "bootstrap_replicates_valid": len(bootstrap_estimates),
                }
            )
            bootstrap_rows.extend(
                {
                    "method": method,
                    "category": category,
                    "bootstrap_replicate": replicate_index + 1,
                    "icc": bootstrap_value,
                }
                for replicate_index, bootstrap_value in enumerate(
                    bootstrap_estimates
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def calculate_paired_icc_comparison(
    areas: pd.DataFrame, n_bootstrap: int, seed: int, test_tail: str
) -> pd.DataFrame:
    """Compare soft and hard ICCs with exact paired method-label permutations."""

    selected = areas[areas["category"].isin(PRIMARY_CATEGORIES)]
    rng = np.random.default_rng(seed + 1)
    rows: list[dict[str, Any]] = []
    for category in PRIMARY_CATEGORIES:
        subset = selected[selected["category"] == category]
        wide = subset.pivot(
            index="sample", columns=["method", "rater"], values="area_cm2"
        ).sort_index()
        expected_columns = pd.MultiIndex.from_product(
            [METHOD_ORDER, RATER_ORDER], names=["method", "rater"]
        )
        wide = wide.reindex(columns=expected_columns)
        if wide.isna().any().any():
            raise ValueError(f"Incomplete paired ICC data for {category}")

        hard = wide["hard"].loc[:, list(RATER_ORDER)].to_numpy(dtype=np.float64)
        soft = wide["soft"].loc[:, list(RATER_ORDER)].to_numpy(dtype=np.float64)
        hard_icc = icc_absolute_agreement_single(hard)
        soft_icc = icc_absolute_agreement_single(soft)
        observed_delta = soft_icc - hard_icc

        bootstrap_deltas: list[float] = []
        for _ in range(n_bootstrap):
            indices = rng.integers(0, len(wide), len(wide))
            delta = (
                icc_absolute_agreement_single(soft[indices])
                - icc_absolute_agreement_single(hard[indices])
            )
            if math.isfinite(delta):
                bootstrap_deltas.append(delta)
        if not bootstrap_deltas:
            raise RuntimeError(f"No valid paired bootstrap replicates for {category}")

        bootstrap_array = np.asarray(bootstrap_deltas, dtype=np.float64)
        lower, upper = np.percentile(bootstrap_array, [2.5, 97.5])
        permutation_deltas: list[float] = []
        permutation_count = 1 << len(wide)
        row_numbers = np.arange(len(wide), dtype=np.uint64)
        for permutation_code in range(permutation_count):
            swap = ((np.uint64(permutation_code) >> row_numbers) & 1).astype(bool)
            permuted_hard = hard.copy()
            permuted_soft = soft.copy()
            permuted_hard[swap] = soft[swap]
            permuted_soft[swap] = hard[swap]
            delta = (
                icc_absolute_agreement_single(permuted_soft)
                - icc_absolute_agreement_single(permuted_hard)
            )
            if math.isfinite(delta):
                permutation_deltas.append(delta)
        permutation_array = np.asarray(permutation_deltas, dtype=np.float64)
        p_one_sided = float(
            np.count_nonzero(permutation_array >= observed_delta - 1e-15)
            / len(permutation_array)
        )
        p_two_sided = float(
            np.count_nonzero(
                np.abs(permutation_array) >= abs(observed_delta) - 1e-15
            )
            / len(permutation_array)
        )
        rows.append(
            {
                "category": category,
                "hard_icc": hard_icc,
                "soft_icc": soft_icc,
                "delta_icc_soft_minus_hard": observed_delta,
                "delta_ci95_lower_paired_bootstrap": float(lower),
                "delta_ci95_upper_paired_bootstrap": float(upper),
                "p_one_sided_paired_permutation_soft_greater": p_one_sided,
                "p_two_sided_paired_permutation": p_two_sided,
                "alternative_one_sided": "ICC_soft > ICC_hard",
                "n_samples": len(wide),
                "bootstrap_replicates_requested": n_bootstrap,
                "bootstrap_replicates_valid": len(bootstrap_array),
                "permutation_scheme": "exact paired sample-level method-label swaps",
                "permutation_replicates_valid": len(permutation_array),
            }
        )

    result = pd.DataFrame(rows)
    result["p_one_sided_holm"] = multipletests(
        result["p_one_sided_paired_permutation_soft_greater"].to_numpy(),
        method="holm",
    )[1]
    result["p_two_sided_holm"] = multipletests(
        result["p_two_sided_paired_permutation"].to_numpy(),
        method="holm",
    )[1]
    if test_tail == "one-sided":
        result["p_selected_raw"] = result[
            "p_one_sided_paired_permutation_soft_greater"
        ]
        result["p_selected_holm"] = result["p_one_sided_holm"]
        result["alternative_selected"] = "ICC_soft > ICC_hard"
    else:
        result["p_selected_raw"] = result["p_two_sided_paired_permutation"]
        result["p_selected_holm"] = result["p_two_sided_holm"]
        result["alternative_selected"] = "ICC_soft != ICC_hard"
    result["test_tail_selected"] = test_tail
    return result


def build_difference_table(areas: pd.DataFrame) -> pd.DataFrame:
    selected = areas[areas["category"].isin(PRIMARY_CATEGORIES)]
    wide = selected.pivot(
        index=["sample", "method", "category"], columns="rater", values="area_cm2"
    ).reset_index()
    wide.columns.name = None
    if wide[list(RATER_ORDER)].isna().any().any():
        raise ValueError("Every method/category/sample must have all rater areas")

    pairwise_tables: list[pd.DataFrame] = []
    for rater_1, rater_2 in combinations(RATER_ORDER, 2):
        pairwise = wide[["sample", "method", "category"]].copy()
        pairwise["rater_pair"] = f"{rater_1}-{rater_2}"
        pairwise["rater_1"] = rater_1
        pairwise["rater_2"] = rater_2
        pairwise["rater_1_area_cm2"] = wide[rater_1]
        pairwise["rater_2_area_cm2"] = wide[rater_2]
        pairwise["absolute_difference_cm2"] = (
            pairwise["rater_1_area_cm2"] - pairwise["rater_2_area_cm2"]
        ).abs()
        denominator = (
            pairwise["rater_1_area_cm2"] + pairwise["rater_2_area_cm2"]
        )
        pairwise["symmetric_relative_difference"] = np.divide(
            2.0 * pairwise["absolute_difference_cm2"],
            denominator,
            out=np.zeros(len(pairwise), dtype=np.float64),
            where=denominator.to_numpy() > 0,
        )
        pairwise["relative_difference_pct"] = (
            100.0 * pairwise["symmetric_relative_difference"]
        )
        pairwise["both_raters_zero"] = denominator.eq(0)
        pairwise_tables.append(pairwise)

    return pd.concat(pairwise_tables, ignore_index=True).sort_values(
        ["sample", "category", "rater_pair", "method"]
    )


def fit_mixed_model(differences: pd.DataFrame, test_tail: str):
    sample_differences = (
        differences.groupby(["sample", "method", "category"], as_index=False)[
            "relative_difference_pct"
        ]
        .mean()
    )
    formula = (
        'relative_difference_pct ~ C(method, Treatment(reference="hard")) '
        '+ C(category, Treatment(reference="granulation"))'
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = smf.mixedlm(
            formula, sample_differences, groups=sample_differences["sample"]
        )
        fitted = model.fit(reml=False, method="powell", disp=False)
    warning_messages = [str(item.message) for item in captured]

    intervals = fitted.conf_int()
    coefficient_rows: list[dict[str, Any]] = []
    for term in fitted.fe_params.index:
        estimate = float(fitted.fe_params[term])
        standard_error = float(fitted.bse_fe[term])
        z_value = estimate / standard_error
        p_two_sided = float(2.0 * norm.sf(abs(z_value)))
        row = {
            "term": term,
            "estimate_percentage_points": estimate,
            "standard_error": standard_error,
            "z_value": z_value,
            "p_two_sided": p_two_sided,
            "ci95_lower": float(intervals.loc[term, 0]),
            "ci95_upper": float(intervals.loc[term, 1]),
            "p_one_sided_lower": float(norm.cdf(z_value)) if "method" in term else np.nan,
        }
        coefficient_rows.append(row)
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients["p_selected_primary_contrast"] = np.nan
    coefficients["test_tail_selected"] = ""
    method_mask = coefficients["term"].str.contains("method")
    selected_p_column = (
        "p_one_sided_lower" if test_tail == "one-sided" else "p_two_sided"
    )
    coefficients.loc[method_mask, "p_selected_primary_contrast"] = (
        coefficients.loc[method_mask, selected_p_column]
    )
    coefficients.loc[method_mask, "test_tail_selected"] = test_tail
    diagnostics = {
        "formula": formula,
        "fit_method": "maximum likelihood",
        "random_effect": "sample random intercept",
        "n_observations": int(fitted.nobs),
        "n_samples": int(sample_differences["sample"].nunique()),
        "n_raters": len(RATER_ORDER),
        "n_rater_pairs": math.comb(len(RATER_ORDER), 2),
        "pairwise_rows_before_within_sample_aggregation": len(differences),
        "response_aggregation": "mean pairwise SMAPE within sample/method/category",
        "converged": bool(fitted.converged),
        "log_likelihood": float(fitted.llf),
        "sample_random_intercept_variance": float(fitted.cov_re.iloc[0, 0]),
        "residual_variance": float(fitted.scale),
        "warnings": warning_messages,
        "primary_test_tail": test_tail,
        "primary_one_sided_alternative": "soft - hard < 0",
    }
    return fitted, coefficients, diagnostics


def write_summary(
    path: Path,
    icc_table: pd.DataFrame,
    icc_comparison: pd.DataFrame,
    coefficients: pd.DataFrame,
    diagnostics: dict[str, Any],
    test_tail: str,
) -> None:
    method_term = coefficients[coefficients["term"].str.contains("method")].iloc[0]
    lines = [
        "# Inter-rater wound-area agreement analysis",
        "",
        "## Prespecified analysis",
        "",
        "- ICC: ICC(A,1), two-way random-effects, absolute-agreement, single-measure.",
        "- Polygon coordinates are scaled to each original image and rasterized at original resolution before area calculation.",
        "- Primary categories: granulation and suppuration.",
        "- Hard overlap: the larger COCO category_id wins.",
        "- Soft overlap: each active foreground category receives equal pixel mass.",
        f"- Raters: {', '.join(RATER_ORDER)}; all {math.comb(len(RATER_ORDER), 2)} rater pairs are included.",
        "- Pairwise error: 100 * 2 * |rater 1 - rater 2| / (rater 1 + rater 2); the three pairwise values are averaged within each sample/method/category for plotting and modelling.",
        "- If both raters in a pair report zero area, relative difference is defined as 0%; if only one is zero, it is 200%.",
        "- Fixed effects: method and category; random intercept: sample.",
        f"- Primary test tail: {test_tail}.",
        "- One-sided alternatives, when selected: ICC(soft) > ICC(hard) and mixed-model soft - hard < 0.",
        "- Normal is implicit image background and is excluded. Eschar is descriptive only because only one sample is non-zero.",
        "",
        "## ICC results",
        "",
    ]
    for row in icc_table.itertuples(index=False):
        lines.append(
            f"- {row.method}/{row.category}: ICC = {row.icc:.4f}, "
            f"bootstrap 95% CI [{row.ci95_lower_cluster_bootstrap:.4f}, "
            f"{row.ci95_upper_cluster_bootstrap:.4f}], n = {row.n_samples}."
        )
    lines.extend(["", "## Paired bootstrap comparison of ICC", ""])
    for row in icc_comparison.itertuples(index=False):
        lines.append(
            f"- {row.category}: delta ICC (soft - hard) = "
            f"{row.delta_icc_soft_minus_hard:.4f}, paired-bootstrap 95% CI "
            f"[{row.delta_ci95_lower_paired_bootstrap:.4f}, "
            f"{row.delta_ci95_upper_paired_bootstrap:.4f}], {test_tail} exact paired "
            f"permutation p = {row.p_selected_raw:.6g}, Holm-adjusted "
            f"p = {row.p_selected_holm:.6g}."
        )
    lines.extend(
        [
            "",
            "## Primary mixed-model contrast",
            "",
            f"- Soft - hard: {method_term['estimate_percentage_points']:.4f} percentage points "
            f"(95% CI {method_term['ci95_lower']:.4f} to {method_term['ci95_upper']:.4f}; "
            f"selected {test_tail} p = {method_term['p_selected_primary_contrast']:.6g}; "
            f"one-sided p = {method_term['p_one_sided_lower']:.6g}; "
            f"two-sided p = {method_term['p_two_sided']:.6g}).",
            f"- Model convergence: {diagnostics['converged']}; observations = {diagnostics['n_observations']}; "
            f"samples = {diagnostics['n_samples']}.",
            "",
            (
                "For a one-sided analysis, the directional claim is supported only if "
                "the soft - hard estimate is negative and the selected p value meets "
                "the prespecified threshold. A two-sided analysis tests any method difference."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")
    args.result_dir.mkdir(parents=True, exist_ok=True)

    rulers = load_rulers(args.icc_dir / "ruler.csv")
    annotation_files = load_annotation_files(args.icc_dir / "polygon_annot")
    areas, datasets = collect_areas(args.icc_dir, annotation_files, rulers)
    missing_primary = set(PRIMARY_CATEGORIES) - set(areas["category"])
    if missing_primary:
        raise ValueError(f"Missing primary categories: {sorted(missing_primary)}")

    differences = build_difference_table(areas)
    icc_table, icc_bootstrap = calculate_icc_table(
        areas, args.bootstrap, args.seed
    )
    icc_comparison = calculate_paired_icc_comparison(
        areas, args.bootstrap, args.seed, args.test_tail
    )
    fitted, coefficients, diagnostics = fit_mixed_model(
        differences, args.test_tail
    )

    areas.to_csv(args.result_dir / "areas_by_sample.csv", index=False, float_format="%.8f")
    differences.to_csv(
        args.result_dir / "inter_rater_relative_differences.csv",
        index=False,
        float_format="%.8f",
    )
    icc_table.to_csv(args.result_dir / "icc_by_category.csv", index=False, float_format="%.8f")
    icc_bootstrap.to_csv(
        args.result_dir / "icc_bootstrap_replicates.csv",
        index=False,
        float_format="%.8f",
    )
    icc_comparison.to_csv(
        args.result_dir / "icc_method_comparison_bootstrap.csv",
        index=False,
        float_format="%.8g",
    )
    coefficients.to_csv(
        args.result_dir / "mixed_model_coefficients.csv", index=False, float_format="%.8g"
    )
    (args.result_dir / "mixed_model_summary.txt").write_text(
        fitted.summary().as_text()
        + "\n\nDiagnostics\n"
        + json.dumps(diagnostics, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    write_summary(
        args.result_dir / "analysis_summary.txt",
        icc_table,
        icc_comparison,
        coefficients,
        diagnostics,
        args.test_tail,
    )

    print(f"Area rows: {len(areas)}")
    print(f"Relative-difference rows: {len(differences)}")
    print(f"Primary test tail: {args.test_tail}")
    print(f"Results: {args.result_dir.resolve()}")


if __name__ == "__main__":
    main()
