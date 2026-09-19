"""COCO data validation and selective soft-target generation.

The central contract is that targets are one-hot outside overlaps between
different semantic classes. Only pixels with two or more active class masks
receive distance-weighted soft labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset


class DatasetValidationError(ValueError):
    """Raised when a COCO file or one of its referenced assets is invalid."""


@dataclass(frozen=True)
class CategoryMapping:
    """Deterministic mapping between COCO categories and model channels."""

    channel_to_category_id: tuple[int, ...]
    channel_to_name: tuple[str, ...]
    normal_category_id: int

    @property
    def num_classes(self) -> int:
        return len(self.channel_to_category_id)

    @property
    def category_id_to_channel(self) -> dict[int, int]:
        return {
            category_id: channel
            for channel, category_id in enumerate(self.channel_to_category_id)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal_category_id": self.normal_category_id,
            "num_classes": self.num_classes,
            "channels": [
                {
                    "channel": channel,
                    "category_id": category_id,
                    "category_name": self.channel_to_name[channel],
                }
                for channel, category_id in enumerate(self.channel_to_category_id)
            ],
        }


@dataclass(frozen=True)
class PreflightReport:
    annotation_path: str
    image_dir: str
    total_images: int
    annotated_images: int
    excluded_unannotated_images: int
    annotations: int
    categories: int
    mapping: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_coco_json(annotation_path: str | Path) -> dict[str, Any]:
    """Load JSON and turn parser failures into actionable validation errors."""

    path = Path(annotation_path)
    if not path.is_file():
        raise DatasetValidationError(f"COCO annotation file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(
            f"COCO annotation is not valid UTF-8: {path}: byte {exc.start}: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(
            f"Invalid COCO JSON: {path}: line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise DatasetValidationError(f"COCO root must be an object: {path}")
    return data


def build_category_mapping(
    categories: Sequence[Mapping[str, Any]], normal_name: str = "normal"
) -> CategoryMapping:
    if not isinstance(categories, list) or not categories:
        raise DatasetValidationError("COCO 'categories' must be a non-empty list")

    seen_ids: set[int] = set()
    normalized_name = normal_name.strip().casefold()
    normal_ids: list[int] = []
    parsed: list[tuple[int, str]] = []
    for index, category in enumerate(categories):
        if not isinstance(category, Mapping):
            raise DatasetValidationError(f"categories[{index}] must be an object")
        category_id = category.get("id")
        name = category.get("name")
        if not isinstance(category_id, int):
            raise DatasetValidationError(f"categories[{index}].id must be an integer")
        if category_id in seen_ids:
            raise DatasetValidationError(f"Duplicate category id: {category_id}")
        if not isinstance(name, str) or not name.strip():
            raise DatasetValidationError(
                f"Category {category_id} must have a non-empty name"
            )
        seen_ids.add(category_id)
        clean_name = name.strip()
        parsed.append((category_id, clean_name))
        if clean_name.casefold() == normalized_name:
            normal_ids.append(category_id)

    if len(normal_ids) != 1:
        raise DatasetValidationError(
            f"Expected exactly one category named {normal_name!r} "
            f"(case-insensitive), found {len(normal_ids)}"
        )

    normal_id = normal_ids[0]
    foreground = sorted(
        ((category_id, name) for category_id, name in parsed if category_id != normal_id),
        key=lambda item: item[0],
    )
    return CategoryMapping(
        channel_to_category_id=(normal_id, *(item[0] for item in foreground)),
        channel_to_name=(normal_name, *(item[1] for item in foreground)),
        normal_category_id=normal_id,
    )


def _annotation_label(annotation: Mapping[str, Any]) -> str:
    return f"annotation id={annotation.get('id', '<missing>')}"


def decode_segmentation(
    segmentation: Any,
    height: int,
    width: int,
    *,
    annotation_label: str = "annotation",
) -> np.ndarray:
    """Decode COCO polygon or RLE segmentation to a boolean HxW mask."""

    if height <= 0 or width <= 0:
        raise DatasetValidationError(
            f"{annotation_label}: image dimensions must be positive"
        )
    try:
        if isinstance(segmentation, list):
            if not segmentation:
                raise DatasetValidationError(
                    f"{annotation_label}: polygon segmentation is empty"
                )
            for polygon_index, polygon in enumerate(segmentation):
                if (
                    not isinstance(polygon, list)
                    or len(polygon) < 6
                    or len(polygon) % 2 != 0
                ):
                    raise DatasetValidationError(
                        f"{annotation_label}: polygon {polygon_index} must contain "
                        "at least three x/y pairs"
                    )
                if not all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in polygon
                ):
                    raise DatasetValidationError(
                        f"{annotation_label}: polygon {polygon_index} contains "
                        "non-finite coordinates"
                    )
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation, Mapping):
            if "counts" not in segmentation or "size" not in segmentation:
                raise DatasetValidationError(
                    f"{annotation_label}: RLE requires 'counts' and 'size'"
                )
            size = segmentation["size"]
            if list(size) != [height, width]:
                raise DatasetValidationError(
                    f"{annotation_label}: RLE size {size!r} does not match "
                    f"image size [{height}, {width}]"
                )
            if isinstance(segmentation["counts"], list):
                rle = mask_utils.frPyObjects(dict(segmentation), height, width)
            else:
                rle = dict(segmentation)
                if isinstance(rle["counts"], str):
                    rle["counts"] = rle["counts"].encode("ascii")
        else:
            raise DatasetValidationError(
                f"{annotation_label}: segmentation must be polygon list or RLE object"
            )

        decoded = mask_utils.decode(rle)
    except DatasetValidationError:
        raise
    except Exception as exc:
        raise DatasetValidationError(
            f"{annotation_label}: failed to decode segmentation: {exc}"
        ) from exc

    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    if decoded.shape != (height, width):
        raise DatasetValidationError(
            f"{annotation_label}: decoded mask has shape {decoded.shape}, "
            f"expected {(height, width)}"
        )
    mask = np.asarray(decoded, dtype=bool)
    if not mask.any():
        raise DatasetValidationError(f"{annotation_label}: decoded mask is empty")
    return mask


def validate_coco_dataset(
    annotation_path: str | Path,
    image_dir: str | Path,
    *,
    normal_name: str = "normal",
    decode_masks: bool = True,
) -> tuple[
    dict[str, Any],
    CategoryMapping,
    PreflightReport,
    dict[int, list[dict[str, Any]]],
]:
    """Validate COCO metadata, referenced images, and segmentations."""

    annotation_path = Path(annotation_path)
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise DatasetValidationError(f"Image directory does not exist: {image_dir}")

    data = load_coco_json(annotation_path)
    for key in ("images", "annotations", "categories"):
        if key not in data or not isinstance(data[key], list):
            raise DatasetValidationError(f"COCO '{key}' must be a list")

    mapping = build_category_mapping(data["categories"], normal_name=normal_name)
    valid_category_ids = set(mapping.channel_to_category_id)

    images_by_id: dict[int, dict[str, Any]] = {}
    for index, image in enumerate(data["images"]):
        if not isinstance(image, dict):
            raise DatasetValidationError(f"images[{index}] must be an object")
        image_id = image.get("id")
        if not isinstance(image_id, int):
            raise DatasetValidationError(f"images[{index}].id must be an integer")
        if image_id in images_by_id:
            raise DatasetValidationError(f"Duplicate image id: {image_id}")
        file_name = image.get("file_name")
        height, width = image.get("height"), image.get("width")
        if not isinstance(file_name, str) or not file_name:
            raise DatasetValidationError(f"Image {image_id} has invalid file_name")
        if not isinstance(height, int) or height <= 0:
            raise DatasetValidationError(f"Image {image_id} has invalid height")
        if not isinstance(width, int) or width <= 0:
            raise DatasetValidationError(f"Image {image_id} has invalid width")
        image_path = image_dir / file_name
        if not image_path.is_file():
            raise DatasetValidationError(
                f"Image {image_id} references missing file: {image_path}"
            )
        try:
            with Image.open(image_path) as pil_image:
                actual_size = pil_image.size
        except Exception as exc:
            raise DatasetValidationError(
                f"Image {image_id} cannot be opened: {image_path}: {exc}"
            ) from exc
        if actual_size != (width, height):
            raise DatasetValidationError(
                f"Image {image_id} size mismatch: COCO={(width, height)}, "
                f"file={actual_size}: {image_path}"
            )
        images_by_id[image_id] = image

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    annotation_ids: set[int] = set()
    for index, annotation in enumerate(data["annotations"]):
        if not isinstance(annotation, dict):
            raise DatasetValidationError(f"annotations[{index}] must be an object")
        label = _annotation_label(annotation)
        annotation_id = annotation.get("id")
        if not isinstance(annotation_id, int):
            raise DatasetValidationError(f"annotations[{index}].id must be an integer")
        if annotation_id in annotation_ids:
            raise DatasetValidationError(f"Duplicate annotation id: {annotation_id}")
        annotation_ids.add(annotation_id)

        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if image_id not in images_by_id:
            raise DatasetValidationError(f"{label}: unknown image_id {image_id!r}")
        if category_id not in valid_category_ids:
            raise DatasetValidationError(
                f"{label}: unknown category_id {category_id!r}"
            )
        if "segmentation" not in annotation:
            raise DatasetValidationError(f"{label}: missing segmentation")
        if decode_masks:
            image = images_by_id[image_id]
            decode_segmentation(
                annotation["segmentation"],
                image["height"],
                image["width"],
                annotation_label=label,
            )
        annotations_by_image[image_id].append(annotation)

    annotated_images = len(annotations_by_image)
    report = PreflightReport(
        annotation_path=str(annotation_path.resolve()),
        image_dir=str(image_dir.resolve()),
        total_images=len(images_by_id),
        annotated_images=annotated_images,
        excluded_unannotated_images=len(images_by_id) - annotated_images,
        annotations=len(data["annotations"]),
        categories=mapping.num_classes,
        mapping=mapping.to_dict(),
    )
    return data, mapping, report, dict(annotations_by_image)


def build_class_masks(
    image: Mapping[str, Any],
    annotations: Iterable[Mapping[str, Any]],
    mapping: CategoryMapping,
) -> np.ndarray:
    """Union instances per category and construct Channel-0 background."""

    height, width = int(image["height"]), int(image["width"])
    masks = np.zeros((mapping.num_classes, height, width), dtype=bool)
    category_to_channel = mapping.category_id_to_channel

    for annotation in annotations:
        category_id = int(annotation["category_id"])
        if category_id not in category_to_channel:
            raise DatasetValidationError(
                f"{_annotation_label(annotation)}: category {category_id} "
                "is not present in the mapping"
            )
        channel = category_to_channel[category_id]
        mask = decode_segmentation(
            annotation["segmentation"],
            height,
            width,
            annotation_label=_annotation_label(annotation),
        )
        masks[channel] |= mask

    explicit_normal = masks[0].copy()
    foreground_union = np.any(masks[1:], axis=0)
    masks[0] = explicit_normal | ~foreground_union
    return masks


def build_soft_target(
    class_masks: np.ndarray, *, epsilon: float = 1e-6
) -> np.ndarray:
    """Create a KxHxW probability target from K binary semantic masks."""

    masks = np.asarray(class_masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[0] < 1:
        raise ValueError("class_masks must have shape (K, H, W) with K >= 1")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")

    active_count = masks.sum(axis=0)
    if np.any(active_count == 0):
        coordinates = np.argwhere(active_count == 0)[0]
        raise ValueError(
            "Every pixel must have at least one active class; first empty pixel "
            f"is at (y={coordinates[0]}, x={coordinates[1]})"
        )

    target = np.zeros(masks.shape, dtype=np.float64)
    single = active_count == 1
    target[:, single] = masks[:, single].astype(np.float64)

    overlap = active_count >= 2
    if np.any(overlap):
        distances = np.stack(
            [distance_transform_edt(mask) for mask in masks], axis=0
        )
        weights = (distances + epsilon) * masks
        denominator = weights[:, overlap].sum(axis=0)
        if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
            raise RuntimeError("Invalid distance weights in overlap region")
        target[:, overlap] = weights[:, overlap] / denominator

    target = target.astype(np.float32)
    sums = target.sum(axis=0, dtype=np.float64)
    if not np.all(np.isfinite(target)):
        raise RuntimeError("Soft target contains NaN or infinity")
    if np.any(target < 0) or np.any(target > 1):
        raise RuntimeError("Soft target is outside [0, 1]")
    if not np.allclose(sums, 1.0, rtol=0.0, atol=1e-6):
        max_error = float(np.max(np.abs(sums - 1.0)))
        raise RuntimeError(f"Soft target simplex violation; max error={max_error}")
    if np.any(target[:, ~overlap] != masks[:, ~overlap]):
        raise RuntimeError("Non-overlap pixels are not strictly one-hot")
    if np.any(target[~masks] != 0):
        raise RuntimeError("Inactive classes received non-zero probability")
    return target


class JointResizeAndFlip:
    """Resize image/masks jointly, with optional train-time flips."""

    def __init__(
        self,
        size: int | tuple[int, int],
        *,
        horizontal_flip_probability: float = 0.0,
        vertical_flip_probability: float = 0.0,
    ) -> None:
        if isinstance(size, int):
            size = (size, size)
        self.height, self.width = size
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Transform size must be positive")
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability

    def __call__(
        self, image: np.ndarray, masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize(
            (self.width, self.height), resample=Image.Resampling.BILINEAR
        )
        resized_masks = np.stack(
            [
                np.asarray(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (self.width, self.height),
                        resample=Image.Resampling.NEAREST,
                    )
                )
                > 0
                for mask in masks
            ],
            axis=0,
        )
        image_array = np.asarray(pil_image, dtype=np.uint8)

        if random.random() < self.horizontal_flip_probability:
            image_array = np.flip(image_array, axis=1)
            resized_masks = np.flip(resized_masks, axis=2)
        if random.random() < self.vertical_flip_probability:
            image_array = np.flip(image_array, axis=0)
            resized_masks = np.flip(resized_masks, axis=1)
        return np.ascontiguousarray(image_array), np.ascontiguousarray(resized_masks)


class CocoSoftSegmentationDataset(Dataset):
    """PyTorch Dataset returning normalized images and selective Soft GT."""

    def __init__(
        self,
        annotation_path: str | Path,
        image_dir: str | Path,
        *,
        image_ids: Sequence[int] | None = None,
        transform: Callable[
            [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
        ]
        | None = None,
        epsilon: float = 1e-6,
        normal_name: str = "normal",
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        data, mapping, report, annotations_by_image = validate_coco_dataset(
            annotation_path, image_dir, normal_name=normal_name
        )
        self.annotation_path = Path(annotation_path)
        self.image_dir = Path(image_dir)
        self.mapping = mapping
        self.report = report
        self.annotations_by_image = annotations_by_image
        self.transform = transform
        self.epsilon = epsilon

        mean_array = np.asarray(mean, dtype=np.float32)
        std_array = np.asarray(std, dtype=np.float32)
        if mean_array.shape != (3,) or std_array.shape != (3,):
            raise ValueError("mean and std must each contain three values")
        if np.any(std_array <= 0):
            raise ValueError("std values must be positive")
        self.mean = mean_array[:, None, None]
        self.std = std_array[:, None, None]

        all_annotated_ids = set(annotations_by_image)
        if image_ids is None:
            selected_ids = all_annotated_ids
        else:
            selected_ids = set(image_ids)
            unknown = selected_ids - all_annotated_ids
            if unknown:
                raise DatasetValidationError(
                    "Requested image ids are not annotated: "
                    + ", ".join(map(str, sorted(unknown)))
                )
        self.images = [
            image
            for image in data["images"]
            if image["id"] in selected_ids
        ]
        if not self.images:
            raise DatasetValidationError("No annotated images selected")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_info = self.images[index]
        image_path = self.image_dir / image_info["file_name"]
        with Image.open(image_path) as pil_image:
            image = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)

        masks = build_class_masks(
            image_info,
            self.annotations_by_image[image_info["id"]],
            self.mapping,
        )
        if self.transform is not None:
            image, masks = self.transform(image, masks)
        target = build_soft_target(masks, epsilon=self.epsilon)

        image_chw = np.moveaxis(image.astype(np.float32) / 255.0, -1, 0)
        image_chw = (image_chw - self.mean) / self.std
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_chw)),
            "target": torch.from_numpy(target),
            "image_id": int(image_info["id"]),
            "file_name": image_info["file_name"],
        }


def _preflight_cli() -> int:
    parser = argparse.ArgumentParser(description="Validate a COCO soft-seg dataset")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--normal-name", default="normal")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip segmentation decoding (faster but less thorough)",
    )
    args = parser.parse_args()
    try:
        _, _, report, _ = validate_coco_dataset(
            args.annotations,
            args.images,
            normal_name=args.normal_name,
            decode_masks=not args.metadata_only,
        )
    except DatasetValidationError as exc:
        print(f"PRECHECK FAILED: {exc}")
        return 2
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_preflight_cli())
