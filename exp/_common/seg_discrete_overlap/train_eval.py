"""Single-stage, detector-free discrete-overlap semantic segmentation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent.parent
SEGMENTATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data_module = _load_module(
    "_seg_discrete_data", SEGMENTATION_ROOT / "data.py"
)
combination_module = _load_module(
    "_seg_discrete_combinations", SEGMENTATION_ROOT / "combinations.py"
)
metric_module = _load_module(
    "_seg_discrete_metrics", SEGMENTATION_ROOT / "discrete_metrics.py"
)

from metrics import format_metrics, metrics_with_class_names  # noqa: E402


CombinationCodec = combination_module.CombinationCodec
REPRESENTATION = combination_module.REPRESENTATION
build_combination_model = combination_module.build_combination_model
DiscreteSegmentationMetrics = metric_module.DiscreteSegmentationMetrics
per_image_discrete_metrics = metric_module.per_image_discrete_metrics
CocoSoftSegmentationDataset = data_module.CocoSoftSegmentationDataset
DatasetValidationError = data_module.DatasetValidationError
JointResizeAndFlip = data_module.JointResizeAndFlip
validate_coco_dataset = data_module.validate_coco_dataset


class JointResizeFlipRotate90(JointResizeAndFlip):
    """Square resize plus matched image/mask flips and 90-degree rotations."""

    def __call__(self, image: np.ndarray, masks: np.ndarray):
        image, masks = super().__call__(image, masks)
        if random.random() < 0.5:
            quadrants = random.randint(1, 3)
            image = np.rot90(image, quadrants, axes=(0, 1))
            masks = np.rot90(masks, quadrants, axes=(1, 2))
        return np.ascontiguousarray(image), np.ascontiguousarray(masks)


class CachedCocoSoftSegmentationDataset(CocoSoftSegmentationDataset):
    """Memory-mapped fixed-size RGB/state cache with dynamic augmentation."""

    CACHE_VERSION = 1

    def __init__(
        self,
        annotation_path: str | Path,
        image_dir: str | Path,
        *,
        cache_dir: str | Path,
        image_size: int,
        image_ids: Sequence[int] | None = None,
        normal_name: str = "normal",
        augment: bool = False,
    ) -> None:
        super().__init__(
            annotation_path,
            image_dir,
            image_ids=image_ids,
            normal_name=normal_name,
            transform=None,
        )
        self.cache_dir = Path(cache_dir)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_images_path = self.cache_dir / "images.npy"
        self.cache_states_path = self.cache_dir / "state_indices.npy"
        self.cache_manifest_path = self.cache_dir / "manifest.json"
        self._cached_images = None
        self._cached_states = None
        self._semantic_lookup = CombinationCodec(
            self.mapping.num_classes
        ).table.numpy()
        self._ensure_cache()

    def _identity(self) -> dict[str, Any]:
        annotation_stat = self.annotation_path.stat()
        records = []
        for info in self.images:
            image_path = self.image_dir / info["file_name"]
            image_stat = image_path.stat()
            records.append({
                "image_id": int(info["id"]),
                "file_name": str(info["file_name"]),
                "size": int(image_stat.st_size),
                "mtime_ns": int(image_stat.st_mtime_ns),
            })
        return {
            "cache_version": self.CACHE_VERSION,
            "representation": REPRESENTATION,
            "image_size": self.image_size,
            "annotations": str(self.annotation_path.resolve()),
            "annotations_size": int(annotation_stat.st_size),
            "annotations_mtime_ns": int(annotation_stat.st_mtime_ns),
            "image_dir": str(self.image_dir.resolve()),
            "mapping": self.mapping.to_dict(),
            "images": records,
        }

    def _cache_is_current(self, identity: dict[str, Any]) -> bool:
        if not (
            self.cache_manifest_path.is_file()
            and self.cache_images_path.is_file()
            and self.cache_states_path.is_file()
        ):
            return False
        try:
            existing = json.loads(self.cache_manifest_path.read_text(encoding="utf-8"))
            cached_images = np.load(
                self.cache_images_path, mmap_mode="r", allow_pickle=False
            )
            cached_states = np.load(
                self.cache_states_path, mmap_mode="r", allow_pickle=False
            )
        except (OSError, ValueError, EOFError):
            return False
        count = len(self.images)
        size = self.image_size
        arrays_valid = (
            cached_images.shape == (count, size, size, 3)
            and cached_images.dtype == np.uint8
            and cached_states.shape == (count, size, size)
            and cached_states.dtype == np.uint8
        )
        return existing.get("identity") == identity and arrays_valid

    def _ensure_cache(self) -> None:
        identity = self._identity()
        if self._cache_is_current(identity):
            return

        count = len(self.images)
        shape = (count, self.image_size, self.image_size)
        temporary_images = self.cache_dir / "images.building.npy"
        temporary_states = self.cache_dir / "state_indices.building.npy"
        image_store = np.lib.format.open_memmap(
            temporary_images,
            mode="w+",
            dtype=np.uint8,
            shape=(*shape, 3),
        )
        state_store = np.lib.format.open_memmap(
            temporary_states,
            mode="w+",
            dtype=np.uint8,
            shape=shape,
        )
        resize = JointResizeAndFlip(self.image_size)
        bits = (1 << np.arange(self.mapping.num_classes, dtype=np.uint16))[:, None, None]
        for index, info in enumerate(tqdm(self.images, desc="building resized cache")):
            with Image.open(self.image_dir / info["file_name"]) as handle:
                image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            masks = data_module.build_class_masks(
                info,
                self.annotations_by_image[info["id"]],
                self.mapping,
            )
            image, masks = resize(image, masks)
            bit_codes = (masks.astype(np.uint16) * bits).sum(axis=0)
            if np.any(bit_codes == 0):
                raise RuntimeError(f"Cache target contains an empty state: {info['file_name']}")
            image_store[index] = image
            state_store[index] = (bit_codes - 1).astype(np.uint8)
        image_store.flush()
        state_store.flush()
        del image_store, state_store
        temporary_images.replace(self.cache_images_path)
        temporary_states.replace(self.cache_states_path)
        save_json(self.cache_manifest_path, {
            "identity": identity,
            "image_array": {
                "dtype": "uint8",
                "shape": [count, self.image_size, self.image_size, 3],
            },
            "state_array": {
                "dtype": "uint8",
                "shape": [count, self.image_size, self.image_size],
                "meaning": "combination state index (bit code - 1)",
            },
        })

    def _open_cache(self) -> None:
        if self._cached_images is None:
            self._cached_images = np.load(
                self.cache_images_path, mmap_mode="r", allow_pickle=False
            )
            self._cached_states = np.load(
                self.cache_states_path, mmap_mode="r", allow_pickle=False
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cached_images"] = None
        state["_cached_states"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        self._open_cache()
        image = np.asarray(self._cached_images[index])
        states = np.asarray(self._cached_states[index])
        if self.augment:
            if random.random() < 0.5:
                image, states = np.flip(image, axis=1), np.flip(states, axis=1)
            if random.random() < 0.5:
                image, states = np.flip(image, axis=0), np.flip(states, axis=0)
            if random.random() < 0.5:
                quadrants = random.randint(1, 3)
                image = np.rot90(image, quadrants, axes=(0, 1))
                states = np.rot90(states, quadrants, axes=(0, 1))
        image = np.ascontiguousarray(image)
        states = np.ascontiguousarray(states)
        target = np.moveaxis(self._semantic_lookup[states], -1, 0)
        image_chw = np.moveaxis(image.astype(np.float32) / 255.0, -1, 0)
        image_chw = (image_chw - self.mean) / self.std
        info = self.images[index]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_chw)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "image_id": int(info["id"]),
            "file_name": str(info["file_name"]),
        }


class CocoFullResolutionSegmentationEvaluationDataset(Dataset):
    """Return native-resolution RGB images and exact uniform-overlap targets."""

    def __init__(
        self,
        annotation_path: str | Path,
        image_dir: str | Path,
        *,
        image_ids: Sequence[int] | None = None,
        normal_name: str = "normal",
    ) -> None:
        data, mapping, report, annotations = validate_coco_dataset(
            annotation_path, image_dir, normal_name=normal_name
        )
        available = set(annotations)
        selected = available if image_ids is None else {int(value) for value in image_ids}
        unknown = selected - available
        if unknown:
            raise DatasetValidationError(
                f"Requested image ids are not annotated: {sorted(unknown)}"
            )
        self.image_dir = Path(image_dir)
        self.mapping = mapping
        self.report = report
        self.annotations = annotations
        self.images = [item for item in data["images"] if item["id"] in selected]
        if not self.images:
            raise DatasetValidationError("No annotated images selected")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        info = self.images[index]
        with Image.open(self.image_dir / info["file_name"]) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        masks = data_module.build_class_masks(
            info, self.annotations[info["id"]], self.mapping
        )
        target = data_module.build_soft_target(masks)
        return {
            "image": image,
            "target": torch.from_numpy(target),
            "image_id": int(info["id"]),
            "file_name": str(info["file_name"]),
        }

ARCHITECTURE = "single_stage_discrete_overlap_v2_imbalance_aware"
LOSS_NAME = "state_weighted_focal_foreground_dice_expected_l1_v1"
LOSS_COMPONENTS = (
    "total",
    "focal_ce",
    "foreground_dice_loss",
    "expected_l1",
    "combination_ce",
)
BACKBONES = {
    "resnet50": "resnet50",
    "efficientnet-b4": "efficientnet-b4",
    "hrnet-w32": "tu-hrnet_w32",
}
BACKBONE_ALIASES = {
    **BACKBONES,
    "efficientnet_b4": "efficientnet-b4",
    "hrnet_w32": "tu-hrnet_w32",
    "tu-hrnet_w32": "tu-hrnet_w32",
}
DEFAULT_RESNET_CHECKPOINT = Path(
    Path(torch.hub.get_dir()) / "checkpoints" / "resnet50-0676ba61.pth"
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class ImbalanceAwareCombinationLoss(nn.Module):
    """Sparse-foreground loss over exact overlap states and semantic marginals."""

    def __init__(
        self,
        class_names: Sequence[str],
        *,
        focal_weight: float = 1.3,
        foreground_dice_weight: float = 0.7,
        expected_l1_weight: float = 1.0,
        focal_gamma: float = 2.8,
        normal_weight: float = 0.3,
        granulation_weight: float = 1.0,
        eschar_weight: float = 2.8,
        suppuration_weight: float = 3.8,
        other_foreground_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.codec = CombinationCodec(len(class_names))
        self.focal_weight = float(focal_weight)
        self.foreground_dice_weight = float(foreground_dice_weight)
        self.expected_l1_weight = float(expected_l1_weight)
        self.focal_gamma = float(focal_gamma)

        named_weights = {
            "normal": normal_weight,
            "background": normal_weight,
            "granulation": granulation_weight,
            "eschar": eschar_weight,
            "suppuration": suppuration_weight,
        }
        semantic_weights = []
        for channel, name in enumerate(class_names):
            if channel == 0:
                weight = normal_weight
            else:
                weight = named_weights.get(name.strip().lower(), other_foreground_weight)
            semantic_weights.append(float(weight))
        semantic = torch.tensor(semantic_weights, dtype=torch.float32)
        state_weights = torch.stack([
            semantic[row > 0].max() for row in self.codec.table
        ])
        states = self.codec.table
        self.register_buffer("semantic_weights", semantic)
        self.register_buffer("state_weights", state_weights)
        self.register_buffer(
            "costs", torch.abs(states[:, None, :] - states[None, :, :]).sum(dim=2)
        )

    def metadata(self, class_names: Sequence[str]) -> dict[str, Any]:
        return {
            "name": LOSS_NAME,
            "formula": (
                f"{self.focal_weight}*state_weighted_focal_ce + "
                f"{self.foreground_dice_weight}*foreground_semantic_dice + "
                f"{self.expected_l1_weight}*state_weighted_expected_l1"
            ),
            "focal_weight": self.focal_weight,
            "foreground_dice_weight": self.foreground_dice_weight,
            "expected_l1_weight": self.expected_l1_weight,
            "focal_gamma": self.focal_gamma,
            "semantic_class_weights": {
                str(name): float(weight)
                for name, weight in zip(class_names, self.semantic_weights.tolist())
            },
            "state_weights": self.state_weights.tolist(),
            "dice_channels": list(class_names[1:]),
        }

    def components(self, logits: torch.Tensor, target: torch.Tensor):
        if logits.ndim != 4 or target.ndim != 4:
            raise ValueError("logits and target must both have shape BxCxHxW")
        if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[2:]:
            raise ValueError("logits and target batch/spatial shapes differ")
        if logits.shape[1] != self.codec.num_states:
            raise ValueError("logit channel count does not match overlap states")
        if target.shape[1] != self.codec.num_classes:
            raise ValueError("target channel count does not match semantic classes")

        target_states = self.codec.encode(target)
        log_probabilities = F.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()
        true_log_probability = log_probabilities.gather(
            1, target_states.unsqueeze(1)
        ).squeeze(1)
        true_probability = true_log_probability.exp()
        pixel_weights = self.state_weights[target_states]

        combination_ce = -true_log_probability.mean()
        focal_pixels = (
            (1.0 - true_probability).pow(self.focal_gamma) * -true_log_probability
        )
        # Match the reference implementation's alpha-weighted pixel mean. In
        # particular, normal=0.3 reduces its absolute contribution instead of
        # being cancelled by normalization through the sum of weights.
        focal_ce = (focal_pixels * pixel_weights).mean()

        pixel_costs = self.costs[:, target_states].permute(1, 0, 2, 3)
        expected_l1_pixels = (probabilities * pixel_costs).sum(dim=1)
        expected_l1 = (expected_l1_pixels * pixel_weights).mean()

        semantic_probability = torch.einsum(
            "bshw,sc->bchw", probabilities, self.codec.table
        )
        foreground_probability = semantic_probability[:, 1:]
        foreground_target = target[:, 1:]
        present = foreground_target.sum(dim=(0, 2, 3)) > 0
        if bool(present.any()):
            intersection = (foreground_probability * foreground_target).sum(
                dim=(0, 2, 3)
            )
            denominator = (
                foreground_probability.sum(dim=(0, 2, 3))
                + foreground_target.sum(dim=(0, 2, 3))
            )
            foreground_dice_loss = 1.0 - (
                (2.0 * intersection + 1e-6) / (denominator + 1e-6)
            )[present].mean()
        else:
            foreground_dice_loss = foreground_probability.sum() * 0.0

        total = (
            self.focal_weight * focal_ce
            + self.foreground_dice_weight * foreground_dice_loss
            + self.expected_l1_weight * expected_l1
        )
        return {
            "total": total,
            "focal_ce": focal_ce,
            "foreground_dice_loss": foreground_dice_loss,
            "expected_l1": expected_l1,
            "combination_ce": combination_ce,
        }

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.components(logits, target)["total"]


def normalize_backbone(value: str) -> str:
    try:
        return BACKBONE_ALIASES[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backbone {value!r}; choose one of: {', '.join(BACKBONES)}"
        ) from exc


def prepare_encoder(args: argparse.Namespace) -> None:
    args.encoder = normalize_backbone(args.backbone)
    args.resolved_encoder_weights = None if args.encoder_weights == "none" else "imagenet"
    if (
        args.encoder_checkpoint is None
        and args.encoder == "resnet50"
        and args.encoder_weights == "imagenet"
        and DEFAULT_RESNET_CHECKPOINT.is_file()
    ):
        args.encoder_checkpoint = DEFAULT_RESNET_CHECKPOINT
        args.resolved_encoder_weights = None
    if args.encoder_checkpoint is not None and not args.encoder_checkpoint.is_file():
        raise FileNotFoundError(
            f"Encoder checkpoint does not exist: {args.encoder_checkpoint}"
        )
    if args.encoder_checkpoint is not None:
        args.resolved_encoder_weights = None


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def worker_seed(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "handler"
    }


def split_ids(ids: Iterable[int], fraction: float, seed: int):
    values = sorted(int(value) for value in ids)
    if len(values) < 2:
        raise DatasetValidationError("At least two annotated images are required for a split")
    if not 0 < fraction < 1:
        raise ValueError("split fraction must be between 0 and 1")
    random.Random(seed).shuffle(values)
    count = min(max(1, round(len(values) * fraction)), len(values) - 1)
    return sorted(values[count:]), sorted(values[:count])


def _external_split(path: Path, images: Path, mapping, normal_name: str):
    _, other_mapping, _, annotations = validate_coco_dataset(
        path, images, normal_name=normal_name
    )
    if other_mapping != mapping:
        raise DatasetValidationError("Train/validation/test category mappings differ")
    return path, images, sorted(annotations)


def dataset_splits(args: argparse.Namespace):
    _, mapping, _, annotations = validate_coco_dataset(
        args.annotations, args.images, normal_name=args.normal_name
    )
    if getattr(args, "split_file", None):
        split = json.loads(args.split_file.read_text(encoding="utf-8"))["splits"]
        groups = [list(map(int, split[key]["image_ids"])) for key in ("train", "val", "test")]
        if any(not group or len(group) != len(set(group)) for group in groups):
            raise DatasetValidationError("Fixed splits must contain unique nonempty ID lists")
        if any(set(groups[i]) & set(groups[j]) for i, j in ((0, 1), (0, 2), (1, 2))):
            raise DatasetValidationError("Fixed splits overlap")
        if any(set(group) - set(annotations) for group in groups):
            raise DatasetValidationError("Fixed split refers to missing annotated image IDs")
        return mapping, *((args.annotations, args.images, ids) for ids in groups)
    remaining = sorted(annotations)
    if args.test_annotations:
        test = _external_split(
            args.test_annotations, args.test_images or args.images, mapping, args.normal_name
        )
    else:
        remaining, ids = split_ids(remaining, args.test_fraction, args.seed)
        test = (args.annotations, args.images, ids)
    if args.val_annotations:
        validation = _external_split(
            args.val_annotations, args.val_images or args.images, mapping, args.normal_name
        )
    else:
        remaining, ids = split_ids(remaining, args.val_fraction, args.seed + 1)
        validation = (args.annotations, args.images, ids)
    train = (args.annotations, args.images, remaining)

    def image_paths(spec) -> set[str]:
        payload = json.loads(Path(spec[0]).read_text(encoding="utf-8"))
        selected = set(spec[2])
        return {
            str((Path(spec[1]) / item["file_name"]).resolve()).casefold()
            for item in payload["images"]
            if item["id"] in selected
        }

    groups = [image_paths(spec) for spec in (train, validation, test)]
    if any(not group for group in groups):
        raise DatasetValidationError("Train/validation/test must all contain images")
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise DatasetValidationError("Train/validation/test contain overlapping image paths")
    return mapping, train, validation, test


def split_metadata(train, validation, test):
    def describe(spec):
        return {
            "annotations": str(Path(spec[0]).resolve()),
            "images": str(Path(spec[1]).resolve()),
            "image_ids": sorted(int(value) for value in spec[2]),
        }

    return {
        "format_version": 1,
        "splits": {
            "train": describe(train),
            "val": describe(validation),
            "test": describe(test),
        },
    }


def build_model(
    num_classes: int,
    encoder: str,
    checkpoint: Path | None = None,
    encoder_weights: str | None = None,
):
    return build_combination_model(
        num_classes,
        encoder_name=normalize_backbone(encoder),
        encoder_weights=encoder_weights,
        encoder_checkpoint=checkpoint,
    )


def build_loss(args: argparse.Namespace, class_names: Sequence[str]):
    return ImbalanceAwareCombinationLoss(
        class_names,
        focal_weight=args.focal_weight,
        foreground_dice_weight=args.foreground_dice_weight,
        expected_l1_weight=args.expected_l1_weight,
        focal_gamma=args.focal_gamma,
        normal_weight=args.normal_weight,
        granulation_weight=args.granulation_weight,
        eschar_weight=args.eschar_weight,
        suppuration_weight=args.suppuration_weight,
        other_foreground_weight=args.other_foreground_weight,
    )


def set_encoder_trainable(model, trainable: bool) -> None:
    for parameter in model.network.encoder.parameters():
        parameter.requires_grad = trainable


def optimizer_learning_rates(optimizer) -> dict[str, float]:
    return {
        str(group.get("name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def loader_options(args: argparse.Namespace, device: torch.device):
    options: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and device.type == "cuda",
        "worker_init_fn": worker_seed,
    }
    if args.num_workers:
        options.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)
    return options


def train_epoch(model, loader, optimizer, criterion, device, *, encoder_frozen: bool):
    model.train()
    if encoder_frozen:
        model.network.encoder.eval()
    totals = {
        key: torch.zeros((), device=device)
        for key in LOSS_COMPONENTS
    }
    samples = 0
    for batch in tqdm(loader, desc="segmenter train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        components = criterion.components(model(image), target)
        optimizer.zero_grad(set_to_none=True)
        components["total"].backward()
        optimizer.step()
        size = len(image)
        samples += size
        for key in totals:
            totals[key] += components[key].detach() * size
    if not samples:
        raise RuntimeError("Training loader produced no batches")
    return {key: float(value) / samples for key, value in totals.items()}


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, mapping):
    model.eval()
    totals = {
        key: torch.zeros((), device=device)
        for key in LOSS_COMPONENTS
    }
    accumulator = DiscreteSegmentationMetrics(mapping.num_classes)
    samples = 0
    for batch in tqdm(loader, desc="segmenter validation", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        logits = model(image)
        components = criterion.components(logits, target)
        accumulator.update(model.codec.hard_probabilities(logits), target)
        size = len(image)
        samples += size
        for key in totals:
            totals[key] += components[key].detach() * size
    if not samples:
        raise RuntimeError("Validation loader produced no batches")
    losses = {key: float(value) / samples for key, value in totals.items()}
    metrics = metrics_with_class_names(accumulator.compute(), mapping.channel_to_name)
    return losses, metrics


def save_checkpoint(
    path: Path, *, model, optimizer, scheduler, epoch: int, mapping,
    args: argparse.Namespace, data_split, record, best_score: float, bad_epochs: int,
    criterion,
) -> None:
    payload = {
        "format_version": 2,
        "architecture": ARCHITECTURE,
        "representation": REPRESENTATION,
        "stage": "segmenter",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "mapping": mapping.to_dict(),
        "combination_mapping": CombinationCodec(mapping.num_classes).metadata(
            mapping.channel_to_name
        ),
        "loss": criterion.metadata(mapping.channel_to_name),
        "backbone": args.backbone,
        "encoder_name": args.encoder,
        "args": serializable_args(args),
        "data_split": data_split,
        "record": record,
        "training_state": {
            "best_score": best_score,
            "epochs_without_improvement": bad_epochs,
            "selection_metric": "soft_dice_foreground_macro",
            "selection_mode": "max",
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Segmenter checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("architecture") != ARCHITECTURE
        or payload.get("stage") != "segmenter"
        or payload.get("representation") != REPRESENTATION
    ):
        raise RuntimeError(f"Not a {ARCHITECTURE} checkpoint: {path}")
    mapping = payload.get("mapping", {})
    names = [item["category_name"] for item in mapping.get("channels", [])]
    expected = CombinationCodec(int(mapping.get("num_classes", 0))).metadata(names)
    if payload.get("combination_mapping") != expected:
        raise RuntimeError("Checkpoint combination mapping is missing or incompatible")
    return payload


def normalise_image(image: np.ndarray) -> torch.Tensor:
    array = np.moveaxis(image.astype(np.float32) / 255.0, -1, 0)
    mean = np.asarray((0.485, 0.456, 0.406), np.float32)[:, None, None]
    std = np.asarray((0.229, 0.224, 0.225), np.float32)[:, None, None]
    return torch.from_numpy(np.ascontiguousarray((array - mean) / std))


@torch.no_grad()
def predict_image(model, image: np.ndarray, device: torch.device, image_size: int):
    resized = np.asarray(
        Image.fromarray(image).resize((image_size, image_size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    logits = model(normalise_image(resized).unsqueeze(0).to(device))
    logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
    return model.codec.hard_probabilities(logits)[0].cpu().numpy()


def safe_name(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip(" ._")[:120] or "unnamed"


def save_probability_png(probability: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.clip(np.rint(probability * 65535), 0, 65535).astype(np.uint16)
    ).save(path)


def save_discrete(probability: np.ndarray, directory: Path, output: Path):
    states = CombinationCodec(probability.shape[0]).encode(
        torch.from_numpy(probability).unsqueeze(0)
    )[0]
    codes = (states.numpy() + 1).astype(np.uint16)
    directory.mkdir(parents=True, exist_ok=True)
    code_path = directory / "combination_codes.png"
    archive_path = directory / "probabilities.npz"
    Image.fromarray(codes).save(code_path)
    np.savez_compressed(
        archive_path,
        probabilities=probability.astype(np.float32),
        combination_codes=codes,
    )
    return {
        "combination_codes": code_path.relative_to(output).as_posix(),
        "probability_archive": archive_path.relative_to(output).as_posix(),
    }


def encoding_metadata():
    return {
        "combination_codes": "Exact uint16 bit code; bit c indicates semantic channel c",
        "probability_archive": "NPZ with float32 CxHxW probabilities and uint16 HxW codes",
        "probability_png": "16-bit preview; stored_value / 65535",
        "authoritative": "combination_codes",
    }


@torch.no_grad()
def export_predictions(model, dataset, device, output: Path, image_size: int, limit: int = 0):
    model.eval()
    accumulator = DiscreteSegmentationMetrics(dataset.mapping.num_classes)
    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    count = min(len(dataset), limit) if limit else len(dataset)
    for index in tqdm(range(count), desc="full-image discrete export", leave=False):
        sample = dataset[index]
        prediction_np = predict_image(model, sample["image"], device, image_size)
        prediction = torch.from_numpy(prediction_np).unsqueeze(0)
        target = sample["target"].unsqueeze(0)
        accumulator.update(prediction, target)
        image_id, file_name = int(sample["image_id"]), str(sample["file_name"])
        rows.extend(per_image_discrete_metrics(
            prediction, target, class_names=dataset.mapping.channel_to_name,
            image_ids=[image_id], file_names=[file_name],
        ))
        key = f"{image_id:06d}_{safe_name(Path(file_name).stem)}"
        root = output / "probabilities" / key
        prediction_root, target_root = root / "prediction", root / "ground_truth"
        target_np = target[0].numpy()
        exact_prediction = save_discrete(prediction_np, prediction_root, output)
        exact_target = save_discrete(target_np, target_root, output)
        maps = []
        for channel, class_name in enumerate(dataset.mapping.channel_to_name):
            if channel == 0:
                continue
            filename = f"{channel:02d}_{safe_name(class_name)}.png"
            save_probability_png(prediction_np[channel], prediction_root / filename)
            save_probability_png(target_np[channel], target_root / filename)
            maps.append({
                "channel": channel,
                "class_name": class_name,
                "prediction_probability_map": str(
                    (Path("probabilities") / key / "prediction" / filename).as_posix()
                ),
                "ground_truth_probability_map": str(
                    (Path("probabilities") / key / "ground_truth" / filename).as_posix()
                ),
            })
        manifest.append({
            "image_id": image_id,
            "file_name": file_name,
            "height": int(sample["image"].shape[0]),
            "width": int(sample["image"].shape[1]),
            "model_input_size": [image_size, image_size],
            "prediction": exact_prediction,
            "ground_truth": exact_target,
            "positive_class_probability_maps": maps,
        })
    if not rows:
        raise RuntimeError("Export dataset produced no images")
    metrics = metrics_with_class_names(accumulator.compute(), dataset.mapping.channel_to_name)
    metrics.update({
        "scope": "single-stage full-image segmentation",
        "architecture": ARCHITECTURE,
        "representation": REPRESENTATION,
        "images": len(manifest),
        "model_input_size": [image_size, image_size],
    })
    save_json(output / "metrics.json", metrics)
    save_json(output / "mask_manifest.json", {
        "encoding": encoding_metadata(),
        "combination_mapping": CombinationCodec(dataset.mapping.num_classes).metadata(
            dataset.mapping.channel_to_name
        ),
        "images": manifest,
    })
    with (output / "per_image_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return metrics


def preflight(args: argparse.Namespace) -> int:
    _, mapping, report, _ = validate_coco_dataset(
        args.annotations, args.images, normal_name=args.normal_name,
        decode_masks=not args.metadata_only,
    )
    payload = report.to_dict()
    payload.update({
        "architecture": ARCHITECTURE,
        "input_mode": "full-image-square-resize",
        "combination_mapping": CombinationCodec(mapping.num_classes).metadata(
            mapping.channel_to_name
        ),
    })
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def train(args: argparse.Namespace) -> int:
    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0:
        raise ValueError("epochs, batch size and image size must be positive")
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("num_workers must be non-negative and prefetch_factor positive")
    if args.encoder_learning_rate <= 0 or args.decoder_learning_rate <= 0:
        raise ValueError("encoder and decoder learning rates must be positive")
    loss_weights = (
        args.focal_weight,
        args.foreground_dice_weight,
        args.expected_l1_weight,
    )
    class_weights = (
        args.normal_weight,
        args.granulation_weight,
        args.eschar_weight,
        args.suppuration_weight,
        args.other_foreground_weight,
    )
    if args.weight_decay < 0 or any(value < 0 for value in loss_weights):
        raise ValueError("loss weights and weight decay must be non-negative")
    if any(value <= 0 for value in class_weights) or args.focal_gamma < 0:
        raise ValueError("class weights must be positive and focal gamma non-negative")
    if args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0:
        raise ValueError("early-stopping arguments must be non-negative")
    if args.scheduler_t0 <= 0 or args.scheduler_t_mult < 1:
        raise ValueError("scheduler T0 must be positive and T-mult at least 1")
    if args.freeze_encoder_epochs < 0:
        raise ValueError("freeze encoder epochs must be non-negative")
    if args.min_learning_rate < 0 or args.max_export_images < 0:
        raise ValueError("minimum learning rate and max export images must be non-negative")
    prepare_encoder(args)
    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    mapping, train_spec, val_spec, test_spec = dataset_splits(args)
    split = split_metadata(train_spec, val_spec, test_spec)
    output = args.output / "segmenter"
    if output.exists() and not args.resume and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}; use --resume or another path")
    output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "config.json", serializable_args(args))
    save_json(args.output / "channel_mapping.json", mapping.to_dict())
    save_json(args.output / "combination_mapping.json", CombinationCodec(
        mapping.num_classes
    ).metadata(mapping.channel_to_name))
    save_json(args.output / "data_split.json", split)

    if args.cache_resized:
        cache_root = args.cache_dir or (args.output / "resized_cache")
        train_data = CachedCocoSoftSegmentationDataset(
            train_spec[0],
            train_spec[1],
            image_ids=train_spec[2],
            normal_name=args.normal_name,
            cache_dir=cache_root / "train",
            image_size=args.image_size,
            augment=True,
        )
        val_data = CachedCocoSoftSegmentationDataset(
            val_spec[0],
            val_spec[1],
            image_ids=val_spec[2],
            normal_name=args.normal_name,
            cache_dir=cache_root / "val",
            image_size=args.image_size,
            augment=False,
        )
        print(f"resized cache={cache_root.resolve()}")
    else:
        train_data = CocoSoftSegmentationDataset(
            train_spec[0], train_spec[1], image_ids=train_spec[2],
            normal_name=args.normal_name,
            transform=JointResizeFlipRotate90(
                args.image_size, horizontal_flip_probability=0.5,
                vertical_flip_probability=0.5,
            ),
        )
        val_data = CocoSoftSegmentationDataset(
            val_spec[0], val_spec[1], image_ids=val_spec[2],
            normal_name=args.normal_name,
            transform=JointResizeAndFlip(args.image_size),
        )
    generator = torch.Generator().manual_seed(args.seed)
    options = loader_options(args, device)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(val_data, shuffle=False, **options)
    model = build_model(
        mapping.num_classes,
        args.encoder,
        args.encoder_checkpoint,
        args.resolved_encoder_weights,
    ).to(device)
    criterion = build_loss(args, mapping.channel_to_name).to(device)
    save_json(args.output / "loss_config.json", criterion.metadata(mapping.channel_to_name))
    encoder_parameters = list(model.network.encoder.parameters())
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in encoder_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": args.encoder_learning_rate,
                "name": "encoder",
            },
            {
                "params": decoder_parameters,
                "lr": args.decoder_learning_rate,
                "name": "decoder",
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.scheduler_t0,
        T_mult=args.scheduler_t_mult,
        eta_min=args.min_learning_rate,
    )
    history_path, last_path = output / "history.json", output / "last.pt"
    history: list[dict[str, Any]] = []
    start_epoch, best_score, bad_epochs = 1, float("-inf"), 0
    if args.resume and last_path.is_file():
        saved = load_checkpoint(last_path, device)
        if saved["mapping"] != mapping.to_dict() or saved["data_split"] != split:
            raise RuntimeError("Resume checkpoint mapping or data split differs")
        if saved.get("encoder_name") != args.encoder:
            raise RuntimeError(
                "Resume checkpoint backbone differs: "
                f"saved={saved.get('encoder_name')!r}, requested={args.encoder!r}"
            )
        if saved.get("loss") != criterion.metadata(mapping.channel_to_name):
            raise RuntimeError("Resume checkpoint loss configuration differs")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        state = saved["training_state"]
        start_epoch = int(saved["epoch"]) + 1
        best_score = float(state["best_score"])
        bad_epochs = int(state["epochs_without_improvement"])
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"resumed segmenter at epoch={start_epoch}")

    print(
        f"device={device}; backbone={args.backbone} ({args.encoder}); "
        f"pretrained={args.resolved_encoder_weights or args.encoder_checkpoint or 'none'}; "
        f"train={len(train_data)} val={len(val_data)} test={len(test_spec[2])}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        encoder_frozen = epoch <= args.freeze_encoder_epochs
        set_encoder_trainable(model, not encoder_frozen)
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            encoder_frozen=encoder_frozen,
        )
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, mapping)
        score = float(val_metrics["soft_dice_foreground_macro"])
        improved = score > best_score + args.early_stopping_min_delta
        if improved:
            best_score, bad_epochs = score, 0
        else:
            bad_epochs += 1
        learning_rates = optimizer_learning_rates(optimizer)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_metrics": val_metrics,
            "selection_metric": "soft_dice_foreground_macro",
            "selection_score": score,
            "encoder_frozen": encoder_frozen,
            "learning_rates": learning_rates,
            "next_learning_rates": optimizer_learning_rates(optimizer),
        }
        history.append(record)
        save_json(history_path, history)
        values = dict(
            model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
            mapping=mapping, args=args, data_split=split, record=record,
            best_score=best_score, bad_epochs=bad_epochs, criterion=criterion,
        )
        save_checkpoint(last_path, **values)
        if improved:
            save_checkpoint(output / "best.pt", **values)
        print(
            f"epoch={epoch} foreground_dice={score:.6f} "
            f"best={best_score:.6f} bad_epochs={bad_epochs} "
            f"encoder={'frozen' if encoder_frozen else 'trainable'}"
        )
        if args.early_stopping_patience and bad_epochs >= args.early_stopping_patience:
            print(f"early stopping at epoch={epoch}")
            break

    if args.export_after_train:
        selected = output / "best.pt"
        if not selected.is_file():
            selected = last_path
        model.load_state_dict(load_checkpoint(selected, device)["model_state_dict"])
        for name, spec in (("train", train_spec), ("val", val_spec), ("test", test_spec)):
            dataset = CocoFullResolutionSegmentationEvaluationDataset(
                spec[0], spec[1], image_ids=spec[2], normal_name=args.normal_name
            )
            metrics = export_predictions(
                model, dataset, device, args.output / "export" / name,
                args.image_size, args.max_export_images,
            )
            print(f"single_stage_export_metrics[{name}]\n{format_metrics(metrics)}")
    return 0


def load_inference_model(path: Path, device: torch.device):
    payload = load_checkpoint(path, device)
    encoder = str(payload["encoder_name"])
    model = build_model(int(payload["mapping"]["num_classes"]), encoder).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def predict(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, payload = load_inference_model(args.checkpoint, device)
    size = args.image_size or int(payload["args"].get("image_size", 256))
    paths = [args.input] if args.input.is_file() else sorted(
        path for path in args.input.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No supported image files found in {args.input}")
    mapping = payload["mapping"]
    names = tuple(item["category_name"] for item in mapping["channels"])
    manifest = {
        "architecture": ARCHITECTURE,
        "mapping": mapping,
        "encoding": encoding_metadata(),
        "combination_mapping": payload["combination_mapping"],
        "model_input_size": [size, size],
        "images": [],
    }
    for index, path in enumerate(tqdm(paths, desc="single-stage predict")):
        with Image.open(path) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        probability = predict_image(model, image, device, size)
        key = f"{index:06d}_{safe_name(path.stem)}"
        root = args.output / "probabilities" / key
        exact = save_discrete(probability, root, args.output)
        maps = []
        for channel, name in enumerate(names):
            if channel == 0:
                continue
            filename = f"{channel:02d}_{safe_name(name)}.png"
            save_probability_png(probability[channel], root / filename)
            maps.append({
                "channel": channel, "class_name": name,
                "probability_map": str((Path("probabilities") / key / filename).as_posix()),
            })
        manifest["images"].append({
            "input": str(path), "height": int(image.shape[0]), "width": int(image.shape[1]),
            "prediction": exact, "positive_class_probability_maps": maps,
        })
    save_json(args.output / "predictions.json", manifest)
    return 0


def export(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, payload = load_inference_model(args.checkpoint, device)
    spec = payload.get("data_split", {}).get("splits", {}).get(args.split)
    annotations = Path(spec["annotations"]) if spec else args.annotations
    images = Path(spec["images"]) if spec else args.images
    image_ids = spec["image_ids"] if spec else None
    dataset = CocoFullResolutionSegmentationEvaluationDataset(
        annotations, images, image_ids=image_ids, normal_name=args.normal_name
    )
    if dataset.mapping.to_dict() != payload["mapping"]:
        raise DatasetValidationError(
            "Checkpoint and export dataset category/channel mappings differ"
        )
    size = args.image_size or int(payload["args"].get("image_size", 256))
    output = args.output / args.split
    save_json(output / "channel_mapping.json", dataset.mapping.to_dict())
    print(format_metrics(export_predictions(
        model, dataset, device, output, size, args.max_images
    )))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-stage full-image discrete-overlap segmentation (no detector)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    defaults = {
        "annotations": Path("dataset/generalHospital/label/annotations.json"),
        "images": Path("dataset/generalHospital"),
    }
    check = sub.add_parser("preflight")
    check.add_argument("--annotations", type=Path, default=defaults["annotations"])
    check.add_argument("--images", type=Path, default=defaults["images"])
    check.add_argument("--normal-name", default="normal")
    check.add_argument("--metadata-only", action="store_true")
    check.set_defaults(handler=preflight)

    training = sub.add_parser("train")
    training.add_argument("--annotations", type=Path, default=defaults["annotations"])
    training.add_argument("--images", type=Path, default=defaults["images"])
    training.add_argument("--val-annotations", type=Path)
    training.add_argument("--val-images", type=Path)
    training.add_argument("--test-annotations", type=Path)
    training.add_argument("--test-images", type=Path)
    training.add_argument("--output", type=Path, default=Path("results/table1_comparison/training/ours"))
    training.add_argument("--split-file", type=Path, help="Use the recorded train/val/test image IDs")
    training.add_argument("--normal-name", default="normal")
    training.add_argument("--epochs", type=int, default=200)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--encoder-learning-rate", type=float, default=7.5e-5)
    training.add_argument("--decoder-learning-rate", type=float, default=7.5e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--image-size", type=int, default=512)
    training.add_argument(
        "--cache-resized", action=argparse.BooleanOptionalAction, default=True
    )
    training.add_argument("--cache-dir", type=Path)
    training.add_argument("--focal-weight", type=float, default=1.3)
    training.add_argument("--foreground-dice-weight", type=float, default=0.7)
    training.add_argument("--expected-l1-weight", type=float, default=1.0)
    training.add_argument("--focal-gamma", type=float, default=2.8)
    training.add_argument("--normal-weight", type=float, default=0.3)
    training.add_argument("--granulation-weight", type=float, default=1.0)
    training.add_argument("--eschar-weight", type=float, default=2.8)
    training.add_argument("--suppuration-weight", type=float, default=3.8)
    training.add_argument("--other-foreground-weight", type=float, default=1.0)
    training.add_argument("--backbone", choices=tuple(BACKBONES), default="resnet50")
    training.add_argument("--encoder-checkpoint", type=Path)
    training.add_argument(
        "--encoder-weights", choices=("imagenet", "none"), default="imagenet"
    )
    training.add_argument("--freeze-encoder-epochs", type=int, default=3)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", default="auto")
    training.add_argument("--num-workers", type=int, default=8)
    training.add_argument("--prefetch-factor", type=int, default=8)
    training.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--val-fraction", type=float, default=0.2)
    training.add_argument("--test-fraction", type=float, default=0.2)
    training.add_argument("--early-stopping-patience", type=int, default=18)
    training.add_argument("--early-stopping-min-delta", type=float, default=0.0001)
    training.add_argument("--scheduler-t0", type=int, default=20)
    training.add_argument("--scheduler-t-mult", type=int, default=2)
    training.add_argument("--min-learning-rate", type=float, default=1e-7)
    training.add_argument("--resume", action="store_true")
    training.add_argument("--export-after-train", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--max-export-images", type=int, default=0)
    training.set_defaults(handler=train)

    inference = sub.add_parser("predict")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output", type=Path, default=Path("results/predictions"))
    inference.add_argument("--image-size", type=int)
    inference.add_argument("--device", default="auto")
    inference.set_defaults(handler=predict)

    exporter = sub.add_parser("export")
    exporter.add_argument("--checkpoint", type=Path, required=True)
    exporter.add_argument("--annotations", type=Path, default=defaults["annotations"])
    exporter.add_argument("--images", type=Path, default=defaults["images"])
    exporter.add_argument("--output", type=Path, default=Path("results/export"))
    exporter.add_argument("--normal-name", default="normal")
    exporter.add_argument("--image-size", type=int)
    exporter.add_argument("--device", default="auto")
    exporter.add_argument("--max-images", type=int, default=0)
    exporter.add_argument("--split", choices=("train", "val", "test"), default="test")
    exporter.set_defaults(handler=export)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except DatasetValidationError as exc:
        print(f"DATASET VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
