"""Train FCN-ResNet50 Dice+Focal baselines on the fixed Table 1 split.

``single_label`` uses the same exclusive masks as Single-label UNet++.
``lookup_multilabel`` predicts the same 15 legal class sets as Ours and decodes
the winning state through the uniform-membership lookup table.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


from _common import settings as s
ROOT = s.REPO
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import metrics_with_class_names
from seg_discrete_overlap.discrete_metrics import DiscreteSegmentationMetrics
from seg_discrete_overlap.train_eval import (
    CachedCocoSoftSegmentationDataset,
    ImbalanceAwareCombinationLoss,
)
from table1_fcn import MODES, Table1FCN, exclusive_dice_focal_loss


SOURCE = s.data_path("exclusive")
SPLIT_PATH = s.data_path("split")
ANNOTATIONS = s.data_path("annotations")
IMAGES = s.data_path("images")
CACHE = s.output() / "cache"
OUTPUTS = {
    "single_label": s.output() / "training" / "fcn_single_label",
    "lookup_multilabel": s.output() / "training" / "fcn_lookup_multilabel",
}
SIZE = int(s.current()["training"]["image_size"])
SEED = int(s.current()["seed"])


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def split_metadata() -> dict:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def split_names() -> dict[str, list[str]]:
    split = split_metadata()["splits"]
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    names = {int(item["id"]): item["file_name"] for item in annotations["images"]}
    return {
        key: [names[int(image_id)] for image_id in split[key]["image_ids"]]
        for key in ("train", "val", "test")
    }


class ExclusiveDataset(Dataset):
    def __init__(self, names: list[str], *, augment: bool, allow_missing: bool = False):
        self.augment = augment
        image_by_stem = {
            path.stem.casefold(): path
            for path in (SOURCE if any(SOURCE.glob("*.png")) else SOURCE / "images").iterdir()
            if path.is_file() and path.suffix.casefold() in {".png", ".jpg", ".jpeg"}
        }
        mask_by_stem = {
            path.stem.casefold(): path
            for path in (SOURCE / "mask").iterdir()
            if path.is_file() and path.suffix.casefold() == ".png"
        }
        images, masks = [], []
        self.missing_names = []
        for name in tqdm(names, desc="cache FCN single-label data", leave=False):
            stem = Path(name).stem.casefold()
            image_path, mask_path = image_by_stem.get(stem), mask_by_stem.get(stem)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path else None
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
            if image is None or mask is None:
                if allow_missing:
                    self.missing_names.append(name)
                    continue
                raise FileNotFoundError(f"Missing single-label image/mask for {name}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))
            masks.append(cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST))
        self.images = np.stack(images).astype(np.uint8)
        self.masks = np.stack(masks).astype(np.uint8)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image, mask = self.images[index], self.masks[index]
        if self.augment:
            if torch.rand(()) < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if torch.rand(()) < 0.5:
                image, mask = image[::-1], mask[::-1]
            turns = int(torch.randint(0, 4, ()).item())
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        image = np.ascontiguousarray(image).astype(np.float32) / 255.0
        image = (image - np.asarray((0.485, 0.456, 0.406), np.float32)) / np.asarray(
            (0.229, 0.224, 0.225), np.float32
        )
        target = np.ascontiguousarray(mask).astype(np.int64) - 1
        target[mask == 0] = 255
        return torch.from_numpy(np.moveaxis(image, -1, 0)), torch.from_numpy(target)


@torch.no_grad()
def validate_single(model, loader, device) -> float:
    model.eval()
    intersection = torch.zeros(5, dtype=torch.float64)
    denominator = torch.zeros(5, dtype=torch.float64)
    for images, targets in loader:
        labels = model(images.to(device, non_blocking=True))["out"].argmax(1).cpu()
        valid = targets != 255
        for channel in (1, 2, 4):
            prediction, truth = (labels == channel) & valid, targets == channel
            intersection[channel] += (prediction & truth).sum()
            denominator[channel] += prediction.sum() + truth.sum()
    dice = (2.0 * intersection / denominator.clamp_min(1)).tolist()
    return float(np.mean([dice[1], dice[2], dice[4]]))


@torch.no_grad()
def validate_lookup(model, loader, device) -> float:
    model.eval()
    accumulator = DiscreteSegmentationMetrics(4)
    for batch in loader:
        target = batch["target"]
        logits = model(batch["image"].to(device, non_blocking=True))["out"]
        accumulator.update(model.decode(logits).cpu(), target)
    metrics = metrics_with_class_names(
        accumulator.compute(), ("normal", "eschar", "granulation", "suppuration")
    )
    return float(metrics["soft_dice_foreground_macro"])


def checkpoint_payload(model, mode, epoch, score, args, missing_names) -> dict:
    return {
        "format_version": 1,
        "architecture": "torchvision_fcn_resnet50",
        "mode": mode,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "image_size": SIZE,
        "data_split": split_metadata(),
        "loss": (
            "exclusive_dice_plus_focal"
            if mode == "single_label"
            else "1.3*state_weighted_focal_plus_0.7*foreground_semantic_dice"
        ),
        "selection_metric": "three_tissue_macro_dice",
        "selection_score": score,
        "training_images_missing_single_label_mask": missing_names,
        "args": vars(args),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--early-stopping-patience", type=int, default=18)
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    output = args.output or OUTPUTS[args.mode]
    segmenter = output / "segmenter"
    if segmenter.exists() and any(segmenter.iterdir()):
        raise FileExistsError(f"Output is not empty: {segmenter}")
    segmenter.mkdir(parents=True, exist_ok=True)

    missing_names: list[str] = []
    if args.mode == "single_label":
        names = split_names()
        train_data = ExclusiveDataset(names["train"], augment=True, allow_missing=True)
        val_data = ExclusiveDataset(names["val"], augment=False)
        missing_names = train_data.missing_names
    else:
        split = split_metadata()["splits"]
        train_data = CachedCocoSoftSegmentationDataset(
            ANNOTATIONS, IMAGES, image_ids=split["train"]["image_ids"],
            cache_dir=CACHE / "train", image_size=SIZE, augment=True,
        )
        val_data = CachedCocoSoftSegmentationDataset(
            ANNOTATIONS, IMAGES, image_ids=split["val"]["image_ids"],
            cache_dir=CACHE / "val", image_size=SIZE, augment=False,
        )
    generator = torch.Generator().manual_seed(SEED)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_data, shuffle=True, generator=generator, **loader_options
    )
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    model = Table1FCN(args.mode, pretrained=True).to(device)
    lookup_loss = None
    if args.mode == "lookup_multilabel":
        lookup_loss = ImbalanceAwareCombinationLoss(
            ("normal", "eschar", "granulation", "suppuration"),
            focal_weight=1.3,
            foreground_dice_weight=0.7,
            expected_l1_weight=0.0,
            focal_gamma=2.8,
            normal_weight=0.3,
            granulation_weight=1.0,
            eschar_weight=2.8,
            suppuration_weight=3.8,
        ).to(device)
    backbone_parameters = list(model.network.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 7.5e-5},
            {"params": head_parameters, "lr": 7.5e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )
    history, best_score, bad_epochs = [], float("-inf"), 0
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= 3
        for parameter in model.network.backbone.parameters():
            parameter.requires_grad = not frozen
        model.train()
        if frozen:
            model.network.backbone.eval()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"FCN {args.mode} train", leave=False):
            if args.mode == "single_label":
                images, target = batch
            else:
                images, target = batch["image"], batch["target"]
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            outputs = model(images)
            if args.mode == "single_label":
                loss = exclusive_dice_focal_loss(outputs["out"], target)
                loss = loss + 0.5 * exclusive_dice_focal_loss(outputs["aux"], target)
            else:
                loss = lookup_loss(outputs["out"], target)
                loss = loss + 0.5 * lookup_loss(outputs["aux"], target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(images)
        score = (
            validate_single(model, val_loader, device)
            if args.mode == "single_label"
            else validate_lookup(model, val_loader, device)
        )
        scheduler.step()
        improved = score > best_score + 1e-4
        if improved:
            best_score, bad_epochs = score, 0
        else:
            bad_epochs += 1
        history.append({
            "epoch": epoch,
            "train_loss": running_loss / len(train_data),
            "validation_macro_dice": score,
            "best_score": best_score,
            "epochs_without_improvement": bad_epochs,
        })
        save_json(segmenter / "history.json", history)
        if improved:
            temporary = segmenter / "best.pt.tmp"
            torch.save(
                checkpoint_payload(
                    model, args.mode, epoch, score, args, missing_names
                ),
                temporary,
            )
            temporary.replace(segmenter / "best.pt")
        print(
            f"epoch={epoch} validation_macro_dice={score:.6f} "
            f"best={best_score:.6f} bad_epochs={bad_epochs}"
        )
        if bad_epochs >= args.early_stopping_patience:
            print(f"early stopping at epoch={epoch}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
