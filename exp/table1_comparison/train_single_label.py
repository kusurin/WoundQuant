"""Train the WoundSeg_general exclusive-label UNet++ on the fixed split."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


from _common import settings as s
ROOT = s.REPO
SOURCE = s.data_path("exclusive")
OUTPUT = s.output() / "training" / "single_label_unetplusplus"
SPLIT_PATH = s.data_path("split")
ANNOTATIONS = s.data_path("annotations")
SIZE = int(s.current()["training"]["image_size"])
SEED = int(s.current()["seed"])


def save_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def split_names() -> dict[str, list[str]]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["splits"]
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    names = {int(item["id"]): item["file_name"] for item in annotations["images"]}
    return {
        key: [names[int(image_id)] for image_id in split[key]["image_ids"]]
        for key in ("train", "val", "test")
    }


class ExclusiveDataset(Dataset):
    def __init__(self, names: list[str], augment: bool, allow_missing: bool = False):
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
        for name in tqdm(names, desc="cache single-label data", leave=False):
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
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


def dice_focal_loss(logits, target):
    valid = target != 255
    safe_target = target.masked_fill(~valid, 0)
    probabilities = logits.softmax(1)
    one_hot = F.one_hot(safe_target, 5).permute(0, 3, 1, 2).float()
    valid_channel = valid.unsqueeze(1)
    intersection = (probabilities * one_hot * valid_channel).sum((0, 2, 3))
    denominator = ((probabilities + one_hot) * valid_channel).sum((0, 2, 3))
    dice_loss = 1.0 - ((2 * intersection + 1.0) / (denominator + 1.0)).mean()
    cross_entropy = F.cross_entropy(logits, target, ignore_index=255, reduction="none")
    true_probability = probabilities.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    focal = (((1 - true_probability) ** 2.0) * cross_entropy)[valid].mean()
    return dice_loss + focal


@torch.no_grad()
def validation_macro(model, loader, device):
    model.eval()
    intersection = torch.zeros(5, dtype=torch.float64)
    denominator = torch.zeros(5, dtype=torch.float64)
    for images, targets in loader:
        labels = model(images.to(device, non_blocking=True)).argmax(1).cpu()
        valid = targets != 255
        for channel in (1, 2, 4):  # granulation, eschar, suppuration
            prediction = (labels == channel) & valid
            truth = targets == channel
            intersection[channel] += (prediction & truth).sum()
            denominator[channel] += prediction.sum() + truth.sum()
    dice = (2 * intersection / denominator.clamp_min(1)).tolist()
    return float(np.mean([dice[1], dice[2], dice[4]]))


def main() -> int:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    OUTPUT = args.output
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError("Choose a new --output directory; existing results are retained")
    import segmentation_models_pytorch as smp

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    names = split_names()
    train_data = ExclusiveDataset(names["train"], augment=True, allow_missing=True)
    val_data = ExclusiveDataset(names["val"], augment=False)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4", encoder_weights="imagenet",
        in_channels=3, classes=5,
    ).to(device)
    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": 1e-4},
            {"params": decoder_parameters, "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    history, best, bad_epochs = [], float("-inf"), 0
    for epoch in range(1, args.epochs + 1):
        trainable = epoch > 3
        for parameter in model.encoder.parameters():
            parameter.requires_grad = trainable
        model.train()
        if not trainable:
            model.encoder.eval()
        total = 0.0
        for images, targets in tqdm(train_loader, desc="single-label train", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = dice_focal_loss(model(images), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)
        score = validation_macro(model, val_loader, device)
        scheduler.step()
        improved = score > best + 1e-4
        if improved:
            best, bad_epochs = score, 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "architecture": "UnetPlusPlus",
                    "encoder_name": "efficientnet-b4",
                    "image_size": SIZE,
                    "data_split": names,
                    "training_images_available": len(train_data),
                    "training_images_missing_single_label_mask": train_data.missing_names,
                    "selection_metric": "exclusive_three-tissue_macro_dice",
                    "selection_score": score,
                },
                OUTPUT / "best.pt",
            )
        else:
            bad_epochs += 1
        history.append({
            "epoch": epoch,
            "train_loss": total / len(train_data),
            "validation_macro_dice": score,
            "best_score": best,
            "epochs_without_improvement": bad_epochs,
        })
        save_json(OUTPUT / "history.json", history)
        print(f"epoch={epoch} validation_macro_dice={score:.6f} best={best:.6f} bad_epochs={bad_epochs}")
        if bad_epochs >= 18:
            print(f"early stopping at epoch={epoch}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
