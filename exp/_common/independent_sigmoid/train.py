"""Train the four-channel independent-sigmoid baseline on the fixed split."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from independent_sigmoid.loss import (  # noqa: E402
    LOSS_COMPONENTS,
    IndependentSigmoidBCEDiceLoss,
)
from independent_sigmoid.model import REPRESENTATION, build_model  # noqa: E402
from seg_discrete_overlap.train_eval import (  # noqa: E402
    CachedCocoSoftSegmentationDataset,
    resolve_device,
)


ARCHITECTURE = "unetplusplus_hrnet_w32_independent_sigmoid_v1"
DEFAULT_ANNOTATIONS = Path("dataset/generalHospital/label/annotations.json")
DEFAULT_IMAGES = Path(os.environ.get("WOUNDQUANT_IMAGES", "dataset/generalHospital"))
DEFAULT_SPLIT = Path("dataset/splits.json")
DEFAULT_OUTPUT = Path("results/table1_comparison/training/independent_sigmoid")
DEFAULT_CACHE = Path("results/table1_comparison/cache")


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def serializable_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def fixed_split(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))["splits"]
    return {
        name: [int(value) for value in payload[name]["image_ids"]]
        for name in ("train", "val", "test")
    }


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def loader_options(args: argparse.Namespace, device: torch.device) -> dict:
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and device.type == "cuda",
    }
    if args.num_workers:
        options.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
        )
    return options


@torch.no_grad()
def estimate_pos_weight(dataset, num_classes: int, cap: float) -> torch.Tensor:
    positive = torch.zeros(num_classes, dtype=torch.float64)
    pixel_count = 0
    random_state = random.getstate()
    try:
        for index in tqdm(range(len(dataset)), desc="estimating pos_weight", leave=False):
            target = dataset[index]["target"]
            membership = target > 0
            positive += membership.sum(dim=(1, 2), dtype=torch.float64)
            pixel_count += int(target.shape[1] * target.shape[2])
    finally:
        random.setstate(random_state)
    if torch.any(positive == 0):
        raise RuntimeError("At least one channel has no positive training pixels")
    negative = float(pixel_count) - positive
    return (negative / positive).clamp(max=cap).float()


def train_epoch(model, loader, optimizer, criterion, device, encoder_frozen: bool) -> dict:
    model.train()
    if encoder_frozen:
        model.network.encoder.eval()
    totals = {name: 0.0 for name in LOSS_COMPONENTS}
    samples = 0
    for batch in tqdm(loader, desc="independent sigmoid train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        components = criterion.components(model(image), target)
        components["total"].backward()
        optimizer.step()
        size = len(image)
        samples += size
        for name in totals:
            totals[name] += float(components[name].detach()) * size
    return {name: value / samples for name, value in totals.items()}


@torch.no_grad()
def validate_epoch(model, loader, criterion, device) -> tuple[dict, dict]:
    model.eval()
    totals = {name: 0.0 for name in LOSS_COMPONENTS}
    intersection = None
    denominator = None
    samples = 0
    for batch in tqdm(loader, desc="independent sigmoid validation", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        logits = model(image)
        components = criterion.components(logits, target)
        prediction = model.hard_membership(logits)
        membership = target > 0
        batch_intersection = (prediction & membership).sum(dim=(0, 2, 3)).double().cpu()
        batch_denominator = (
            prediction.sum(dim=(0, 2, 3)) + membership.sum(dim=(0, 2, 3))
        ).double().cpu()
        intersection = (
            batch_intersection if intersection is None else intersection + batch_intersection
        )
        denominator = (
            batch_denominator if denominator is None else denominator + batch_denominator
        )
        size = len(image)
        samples += size
        for name in totals:
            totals[name] += float(components[name].detach()) * size
    loss = {name: value / samples for name, value in totals.items()}
    dice = (2.0 * intersection / denominator.clamp_min(1.0)).tolist()
    metrics = {
        "membership_dice_per_class": dice,
        "membership_dice_foreground_macro": float(np.mean(dice[1:])),
    }
    return loss, metrics


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    criterion,
    epoch: int,
    args: argparse.Namespace,
    mapping,
    split: dict,
    record: dict,
    best_score: float,
    bad_epochs: int,
) -> None:
    payload = {
        "format_version": 1,
        "architecture": ARCHITECTURE,
        "representation": REPRESENTATION,
        "stage": "segmenter",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "mapping": mapping.to_dict(),
        "loss": criterion.metadata(mapping.channel_to_name),
        "encoder_name": args.encoder,
        "threshold": args.threshold,
        "args": serializable_args(args),
        "data_split": split,
        "record": record,
        "training_state": {
            "best_score": best_score,
            "epochs_without_improvement": bad_epochs,
            "selection_metric": "membership_dice_foreground_macro",
            "selection_mode": "max",
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("architecture") != ARCHITECTURE
        or payload.get("representation") != REPRESENTATION
    ):
        raise RuntimeError(f"Not an independent-sigmoid checkpoint: {path}")
    return payload


def load_inference_model(path: Path, device: torch.device):
    payload = load_checkpoint(path, device)
    model = build_model(
        int(payload["mapping"]["num_classes"]),
        encoder_name=str(payload["encoder_name"]),
        encoder_weights=None,
        threshold=float(payload.get("threshold", 0.5)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def predict_image(model, image: np.ndarray, device: torch.device, image_size: int):
    resized = np.asarray(
        Image.fromarray(image).resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )
    array = np.moveaxis(resized.astype(np.float32) / 255.0, -1, 0)
    mean = np.asarray((0.485, 0.456, 0.406), np.float32)[:, None, None]
    std = np.asarray((0.229, 0.224, 0.225), np.float32)[:, None, None]
    tensor = torch.from_numpy(np.ascontiguousarray((array - mean) / std))
    logits = model(tensor.unsqueeze(0).to(device))
    logits = F.interpolate(
        logits,
        size=image.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    return model.hard_uniform_probabilities(logits)[0].cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--normal-name", default="normal")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--encoder", default="tu-hrnet_w32")
    parser.add_argument(
        "--encoder-weights", choices=("imagenet", "none"), default="imagenet"
    )
    parser.add_argument("--encoder-learning-rate", type=float, default=7.5e-5)
    parser.add_argument("--decoder-learning-rate", type=float, default=7.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=18)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--scheduler-t0", type=int, default=20)
    parser.add_argument("--scheduler-t-mult", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=8)
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.image_size) <= 0:
        raise ValueError("epochs, batch size and image size must be positive")
    if min(args.bce_weight, args.dice_weight) < 0 or args.bce_weight + args.dice_weight <= 0:
        raise ValueError("loss weights must be non-negative and not both zero")
    if args.pos_weight_cap <= 0 or not 0 < args.threshold < 1:
        raise ValueError("pos-weight cap must be positive and threshold must be in (0, 1)")

    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    ids = fixed_split(args.split)
    split = {
        "source": str(args.split.resolve()),
        "splits": {
            name: {
                "annotations": str(args.annotations.resolve()),
                "images": str(args.images.resolve()),
                "image_ids": image_ids,
            }
            for name, image_ids in ids.items()
        },
    }
    train_data = CachedCocoSoftSegmentationDataset(
        args.annotations,
        args.images,
        image_ids=ids["train"],
        normal_name=args.normal_name,
        cache_dir=args.cache_dir / "train",
        image_size=args.image_size,
        augment=True,
    )
    val_data = CachedCocoSoftSegmentationDataset(
        args.annotations,
        args.images,
        image_ids=ids["val"],
        normal_name=args.normal_name,
        cache_dir=args.cache_dir / "val",
        image_size=args.image_size,
        augment=False,
    )
    if train_data.mapping.to_dict() != val_data.mapping.to_dict():
        raise RuntimeError("Training and validation channel mappings differ")
    mapping = train_data.mapping
    if mapping.num_classes != 4:
        raise RuntimeError(f"Expected four semantic channels, found {mapping.num_classes}")
    pos_weight = estimate_pos_weight(train_data, mapping.num_classes, args.pos_weight_cap)
    criterion = IndependentSigmoidBCEDiceLoss(
        pos_weight,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
    ).to(device)
    model = build_model(
        mapping.num_classes,
        encoder_name=args.encoder,
        encoder_weights=None if args.encoder_weights == "none" else "imagenet",
        threshold=args.threshold,
    ).to(device)
    encoder_parameters = list(model.network.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_learning_rate, "name": "encoder"},
            {"params": decoder_parameters, "lr": args.decoder_learning_rate, "name": "decoder"},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.scheduler_t0,
        T_mult=args.scheduler_t_mult,
        eta_min=args.min_learning_rate,
    )
    loader_args = loader_options(args, device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)

    segmenter_output = args.output / "segmenter"
    if segmenter_output.exists() and not args.resume and any(segmenter_output.iterdir()):
        raise FileExistsError(
            f"Output is not empty: {segmenter_output}; use --resume or another path"
        )
    segmenter_output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "config.json", serializable_args(args))
    save_json(args.output / "channel_mapping.json", mapping.to_dict())
    save_json(args.output / "data_split.json", split)
    save_json(args.output / "loss_config.json", criterion.metadata(mapping.channel_to_name))

    history_path = segmenter_output / "history.json"
    last_path = segmenter_output / "last.pt"
    history = []
    start_epoch, best_score, bad_epochs = 1, float("-inf"), 0
    if args.resume and last_path.is_file():
        payload = load_checkpoint(last_path, device)
        if payload["mapping"] != mapping.to_dict() or payload["data_split"] != split:
            raise RuntimeError("Resume checkpoint mapping or split differs")
        if payload["loss"] != criterion.metadata(mapping.channel_to_name):
            raise RuntimeError("Resume checkpoint loss configuration differs")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        history = json.loads(history_path.read_text(encoding="utf-8"))
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload["training_state"]["best_score"])
        bad_epochs = int(payload["training_state"]["epochs_without_improvement"])

    print(
        f"device={device}; representation={REPRESENTATION}; train={len(train_data)}; "
        f"val={len(val_data)}; test={len(ids['test'])}; pos_weight={pos_weight.tolist()}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        encoder_frozen = epoch <= args.freeze_encoder_epochs
        for parameter in model.network.encoder.parameters():
            parameter.requires_grad = not encoder_frozen
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, encoder_frozen
        )
        validation_loss, validation_metrics = validate_epoch(
            model, val_loader, criterion, device
        )
        score = float(validation_metrics["membership_dice_foreground_macro"])
        improved = score > best_score + args.early_stopping_min_delta
        if improved:
            best_score, bad_epochs = score, 0
        else:
            bad_epochs += 1
        learning_rates = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
            "selection_score": score,
            "encoder_frozen": encoder_frozen,
            "learning_rates": learning_rates,
        }
        history.append(record)
        save_json(history_path, history)
        checkpoint_args = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epoch=epoch,
            args=args,
            mapping=mapping,
            split=split,
            record=record,
            best_score=best_score,
            bad_epochs=bad_epochs,
        )
        save_checkpoint(last_path, **checkpoint_args)
        if improved:
            save_checkpoint(segmenter_output / "best.pt", **checkpoint_args)
        print(
            f"epoch={epoch} foreground_membership_dice={score:.6f} "
            f"best={best_score:.6f} bad_epochs={bad_epochs}"
        )
        if args.early_stopping_patience and bad_epochs >= args.early_stopping_patience:
            print(f"early stopping at epoch={epoch}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
