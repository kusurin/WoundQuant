"""Shared FCN-ResNet50 model and losses for the two Table 1 baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50

from seg_discrete_overlap.combinations import CombinationCodec


MODES = ("single_label", "lookup_multilabel")


class Table1FCN(nn.Module):
    """Torchvision FCN-ResNet50 with either exclusive or combination logits."""

    def __init__(self, mode: str, *, pretrained: bool = False) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unsupported FCN mode: {mode}")
        self.mode = mode
        self.codec = CombinationCodec(4) if mode == "lookup_multilabel" else None
        output_classes = 5 if mode == "single_label" else self.codec.num_states
        weights = FCN_ResNet50_Weights.DEFAULT if pretrained else None
        self.network = fcn_resnet50(
            weights=weights,
            weights_backbone=None,
            aux_loss=True,
        )
        self.network.classifier[4] = nn.Conv2d(512, output_classes, kernel_size=1)
        self.network.aux_classifier[4] = nn.Conv2d(256, output_classes, kernel_size=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.network(image)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        if self.mode == "lookup_multilabel":
            return self.codec.hard_probabilities(logits)
        labels = logits.argmax(dim=1)
        prediction = torch.zeros(
            (len(labels), 4, *labels.shape[1:]),
            dtype=torch.float32,
            device=labels.device,
        )
        # Exclusive source classes: slight, granulation, eschar, deep, suppuration.
        prediction[:, 0] = ((labels == 0) | (labels == 3)).float()
        prediction[:, 1] = (labels == 2).float()
        prediction[:, 2] = (labels == 1).float()
        prediction[:, 3] = (labels == 4).float()
        return prediction


def exclusive_dice_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match the Dice+Focal objective used by the single-label UNet++ baseline."""

    valid = target != 255
    safe_target = target.masked_fill(~valid, 0)
    probabilities = logits.softmax(1)
    one_hot = F.one_hot(safe_target, 5).permute(0, 3, 1, 2).float()
    valid_channel = valid.unsqueeze(1)
    intersection = (probabilities * one_hot * valid_channel).sum((0, 2, 3))
    denominator = ((probabilities + one_hot) * valid_channel).sum((0, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    cross_entropy = F.cross_entropy(
        logits, target, ignore_index=255, reduction="none"
    )
    true_probability = probabilities.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    focal = (((1.0 - true_probability) ** 2.0) * cross_entropy)[valid].mean()
    return dice + focal


def load_table1_fcn(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    mode = str(payload.get("mode"))
    model = Table1FCN(mode, pretrained=False).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def predict_table1_fcn(
    model: Table1FCN,
    image: np.ndarray,
    device: torch.device,
    image_size: int,
) -> np.ndarray:
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
    logits = model(tensor.unsqueeze(0).to(device))["out"]
    logits = F.interpolate(
        logits, size=image.shape[:2], mode="bilinear", align_corners=False
    )
    return model.decode(logits)[0].cpu().numpy()
