"""BCEWithLogits plus channel-macro soft Dice for multi-hot masks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


LOSS_NAME = "pos_weighted_bce_with_logits_plus_soft_dice_v1"
LOSS_COMPONENTS = ("total", "bce_with_logits", "soft_dice_loss")


class IndependentSigmoidBCEDiceLoss(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor | Sequence[float],
        *,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(pos_weight, dtype=torch.float32)
        if weights.ndim != 1 or len(weights) < 2:
            raise ValueError("pos_weight must contain one value per semantic channel")
        if not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("pos_weight values must be finite and positive")
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("loss weights must be non-negative and not both zero")
        self.register_buffer("pos_weight", weights)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.epsilon = float(epsilon)

    @staticmethod
    def multihot_target(target: torch.Tensor) -> torch.Tensor:
        """Convert the repository's uniform-overlap target to membership labels."""

        return (target > 0).to(target.dtype)

    def components(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits.shape != target.shape or logits.ndim != 4:
            raise ValueError("logits and target must have the same BxCxHxW shape")
        membership = self.multihot_target(target)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            membership,
            pos_weight=self.pos_weight.view(1, -1, 1, 1),
        )
        probability = torch.sigmoid(logits)
        intersection = (probability * membership).sum(dim=(0, 2, 3))
        denominator = probability.sum(dim=(0, 2, 3)) + membership.sum(dim=(0, 2, 3))
        present = denominator > self.epsilon
        dice = (2.0 * intersection + self.epsilon) / (
            denominator + self.epsilon
        )
        dice_loss = 1.0 - dice[present].mean()
        total = self.bce_weight * bce + self.dice_weight * dice_loss
        return {
            "total": total,
            "bce_with_logits": bce,
            "soft_dice_loss": dice_loss,
        }

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.components(logits, target)["total"]

    def metadata(self, class_names: Sequence[str]) -> dict:
        return {
            "name": LOSS_NAME,
            "formula": f"{self.bce_weight}*BCEWithLogits + {self.dice_weight}*soft_dice",
            "target": "multi-hot membership (uniform soft target > 0)",
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
            "pos_weight": {
                str(name): float(weight)
                for name, weight in zip(class_names, self.pos_weight.tolist())
            },
            "dice_channels": list(class_names),
        }
