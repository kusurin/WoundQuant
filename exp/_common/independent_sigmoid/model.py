"""UNet++ with four independent semantic logits."""

from __future__ import annotations

import torch
from torch import nn

from model import SoftUNetPlusPlus


REPRESENTATION = "independent_sigmoid_multihot_v1"


class IndependentSigmoidUNetPlusPlus(SoftUNetPlusPlus):
    """Predict each semantic channel independently with a sigmoid.

    Channel overlaps are legal. ``hard_membership`` guarantees at least one
    active channel per pixel by selecting the largest probability only when no
    channel reaches the threshold.
    """

    def __init__(self, num_classes: int = 4, *, threshold: float = 0.5, **kwargs):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between zero and one")
        super().__init__(num_classes, **kwargs)
        self.threshold = float(threshold)

    @torch.no_grad()
    def predict_proba(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(image))

    @torch.no_grad()
    def hard_membership(self, logits: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        membership = probability >= self.threshold
        empty = ~membership.any(dim=1)
        if bool(empty.any()):
            best = probability.argmax(dim=1)
            fallback = torch.zeros_like(membership)
            fallback.scatter_(1, best.unsqueeze(1), True)
            membership = torch.where(empty.unsqueeze(1), fallback, membership)
        return membership

    @torch.no_grad()
    def hard_uniform_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        membership = self.hard_membership(logits).to(logits.dtype)
        return membership / membership.sum(dim=1, keepdim=True)


def build_model(
    num_classes: int = 4,
    *,
    encoder_name: str = "tu-hrnet_w32",
    encoder_weights: str | None = "imagenet",
    threshold: float = 0.5,
) -> IndependentSigmoidUNetPlusPlus:
    return IndependentSigmoidUNetPlusPlus(
        num_classes,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        threshold=threshold,
    )
