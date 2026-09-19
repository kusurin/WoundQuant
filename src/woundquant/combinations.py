"""Nonempty class sets, exact bit codes, and a discrete-output training loss."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import SoftUNetPlusPlus


REPRESENTATION = "nonempty_class_set_uniform_v1"


class CombinationCodec(nn.Module):
    """State index = bit code - 1; bit c denotes semantic channel c.

    Explicit normal/foreground overlaps are legal. Code 1 is normal only.
    Limit enumeration to eight semantic classes (255 states).
    """

    def __init__(self, num_classes: int):
        super().__init__()
        if not 2 <= num_classes <= 8:
            raise ValueError("Combination classification supports 2..8 semantic classes")
        self.num_classes = num_classes
        self.num_states = (1 << num_classes) - 1
        bits = 1 << torch.arange(num_classes, dtype=torch.long)
        codes = torch.arange(1, self.num_states + 1, dtype=torch.long)
        members = (codes[:, None] & bits[None, :]) != 0
        self.register_buffer("bits", bits)
        self.register_buffer("table", members.float() / members.sum(1, keepdim=True))

    def encode(self, target: torch.Tensor) -> torch.Tensor:
        """Validate uniform B,C,H,W targets and return B,H,W state indices."""
        if target.ndim != 4 or target.shape[1] != self.num_classes:
            raise ValueError("Expected B,C,H,W semantic targets")
        members = target > 0
        codes = (members.long() * self.bits[None, :, None, None]).sum(1)
        if not torch.isfinite(target).all() or (codes == 0).any():
            raise ValueError("Targets must be finite nonempty class sets")
        states = codes - 1
        if not torch.allclose(target, self.decode(states).to(target.dtype), atol=1e-6, rtol=0):
            raise ValueError("Targets must assign exactly 1/k to every active class")
        return states

    def decode(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError("Expected B,H,W state indices")
        if ((states < 0) | (states >= self.num_states)).any():
            raise ValueError("State indices are outside the nonempty combination table")
        return self.table[states].permute(0, 3, 1, 2).contiguous()

    def hard_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != self.num_states:
            raise ValueError(f"Expected B,{self.num_states},H,W combination logits")
        return self.decode(logits.argmax(1))

    def metadata(self, class_names) -> dict:
        if len(class_names) != self.num_classes:
            raise ValueError("Class names do not match the codec")
        return {
            "representation": REPRESENTATION,
            "num_semantic_classes": self.num_classes,
            "num_states": self.num_states,
            "state_index": "bit_code - 1",
            "bit_code": "sum(2**channel for each active semantic channel)",
            "normal_only_bit_code": 1,
            "states": [
                {"state_index": code - 1, "bit_code": code,
                 "classes": [name for c, name in enumerate(class_names) if code & (1 << c)],
                 "active_count": code.bit_count(),
                 "probabilities": self.table[code - 1].tolist()}
                for code in range(1, self.num_states + 1)
            ],
        }


class CombinationLoss(nn.Module):
    """CE(true set) + lambda * E_set[sum_c |q_set[c] - target[c]|].

    The cost table avoids B,S,C,H,W intermediates. This is expected error
    over discrete candidates, not error of an averaged probability map.
    """

    def __init__(self, num_classes: int, expected_l1_weight: float = 1.0):
        super().__init__()
        if not math.isfinite(expected_l1_weight) or expected_l1_weight < 0:
            raise ValueError("expected_l1_weight must be finite and non-negative")
        self.codec = CombinationCodec(num_classes)
        self.expected_l1_weight = expected_l1_weight
        table = self.codec.table
        self.register_buffer("costs", (table[:, None] - table[None, :]).abs().sum(-1))

    def components(self, logits: torch.Tensor, target: torch.Tensor) -> dict:
        if logits.shape != (target.shape[0], self.codec.num_states, *target.shape[2:]):
            raise ValueError("Logit shape does not match combination targets")
        states = self.codec.encode(target)
        # CUDA's fused 2-D NLL forward has no deterministic implementation in
        # the supported PyTorch build. This is mathematically the same
        # unweighted mean cross-entropy, expressed with deterministic tensor
        # operations so strict mode can remain enabled for the segmenter.
        log_probabilities = F.log_softmax(logits, dim=1)
        true_log_probabilities = log_probabilities.gather(
            1, states.unsqueeze(1),
        ).squeeze(1)
        ce = -true_log_probabilities.mean()
        costs = self.costs[:, states].permute(1, 0, 2, 3)
        expected_l1 = (log_probabilities.exp() * costs).sum(1).mean()
        return {"total": ce + self.expected_l1_weight * expected_l1,
                "combination_ce": ce, "expected_l1": expected_l1}

    def forward(self, logits, target):
        return self.components(logits, target)["total"]


class CombinationUNetPlusPlus(SoftUNetPlusPlus):
    def __init__(self, num_classes: int, **kwargs):
        codec = CombinationCodec(num_classes)
        super().__init__(codec.num_states, **kwargs)
        self.num_classes = num_classes  # Public probabilities still have C channels.
        self.num_states = codec.num_states
        self.codec = codec

    @torch.no_grad()
    def predict_proba(self, image):
        return self.codec.hard_probabilities(self(image))


def build_combination_model(num_classes: int, **kwargs):
    return CombinationUNetPlusPlus(num_classes, **kwargs)
