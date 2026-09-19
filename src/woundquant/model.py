"""UNet++ model that returns logits and exposes a stable probability API."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class SoftUNetPlusPlus(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = None,
        encoder_checkpoint: str | Path | None = None,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include background and at least one foreground")
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is required to build UNet++"
            ) from exc

        self.num_classes = num_classes
        self.network = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
        if encoder_checkpoint is not None:
            checkpoint_path = Path(encoder_checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Encoder checkpoint does not exist: {checkpoint_path}"
                )
            state = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise ValueError(
                    f"Encoder checkpoint is not a state dict: {checkpoint_path}"
                )
            state = {
                key.removeprefix("module.").removeprefix("encoder."): value
                for key, value in state.items()
            }
            self.network.encoder.load_state_dict(state)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return unnormalized BxKxHxW logits."""

        return self.network(image)

    @torch.no_grad()
    def predict_proba(self, image: torch.Tensor) -> torch.Tensor:
        """Return channel-normalized probabilities without changing model mode."""

        return torch.softmax(self.forward(image), dim=1)


def build_model(
    num_classes: int,
    *,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
    encoder_checkpoint: str | Path | None = None,
    in_channels: int = 3,
) -> SoftUNetPlusPlus:
    return SoftUNetPlusPlus(
        num_classes,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        encoder_checkpoint=encoder_checkpoint,
        in_channels=in_channels,
    )
