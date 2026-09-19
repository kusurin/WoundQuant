"""Uniform-overlap Soft GT variant.

This variant reuses the COCO validation, mask decoding, transforms, and Dataset
implementation from the parent project.  Its only semantic change is target
generation: every active class at an overlap pixel receives equal probability.
Non-overlap pixels remain strictly one-hot.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_parent_dataset() -> ModuleType:
    parent_dataset = Path(__file__).resolve().parent.parent / "dataset.py"
    specification = importlib.util.spec_from_file_location(
        "_distance_weighted_dataset", parent_dataset
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load parent dataset module: {parent_dataset}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


_parent = _load_parent_dataset()

# Re-export the shared data validation and loading surface so this directory can
# be used as a drop-in training entry point.
for _name in dir(_parent):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_parent, _name)


def build_soft_target(
    class_masks: np.ndarray, *, epsilon: float = 1e-6
) -> np.ndarray:
    """Build a probability target with uniform class mass in overlaps.

    ``epsilon`` is retained for command/API compatibility with the original
    version, but is deliberately not used: this rule has no distance or edge
    weighting.  For a pixel with ``n`` active classes, each active channel is
    assigned exactly ``1 / n``.
    """

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

    target = masks.astype(np.float32)
    target /= active_count[None, :, :].astype(np.float32)

    sums = target.sum(axis=0, dtype=np.float64)
    if not np.all(np.isfinite(target)):
        raise RuntimeError("Soft target contains NaN or infinity")
    if np.any(target < 0) or np.any(target > 1):
        raise RuntimeError("Soft target is outside [0, 1]")
    if not np.allclose(sums, 1.0, rtol=0.0, atol=1e-6):
        max_error = float(np.max(np.abs(sums - 1.0)))
        raise RuntimeError(f"Soft target simplex violation; max error={max_error}")
    if np.any(target[~masks] != 0):
        raise RuntimeError("Inactive classes received non-zero probability")
    return target


# CocoSoftSegmentationDataset resolves build_soft_target from its defining
# module at runtime. Rebind that global so the inherited Dataset uses this
# uniform implementation for every loaded sample.
_parent.build_soft_target = build_soft_target


if __name__ == "__main__":
    raise SystemExit(_parent._preflight_cli())
