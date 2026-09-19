"""Measurement, visualization and portable result export."""
import csv
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from . import __version__
from .inference import Segmenter
from .ruler import RulerNet

COLORS = {"normal": (0, 0, 0), "eschar": (235, 235, 235), "granulation": (0, 170, 255), "suppuration": (255, 220, 0)}

class WoundQuant:
    def __init__(self, checkpoint, ruler_model=None, device="auto"):
        self.segmenter = Segmenter(checkpoint, device)
        self.ruler = RulerNet(ruler_model) if ruler_model else None

    def analyze(self, image, *, pixels_per_cm=None, alpha=.45):
        if not 0 <= alpha <= 1:
            raise ValueError("Overlay opacity must be between zero and one.")
        if pixels_per_cm is not None and (not math.isfinite(pixels_per_cm) or pixels_per_cm <= 0):
            raise ValueError("Manual scale must be positive pixels/cm.")
        image, probability, codes = self.segmenter.predict(image)
        points, status = [], "not_configured"
        scale_source = "manual" if pixels_per_cm is not None else "unavailable"
        if pixels_per_cm is None and self.ruler:
            pixels_per_cm, points, status = self.ruler.predict(image)
            scale_source = "rulernet" if pixels_per_cm else "unavailable"
        elif pixels_per_cm is not None:
            status = "manual"
        colors = np.array([COLORS.get(name.casefold(), (180, 100, 180)) for name in self.segmenter.names], dtype=np.float32)
        foreground = probability[1:].sum(0)
        color = np.einsum("chw,ck->hwk", probability, colors)
        strength = (alpha * foreground)[..., None]
        # Divide by foreground mass to keep the displayed tissue colour stable.
        color = color / np.maximum(foreground[..., None], 1e-8)
        overlay = Image.fromarray(np.clip(np.asarray(image) * (1-strength) + color * strength, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(overlay)
        radius = max(2, round(min(image.size) / 250))
        for x, y in points:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="red")
        areas = []
        for c, name in enumerate(self.segmenter.names):
            mass = float(probability[c].sum(dtype=np.float64))
            areas.append({"class": name, "effective_pixels": mass, "area_cm2": mass / pixels_per_cm**2 if pixels_per_cm else None})
        metadata = {"version": __version__, "checkpoint": self.segmenter.checkpoint_name,
                    "width": image.width, "height": image.height, "pixels_per_cm": pixels_per_cm,
                    "scale_source": scale_source, "ruler_status": status, "ruler_points": points,
                    "class_names": self.segmenter.names, "areas": areas,
                    "representation": "bit c = semantic channel c; active channels share 1/k pixel mass"}
        return overlay, probability, codes, metadata

def export_result(result, directory):
    overlay, probability, codes, metadata = result
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    overlay.save(path / "overlay.png")
    Image.fromarray(codes).save(path / "combination_codes.png")
    np.savez_compressed(path / "memberships.npz", probabilities=probability, class_names=metadata["class_names"])
    (path / "result.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    with (path / "areas.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class", "effective_pixels", "area_cm2"])
        writer.writeheader()
        writer.writerows(metadata["areas"])
    return [str(path / name) for name in ["areas.csv", "result.json", "combination_codes.png", "memberships.npz", "overlay.png"]]

