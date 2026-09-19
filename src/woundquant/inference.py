"""Checkpoint-compatible inference at the original image resolution."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from .combinations import CombinationCodec, REPRESENTATION, build_combination_model

ARCHITECTURE = "single_stage_discrete_overlap_v2_imbalance_aware"

class Segmenter:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("architecture") != ARCHITECTURE or payload.get("representation") != REPRESENTATION or payload.get("stage") != "segmenter":
            raise ValueError("Expected a WoundQuant discrete-overlap segmenter checkpoint.")
        mapping = payload["mapping"]
        self.names = [item["category_name"] for item in mapping["channels"]]
        if payload.get("combination_mapping") != CombinationCodec(mapping["num_classes"]).metadata(self.names):
            raise ValueError("Checkpoint class mapping is incompatible.")
        self.model = build_combination_model(mapping["num_classes"], encoder_name=payload["encoder_name"], encoder_weights=None)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.size = int(payload.get("args", {}).get("image_size", 512))
        self.checkpoint_name = Path(checkpoint).name

    @torch.inference_mode()
    def predict(self, image: Image.Image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        resized = np.asarray(image.resize((self.size, self.size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        normalized = (resized - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))[None].to(self.device)
        logits = self.model(tensor)
        logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
        states = logits.argmax(1)
        probabilities = self.model.codec.decode(states)[0].cpu().numpy()
        codes = (states[0] + 1).cpu().numpy().astype(np.uint8)
        return image, probabilities, codes

