"""RulerNet five-output ONNX adapter, matching the paper's scale calculation.

Geometric-progression reconstruction adapted from ymp5078/RulerNet.
SPDX-License-Identifier: CC-BY-NC-4.0. See THIRD_PARTY_LICENSES.
"""
from pathlib import Path
import numpy as np
from PIL import Image

class RulerNet:
    def __init__(self, model: str | Path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = ["init_point", "dist", "ratio", "direction", "points_info"]
        names = {out.name for out in self.session.get_outputs()}
        if not set(self.output_names) <= names:
            raise ValueError("Use the five-output RulerNet ONNX export, not a training checkpoint.")

    def predict(self, image: Image.Image):
        width, height = image.size
        scale = min(768 / width, 768 / height)
        w, h = max(1, int(width * scale)), max(1, int(height * scale))
        left, top = (768 - w) // 2, (768 - h) // 2
        canvas = np.zeros((768, 768, 3), dtype=np.float32)
        canvas[top:top+h, left:left+w] = np.asarray(image.convert("RGB").resize((w, h)), dtype=np.float32) / 255
        outputs = self.session.run(self.output_names, {self.input_name: canvas.transpose(2, 0, 1)[None].copy()})
        initial, distance, ratio, direction, info = [np.asarray(x[0]).reshape(-1) for x in outputs]
        if not all(np.isfinite(x).all() for x in (initial, distance, ratio, direction, info)):
            return None, [], "invalid_output"
        count = int(info[0])
        if count < 1 or count > 10000 or distance[0] <= 0 or ratio[0] <= 0:
            return None, [], "insufficient_points"
        n = np.arange(-count, count + 1)
        with np.errstate(over="ignore", invalid="ignore"):
            steps = ratio[0] ** n * distance[0]
            zero = np.zeros((1, 2))
            a = np.cumsum(np.vstack([zero, -steps[n < 0][::-1, None] * direction]), axis=0)
            b = np.cumsum(np.vstack([zero, steps[n >= 0, None] * direction]), axis=0)
        points = np.vstack([initial + a[::-1], (initial + b)[1:]])
        xmin, ymin, xmax, ymax = info[1:]
        inside = np.isfinite(points).all(1) & (points[:, 0] >= xmin) & (points[:, 0] <= xmax) & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
        points = points[inside]
        if len(points) < 2:
            return None, [], "insufficient_points"
        pixels_per_cm = float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)) / scale)
        points = (points - [left, top]) / scale
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        if not np.isfinite(pixels_per_cm) or pixels_per_cm <= 0:
            return None, points.tolist(), "invalid_scale"
        return pixels_per_cm, points.tolist(), "valid"
