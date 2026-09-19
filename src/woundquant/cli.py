import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(prog="woundquant")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "predict"):
        p = commands.add_parser(name)
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--ruler-model", type=Path)
        p.add_argument("--device", default="auto")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=7860)
            p.add_argument("--share", action="store_true")
        else:
            p.add_argument("input", type=Path)
            p.add_argument("--output", type=Path, default=Path("results"))
            p.add_argument("--pixels-per-cm", type=float)
    p = commands.add_parser("download-rulernet")
    p.add_argument("--output", type=Path, default=Path("weights/upstream"))
    p.add_argument("--revision", default="main")
    args = parser.parse_args()
    if args.command == "download-rulernet":
        from huggingface_hub import snapshot_download
        path = snapshot_download("ymp5078/RulerNet", revision=args.revision, allow_patterns=["weights/*"], local_dir=args.output)
        print(f"Official RulerNet weights downloaded to {path}. See README.md for ONNX export.")
        return
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.ruler_model and not args.ruler_model.is_file():
        parser.error(f"RulerNet ONNX not found: {args.ruler_model}")
    if args.command == "serve":
        from .app import create_app
        create_app(args.checkpoint, args.ruler_model, args.device).queue().launch(server_name=args.host, server_port=args.port, share=args.share)
    else:
        from PIL import Image
        from .pipeline import WoundQuant, export_result
        engine = WoundQuant(args.checkpoint, args.ruler_model, args.device)
        with Image.open(args.input) as image:
            result = engine.analyze(image, pixels_per_cm=args.pixels_per_cm)
        for path in export_result(result, args.output):
            print(path)

