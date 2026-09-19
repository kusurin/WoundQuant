"""Gradio interface; models are initialized lazily on first analysis."""
import tempfile
from pathlib import Path
from threading import Lock
import gradio as gr
from .pipeline import WoundQuant, export_result

def create_app(checkpoint, ruler_model=None, device="auto"):
    engine = None
    lock = Lock()

    def process(image, manual_scale, alpha):
        nonlocal engine
        if image is None:
            raise gr.Error("请先上传图像。")
        try:
            with lock:
                if engine is None:
                    engine = WoundQuant(checkpoint, ruler_model, device)
                result = engine.analyze(image, pixels_per_cm=float(manual_scale) if manual_scale else None, alpha=alpha)
                # Export within Gradio's managed cache so delete_cache can expire it.
                directory = tempfile.mkdtemp(prefix="woundquant-", dir=demo.GRADIO_CACHE)
                files = export_result(result, directory)
                demo.temp_files.update(files)
            meta = result[3]
            rows = [[row["class"], row["effective_pixels"], row["area_cm2"]] for row in meta["areas"]]
            scale = meta["pixels_per_cm"]
            message = f"比例尺：{scale:.3f} pixels/cm（{meta['scale_source']}）" if scale else f"未获得有效比例尺（{meta['ruler_status']}），仅报告有效像素面积。"
            return result[0], rows, message, files
        except (ValueError, RuntimeError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="WoundQuant", delete_cache=(3600, 86400)) as demo:
        gr.Markdown("# WoundQuant\n创面组织分割 · 标尺识别 · 面积测量")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="创面图像")
                scale = gr.Number(value=0, label="手动比例尺（pixels/cm；0 表示自动检测）", minimum=0)
                alpha = gr.Slider(0, 1, value=.45, label="叠加透明度")
                button = gr.Button("分析图像", variant="primary")
            with gr.Column():
                overlay = gr.Image(type="pil", label="组织分割与标尺刻度")
                table = gr.Dataframe(headers=["组织类别", "有效像素面积", "面积（cm²）"], interactive=False)
                status = gr.Textbox(label="比例尺状态", interactive=False)
                files = gr.File(label="下载结果", file_count="multiple")
        gr.Markdown("蓝色：肉芽组织；白色：痂皮；黄色：脓苔；红点：标尺刻度。重叠区域按组织数均分像素面积。用于研究。")
        button.click(process, [image, scale, alpha], [overlay, table, status, files], concurrency_limit=1, api_name=False)
    Path(demo.GRADIO_CACHE).mkdir(parents=True, exist_ok=True)
    return demo
