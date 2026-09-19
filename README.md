# WoundQuant

WoundQuant segments wound tissue and measures tissue area from photographs. It combines overlap-aware UNet++ segmentation with RulerNet scale estimation and provides a Gradio interface, a command-line interface, and research experiment workflows.

At pixels assigned to multiple tissue classes, each active class receives an equal share of the pixel area. Physical area is computed as effective pixel area divided by the square of the image scale in pixels per centimetre.

## Installation

Use [Pixi](https://pixi.sh) to create the project environment:

```sh
pixi install
```

The environment uses Python 3.11 and PyTorch 2.7.1 with CUDA 12.8. Inference also supports CPU execution. To install into an existing compatible Python environment:

```sh
python -m pip install -e "."
```

Install the experiment dependencies with `python -m pip install -e ".[exp]"`. The Pixi environment includes these dependencies.

## Model weights

Model weights are stored separately from the source code. Supply compatible WoundQuant checkpoints or train them using the experiment workflow below.

| Path | Purpose |
|---|---|
| `weights/woundquant.pt` | WoundQuant checkpoint for the application and CLI |
| `weights/rulernet.onnx` | Five-output RulerNet ONNX model for scale estimation |
| `weights/experiments/<model>/best.pt` | Full experiment checkpoint, including class mappings and training metadata |

RulerNet [source code](https://github.com/ymp5078/RulerNet) and [pretrained weights](https://huggingface.co/ymp5078/RulerNet/tree/main/weights) are available upstream. Download the upstream weights with:

```sh
pixi run woundquant download-rulernet
```

Training checkpoints require ONNX export using the upstream `onnx_standalone.py` workflow before use as `rulernet.onnx`. Experiment evaluation requires full checkpoints; an inference-only checkpoint does not contain all required metadata.

## Usage

Start the Gradio application after placing the weights at the paths above:

```sh
pixi run app
```

Open `http://127.0.0.1:7860`, upload an image, and run the analysis. RulerNet estimates the image scale automatically; a manual pixels-per-centimetre value can also be supplied. If no valid scale is available, the application reports effective pixel areas without physical-area estimates.

For command-line inference:

```sh
woundquant predict image.png --checkpoint weights/woundquant.pt --ruler-model weights/rulernet.onnx --output results/example
woundquant predict image.png --checkpoint weights/woundquant.pt --pixels-per-cm 120 --output results/manual
```

With Pixi, prefix `woundquant` commands with `pixi run`. Outputs include area measurements in CSV and JSON, discrete combination masks, tissue memberships, and an overlay preview.

## Data

`dataset/` contains annotations, ruler measurements, exclusive-label masks, and fixed split IDs. Photographs are not distributed with the repository.

Set the `data` paths in each experiment's `config.json`. The main dataset uses the `generalHospital` annotations. Exclusive-label baselines use their configured image directory and its `mask/` subdirectory. Image paths can also be supplied through `WOUNDQUANT_IMAGES` and `WOUNDQUANT_EXCLUSIVE`, or through the Git-ignored `.local/paths.json`. Environment variables take precedence over local settings and experiment configuration.

## Experiments

Experiments are grouped by their corresponding paper results. Each directory contains an entry point, a JSON configuration, and analysis or training code. Shared implementations live in `exp/_common/`. The workflows export numerical results without manuscript plotting code.

| Directory | Workflow | Command |
|---|---|---|
| `exp/fig1_agreement/` | Inter-rater agreement, ICC, and mixed-effects analysis | `python exp/fig1_agreement/run.py all` |
| `exp/fig2_workflow/` | Segmentation, scale estimation, and area measurement for an example image | `python exp/fig2_workflow/run.py all` |
| `exp/fig4_performance/` | Test-set segmentation and physical-area evaluation | `python exp/fig4_performance/run.py all` |
| `exp/fig5_cases/` | Representative case selection from performance results | `python exp/fig5_cases/run.py all` |
| `exp/table1_comparison/` | Model comparison and loss ablation | `python exp/table1_comparison/run.py all` |

Run commands from the repository root; prefix `python` with `pixi run` when using Pixi. Run the performance workflow before case selection. Agreement analysis uses annotation metadata and image dimensions without requiring photographs.

Configuration paths are relative to the repository root. Use `--config path/to/config.json` to select an alternative configuration. Configure inputs under `data`, weights under `checkpoint` or `models`, and generated artifacts under `output_dir`. Case selection reads the performance configuration specified in `dependencies`.

The example and performance workflows support separate `prepare` and `analyze` steps. Model comparison supports explicit training and evaluation:

```sh
python exp/table1_comparison/run.py train --models ours without_dice
python exp/table1_comparison/run.py evaluate --models ours
python exp/table1_comparison/run.py summarize
```

`all` evaluates configured models and assembles comparison tables; it does not train models. Training recipes are defined under `training` and `models.<model>.training`. After training, update `models.<model>.checkpoint` to the new checkpoint. Set `training_history` to include completed training epochs in the comparison table; this field is left empty when the history is unavailable.

Generated predictions, metrics, and training logs are written to `results/` and excluded from Git. The repository contains no precomputed experiment results.

## License

WoundQuant's original code is licensed under the [MIT License](LICENSE). The RulerNet-derived adapter is subject to CC BY-NC 4.0; see [third-party licenses](THIRD_PARTY_LICENSES). Software licenses do not grant rights to research photographs or datasets.
