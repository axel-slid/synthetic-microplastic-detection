# Microplastic Identification Using AI-Driven Image Segmentation [Replication code]
with Synthetic Ecological Context

![](images/workflow_complex_variation_001.png)

### 3x3 Image-Varied Inference Grid

![3x3 image-varied inference grid 001](images/inference_grid_3x3_image_varied_001.png)

## Repository Layout

- `scripts/`: dataset download, metadata filtering, visualization, and figure helpers.
- `docs/`: dataset notes and generated documentation assets.
- `benchmark/microplastic_benchmark/`: reusable benchmark package for data loading,
  generation, model construction, training, metrics, and evaluation.
- `benchmark/configs/`: benchmark experiment matrices and model settings.
- `benchmark/scripts/`: benchmark orchestration, training, evaluation, and reporting scripts.
- `code/`: earlier stable diffusion, GAN, preprocessing, and segmentation scripts.
- `tests/`: smoke tests for the Stable Diffusion training pipeline.
- `images/` and `overleaf_microplastic_project/`: paper figures and manuscript source.
- `model_settings.txt`: concise settings log for Stable Diffusion and the six semantic
  segmentation models used in the study.

## Reproduce The Study

1. Clone this repository and install the Python dependencies:

   ```bash
   git clone https://github.com/axel-slid/synthetic-microplastic-detection.git
   cd synthetic-microplastic-detection
   python -m pip install -r requirements.txt
   python -m pip install -r benchmark/requirements.txt
   python -m pip install -e benchmark
   ```

2. Download the project data from Harvard Dataverse:

   <https://dataverse.harvard.edu/dataverse/Microplastic-Segmentation-GAN/>

   Place the extracted benchmark data under `benchmark/data/` so the config paths
   resolve as `benchmark/data/c1`, `benchmark/data/c2`, and `benchmark/data/c3`.

3. Prepare and validate the benchmark manifests:

   ```bash
   cd benchmark
   python scripts/validate_data.py --config configs/benchmark.yaml
   python scripts/prepare_manifests.py --config configs/benchmark.yaml
   python scripts/plan_runs.py --config configs/benchmark.yaml --available-only
   ```

4. Run the full semantic replication workflow:

   ```bash
   ./scripts/reproduce_semantic_study.sh
   ```

   This validates data, prepares manifests, plans available runs, trains the
   semantic models, evaluates checkpoints, aggregates metrics, and regenerates
   paper assets.

5. Or train the semantic segmentation models manually for the planned runs:

   ```bash
   python scripts/launch_benchmark.py \
     --config configs/benchmark.yaml \
     --run-matrix results/manifests/run_matrix.csv \
     --family semantic
   ```

   To run one checkpoint at a time:

   ```bash
   python scripts/train_segmenter.py \
     --config configs/benchmark.yaml \
     --run-id baseline_c1__monai_unet__seed13
   ```

6. Evaluate completed checkpoints on the locked C3-clean test set:

   ```bash
   python scripts/evaluate_registry.py --config configs/benchmark.yaml
   python scripts/aggregate_results.py
   ```

7. Regenerate paper assets and reports:

   ```bash
   python scripts/make_paper_assets.py
   python scripts/make_journal_paper_assets.py
   python scripts/verify_paper.py
   ```

The exact model settings used for the study are summarized in
[`model_settings.txt`](model_settings.txt). GPU training workflows require
CUDA-compatible PyTorch plus the model/data assets referenced by the configs.

## Dataset Utilities

The top-level `scripts/` directory also includes utilities for the public
Microplastic Image Explorer metadata workflow.

Download metadata only:

```bash
python scripts/download_image_explorer.py \
  --output-dir data/microplastic_image_explorer
```

Download metadata plus images:

```bash
python scripts/download_image_explorer.py \
  --output-dir data/microplastic_image_explorer \
  --download-images \
  --workers 24
```

## Data And Artifact Policy

The following are ignored because they are large or machine-specific:

- `data/`, `benchmark/data/`
- `generated/`
- `benchmark/results/`, `benchmark/runs/`
- checkpoints and weights such as `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`
- split datasets, generated samples, logs, caches, Office exports, and compiled PDFs

Use the scripts and configs in this repo to recreate datasets, runs, and figures
in a local workspace.
