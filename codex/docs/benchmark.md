# Microplastic Benchmark Harness

This workspace now contains an executable benchmark harness for comparing synthetic generation methods against segmentation model families. The full configured matrix is 7 generation/training conditions x 7 segmenters x 3 seeds = 147 runs once the two new generated datasets exist.

## Current Data Policy

- Primary test set: `results/manifests/c3_clean.csv`.
- Excluded from primary C3 metrics:
  - `006.png`: image/mask size mismatch.
  - `026.png`: pixel-identical to `data/c2/imgs/0481.png`.
  - `088.png`: near-duplicate of `data/c2/imgs/0732.png`.
- `results/manifests/c3_all.csv` is retained for appendix-only reporting.

## Standard Workflow

```bash
python scripts/validate_data.py --config configs/benchmark.yaml
python scripts/prepare_manifests.py --config configs/benchmark.yaml
python scripts/plan_runs.py --config configs/benchmark.yaml --available-only
python scripts/smoke_test.py --config configs/benchmark.yaml
```

The current data supports 5 available generation/training conditions x 7 segmenters x 3 seeds = 105 runs. After the new generated datasets are created, rerun manifests and plan the full matrix without `--available-only`.

```bash
python scripts/generate_synthetic.py --method new_sdxl_inpaint --count 10000
python scripts/generate_synthetic.py --method new_flux_inpaint --count 10000
python scripts/prepare_manifests.py --config configs/benchmark.yaml --require-generated
python scripts/plan_runs.py --config configs/benchmark.yaml
```

Train a semantic run:

```bash
python scripts/train_segmenter.py \
  --config configs/benchmark.yaml \
  --run-id baseline_c1__monai_unet__seed13
```

Train a YOLO segmentation run:

```bash
python scripts/train_yolo_segmenter.py \
  --config configs/benchmark.yaml \
  --run-id baseline_c1__yolo11m_seg__seed13 \
  --weights yolo11m-seg.pt
```

Evaluate all completed PyTorch semantic checkpoints in the registry:

```bash
python scripts/evaluate_registry.py --config configs/benchmark.yaml
python scripts/aggregate_results.py
```

## Notes

- `segmentation_models_pytorch` is required for the SMP backbones listed in the config.
- CUDA currently fails on this machine with an NVIDIA driver/library mismatch, so full training needs the GPU environment fixed first.
- New generated folders are intentionally skipped until they exist; this prevents accidental baseline-only manifests for incomplete generation conditions.
