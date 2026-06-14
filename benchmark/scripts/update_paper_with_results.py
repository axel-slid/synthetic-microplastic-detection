#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER = ROOT / "overleaf_microplastic_project" / "main.tex"


CONDITION_LABELS = {
    "baseline_c1": "Baseline C1",
    "legacy_gan": "Legacy GAN",
    "legacy_sd": "Legacy SD",
    "legacy_gen1": "Legacy gen1",
    "legacy_c2_gen": "Legacy c2\\_gen",
    "new_sdxl_inpaint": "New SDXL inpaint",
    "new_flux_inpaint": "New FLUX inpaint",
    "c2_sdxl_inpaint": "C2 SDXL inpaint",
    "c2_flux_inpaint": "C2 FLUX inpaint",
    "c2_sd2_inpaint": "C2 SD2 inpaint",
    "c2_sd2_fiber_inpaint": "C2 SD2 fiber inpaint",
    "c2_sdxl_texture_inpaint": "C2 SDXL texture inpaint",
    "pure_real": "Pure real",
    "real_plus_inpainting": "Real plus inpainting",
    "real_plus_synthetic": "Real plus synthetic",
    "pure_synthetic": "Pure synthetic",
    "pure_inpainting": "Pure inpainting",
}


MODEL_LABELS = {
    "smp_unet_resnet34": "U-Net ResNet34",
    "smp_unetpp_effb4": "U-Net++ EfficientNet-B4",
    "smp_deeplabv3plus_resnet50": "DeepLabV3+ ResNet50",
    "smp_fpn_effb3": "FPN EfficientNet-B3",
    "monai_unet": "MONAI U-Net",
    "segformer_b2": "SegFormer-B2",
    "yolo11m_seg": "YOLO11m-seg",
    "yolo26n_det": "YOLO26n-det",
}


def fmt(value: float) -> str:
    return f"{value:.3f}"


def latex_rows(frame: pd.DataFrame, group_col: str, label_map: dict[str, str]) -> str:
    rows = []
    grouped = frame.groupby(group_col)
    for name, group in grouped:
        rows.append(
            {
                "label": label_map.get(str(name), str(name).replace("_", "\\_")),
                "runs": len(group),
                "dice_mean": group["dice_mean"].mean(),
                "dice_sd": group["dice_mean"].std(ddof=0),
                "iou_mean": group["iou_mean"].mean(),
                "iou_sd": group["iou_mean"].std(ddof=0),
            }
        )
    rows = sorted(rows, key=lambda r: r["dice_mean"], reverse=True)
    return "\n".join(
        f"    {row['label']} & {row['runs']} & {fmt(row['dice_mean'])} $\\pm$ {fmt(row['dice_sd'])} "
        f"& {fmt(row['iou_mean'])} $\\pm$ {fmt(row['iou_sd'])}\\\\"
        for row in rows
    )


def detection_latex_rows(frame: pd.DataFrame, group_col: str, label_map: dict[str, str]) -> str:
    rows = []
    for name, group in frame.groupby(group_col):
        rows.append(
            {
                "label": label_map.get(str(name), str(name).replace("_", "\\_")),
                "runs": len(group),
                "map50": group["box_map50"].mean(),
                "map50_95": group["box_map50_95"].mean(),
                "precision": group["box_precision_mean"].mean(),
                "recall": group["box_recall_mean"].mean(),
            }
        )
    rows = sorted(rows, key=lambda r: r["map50"], reverse=True)
    return "\n".join(
        f"    {row['label']} & {row['runs']} & {fmt(row['map50'])} "
        f"& {fmt(row['map50_95'])} & {fmt(row['precision'])} & {fmt(row['recall'])}\\\\"
        for row in rows
    )


def condition_table(frame: pd.DataFrame) -> str:
    return rf"""
\begin{{table}}[t]
  \centering
  \caption{{C3-clean benchmark performance by training condition. Values are mean $\pm$
  standard deviation over completed segmentation model and seed runs.}}
  \label{{tab:condition_results}}
  \begin{{tabular}}{{lrrr}}
    \toprule
    \textbf{{Training condition}} & \textbf{{Runs}} & \textbf{{Dice}} & \textbf{{IoU}}\\
    \midrule
{latex_rows(frame, "condition", CONDITION_LABELS)}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""".strip()


def model_table(frame: pd.DataFrame) -> str:
    return rf"""
\begin{{table}}[t]
  \centering
  \caption{{C3-clean benchmark performance by segmentation model. Values are mean $\pm$
  standard deviation over completed training conditions and seeds.}}
  \label{{tab:model_results}}
  \begin{{tabular}}{{lrrr}}
    \toprule
    \textbf{{Segmentation model}} & \textbf{{Runs}} & \textbf{{Dice}} & \textbf{{IoU}}\\
    \midrule
{latex_rows(frame, "model", MODEL_LABELS)}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""".strip()


def best_result_sentence(frame: pd.DataFrame) -> str:
    best = frame.sort_values("dice_mean", ascending=False).iloc[0]
    label = CONDITION_LABELS.get(best.condition, best.condition.replace("_", "\\_"))
    model = MODEL_LABELS.get(best.model, best.model.replace("_", "\\_"))
    return (
        f"The best single run was {label} with {model} at seed {int(best.seed)}, "
        f"with Dice {fmt(best.dice_mean)} and IoU {fmt(best.iou_mean)} on C3-clean."
    )


def build_results_section(frame: pd.DataFrame) -> str:
    n_runs = len(frame)
    n_conditions = frame["condition"].nunique()
    n_models = frame["model"].nunique()
    n_seeds = frame["seed"].nunique()
    return rf"""
\subsection{{Full-Matrix Benchmark Results}}

The available benchmark matrix has now executed on GPU and produced C3-clean metrics for
{n_runs} runs: {n_conditions} training conditions, {n_models} segmentation models, and
{n_seeds} random seeds. Each run is evaluated on the same locked 97-image C3-clean test
set. Tables~\ref{{tab:condition_results}} and~\ref{{tab:model_results}} summarize the
completed empirical results. {best_result_sentence(frame)}

{condition_table(frame)}

{model_table(frame)}

The two configured new-generation conditions, SDXL inpainting and FLUX inpainting, remain
separate from this completed available-data matrix unless their generated folders are
produced and the run matrix is expanded to the full 147-run design. The empirical claims
above are therefore limited to the baseline and legacy synthetic conditions that were
present at execution time.
""".strip()


def sd_condition_table(frame: pd.DataFrame) -> str:
    return rf"""
\begin{{table}}[H]
  \centering
  \caption{{Stable Diffusion ablation segmentation performance by training regime.
  Values are mean $\pm$ standard deviation over completed segmentation model and seed runs.}}
  \label{{tab:sd-ablation-condition-results}}
  \begin{{tabular}}{{lrrr}}
    \toprule
    \textbf{{Training regime}} & \textbf{{Runs}} & \textbf{{Dice}} & \textbf{{IoU}}\\
    \midrule
{latex_rows(frame, "condition", CONDITION_LABELS)}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""".strip()


def sd_model_table(frame: pd.DataFrame) -> str:
    return rf"""
\begin{{table}}[H]
  \centering
  \caption{{Stable Diffusion ablation segmentation performance by model.
  Values are mean $\pm$ standard deviation over completed training regimes and seeds.}}
  \label{{tab:sd-ablation-model-results}}
  \begin{{tabular}}{{lrrr}}
    \toprule
    \textbf{{Segmentation model}} & \textbf{{Runs}} & \textbf{{Dice}} & \textbf{{IoU}}\\
    \midrule
{latex_rows(frame, "model", MODEL_LABELS)}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""".strip()


def build_sd_ablation_section(frame: pd.DataFrame) -> str:
    n_runs = len(frame)
    n_conditions = frame["condition"].nunique()
    n_models = frame["model"].nunique()
    n_seeds = frame["seed"].nunique()
    best = frame.sort_values("dice_mean", ascending=False).iloc[0]
    best_condition = CONDITION_LABELS.get(best.condition, best.condition.replace("_", "\\_"))
    best_model = MODEL_LABELS.get(best.model, best.model.replace("_", "\\_"))
    return rf"""
\subsection{{Stable Diffusion Ablation Results}}

The five-regime Stable Diffusion ablation completed {n_runs} segmentation runs:
{n_conditions} training regimes, {n_models} segmentation models, and {n_seeds} random
seeds. The regimes isolate whether performance comes from real labeled microplastic
images, Stable Diffusion synthetic images, inpainted ecological images, or mixtures of
real and generated data. All checkpoints are evaluated on the same locked ecological
test set used by the core benchmark. Tables~\ref{{tab:sd-ablation-condition-results}}
and~\ref{{tab:sd-ablation-model-results}} summarize the held-out segmentation metrics.
The strongest single ablation run is {best_condition} with {best_model} at seed
{int(best.seed)}, reaching Dice {fmt(best.dice_mean)} and IoU {fmt(best.iou_mean)}.

{sd_condition_table(frame)}

{sd_model_table(frame)}
""".strip()


def detection_condition_table(frame: pd.DataFrame) -> str:
    return rf"""
\begin{{table}}[H]
  \centering
  \caption{{Detection ablation performance by training regime on the held-out ecological
  detection set. Values are means over completed YOLO detector seeds.}}
  \label{{tab:detection-ablation-results}}
  \begin{{tabular}}{{lrrrrr}}
    \toprule
    \textbf{{Training regime}} & \textbf{{Runs}} & \textbf{{mAP50}} & \textbf{{mAP50--95}} & \textbf{{Precision}} & \textbf{{Recall}}\\
    \midrule
{detection_latex_rows(frame, "condition", CONDITION_LABELS)}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""".strip()


def build_detection_section(frame: pd.DataFrame) -> str:
    n_runs = len(frame)
    n_conditions = frame["condition"].nunique()
    n_seeds = frame["seed"].nunique()
    best = frame.sort_values("box_map50", ascending=False).iloc[0]
    best_condition = CONDITION_LABELS.get(best.condition, best.condition.replace("_", "\\_"))
    return rf"""
\subsection{{Detection Ablation Results}}

The detection track completed {n_runs} YOLO detector runs across {n_conditions} training
regimes and {n_seeds} random seeds. These models use bounding boxes derived from the
same binary masks and are evaluated on the held-out ecological detection set. The track
therefore measures the screening task of localizing candidate microplastic regions,
while the segmentation tables above remain the primary evidence for mask quality and
morphology recovery. The strongest detector run by mAP50 is {best_condition} at seed
{int(best.seed)}, with mAP50 {fmt(best.box_map50)} and mAP50--95
{fmt(best.box_map50_95)}.

{detection_condition_table(frame)}
""".strip()


def read_aggregate(path: Path, expected_runs: int, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path).drop_duplicates("run_id", keep="last")
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    if len(frame) < expected_runs:
        raise SystemExit(f"Expected at least {expected_runs} aggregate rows in {path}, found {len(frame)}")
    return frame


def replace_or_insert_subsection(tex: str, title: str, section: str) -> str:
    pattern = rf"\n\\subsection\{{{re.escape(title)}\}}.*?(?=\n\\subsection\{{|\n\\section\{{Discussion\}})"
    if re.search(pattern, tex, flags=re.DOTALL):
        return re.sub(pattern, "\n" + section + "\n", tex, flags=re.DOTALL)
    marker = "\n\\section{Discussion}"
    if marker not in tex:
        raise SystemExit("Could not find Discussion section for result insertion")
    return tex.replace(marker, "\n" + section + "\n" + marker, 1)


def update_future_language(tex: str, sd_frame: pd.DataFrame | None, detection_frame: pd.DataFrame | None) -> str:
    if sd_frame is not None:
        tex = tex.replace(
            "These results motivate the follow-up ablation launched from this study: pure real, real plus inpainting, real plus Stable Diffusion synthetic images, pure synthetic, and pure inpainting training regimes.",
            "These results motivate the five-regime ablation reported below: pure real, real plus inpainting, real plus Stable Diffusion synthetic images, pure synthetic, and pure inpainting training regimes.",
        )
        tex = tex.replace(
            "The next experimental step is to extend the ablation from inpainting variants to broader data-mixture regimes: pure real microplastic images, real plus inpainting, real plus synthetic generation, pure synthetic generation, and pure inpainting.",
            "The completed five-regime ablation extends the comparison from inpainting variants to broader data-mixture regimes: pure real microplastic images, real plus inpainting, real plus synthetic generation, pure synthetic generation, and pure inpainting.",
        )
    if detection_frame is not None:
        tex = tex.replace(
            "a detection-track design for comparing no-synthetic, synthetic, full-synthetic, combined, and all-source training regimes.",
            "a completed detection-track benchmark comparing real-only, real-plus-generated, and generated-only training regimes.",
        )
    return tex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", default="results/reports/aggregate_c3_clean.csv")
    parser.add_argument("--expected-runs", type=int, default=105)
    parser.add_argument("--sd-aggregate", default="results/reports/sd_ablation/aggregate_c3_clean.csv")
    parser.add_argument("--sd-expected-runs", type=int, default=105)
    parser.add_argument("--detection-aggregate", default="results/reports/detection_ablation/aggregate_c3_clean.csv")
    parser.add_argument("--detection-expected-runs", type=int, default=15)
    parser.add_argument("--paper", default=str(DEFAULT_PAPER))
    args = parser.parse_args()

    aggregate_path = ROOT / "benchmark" / args.aggregate
    frame = read_aggregate(
        aggregate_path,
        args.expected_runs,
        {"condition", "model", "seed", "dice_mean", "iou_mean"},
    )

    sd_path = ROOT / "benchmark" / args.sd_aggregate
    sd_frame = None
    if sd_path.exists():
        sd_frame = read_aggregate(
            sd_path,
            args.sd_expected_runs,
            {"condition", "model", "seed", "dice_mean", "iou_mean"},
        )

    detection_path = ROOT / "benchmark" / args.detection_aggregate
    detection_frame = None
    if detection_path.exists():
        detection_frame = read_aggregate(
            detection_path,
            args.detection_expected_runs,
            {"condition", "model", "seed", "box_map50", "box_map50_95", "box_precision_mean", "box_recall_mean"},
        )

    paper = Path(args.paper)
    tex = paper.read_text(encoding="utf-8")
    tex = re.sub(
        r"The full factorial empirical benchmark requires GPU repair:.*?providing a complete scientific protocol for the planned 147-run study\.",
        (
            "The available-data benchmark has been executed on GPU, producing measured "
            "C3-clean Dice and IoU results for the baseline and four legacy synthetic "
            "training conditions. The configured SDXL and FLUX inpainting conditions "
            "remain future extensions until their generated folders are produced."
        ),
        tex,
        flags=re.DOTALL,
    )
    tex = re.sub(
        r"\\subsection\{Full-Matrix Execution Status\}.*?(?=\\section\{Discussion\})",
        lambda _: build_results_section(frame) + "\n\n",
        tex,
        flags=re.DOTALL,
    )
    tex = tex.replace(
        "the present environment reports an NVIDIA driver/library mismatch and PyTorch cannot\ninitialize CUDA.",
        "the benchmark was launched on the repaired CUDA environment.",
    )
    if sd_frame is not None:
        tex = replace_or_insert_subsection(tex, "Stable Diffusion Ablation Results", build_sd_ablation_section(sd_frame))
    if detection_frame is not None:
        tex = replace_or_insert_subsection(tex, "Detection Ablation Results", build_detection_section(detection_frame))
    tex = update_future_language(tex, sd_frame, detection_frame)
    paper.write_text(tex, encoding="utf-8")
    extra = []
    if sd_frame is not None:
        extra.append(f"{len(sd_frame)} SD ablation rows")
    if detection_frame is not None:
        extra.append(f"{len(detection_frame)} detection rows")
    suffix = "; " + ", ".join(extra) if extra else ""
    print(f"Updated {paper} with {len(frame)} aggregate rows{suffix}.")


if __name__ == "__main__":
    main()
