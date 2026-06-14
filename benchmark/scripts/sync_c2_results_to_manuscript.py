#!/usr/bin/env python
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmark"
REPORTS = BENCHMARK / "results" / "reports" / "c2_inpainting_5x5"
MATRIX = BENCHMARK / "results" / "manifests" / "c2_inpainting_5x5" / "run_matrix.csv"
PAPER = ROOT / "overleaf_microplastic_project" / "main.tex"
INCLUDE = ROOT / "overleaf_microplastic_project" / "c2_current_results.tex"


def fmt(value: float) -> str:
    return f"{value:.3f}"


def label_escape(value: str) -> str:
    return value.replace("_", "\\_")


def pm(mean: float, sd: float) -> str:
    return f"{fmt(mean)} $\\pm$ {fmt(sd)}"


def rows_condition(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            f"    {row.label} & {int(row.runs)} & {pm(row.dice_mean_mean, row.dice_mean_sd)} "
            f"& {pm(row.iou_mean_mean, row.iou_mean_sd)} & {fmt(row.boundary_f1_mean_mean)} \\\\"
        )
    return "\n".join(rows)


def rows_model(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            f"    {row.label} & {int(row.runs)} & {pm(row.dice_mean_mean, row.dice_mean_sd)} "
            f"& {pm(row.iou_mean_mean, row.iou_mean_sd)} & {fmt(row.boundary_f1_mean_mean)} \\\\"
        )
    return "\n".join(rows)


def rows_validation(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            f"    {row.label} & {int(row.runs)} & {pm(row.best_val_dice_mean, row.best_val_dice_sd)} "
            f"& {pm(row.best_val_iou_mean, row.best_val_iou_sd)} & {fmt(row.last_val_dice_mean)} \\\\"
        )
    return "\n".join(rows)


def generated_section() -> tuple[str, dict[str, str]]:
    condition = pd.read_csv(REPORTS / "c3_completed_by_condition.csv")
    model = pd.read_csv(REPORTS / "c3_completed_by_model.csv")
    validation = pd.read_csv(REPORTS / "validation_by_condition.csv")
    top = pd.read_csv(REPORTS / "c3_completed_run_metrics.csv").sort_values("dice_mean", ascending=False)
    active = pd.read_csv(REPORTS / "active_runs.csv")
    not_started = pd.read_csv(REPORTS / "not_started_runs.csv")
    histories = pd.read_csv(REPORTS / "training_history_summary.csv")
    matrix = pd.read_csv(MATRIX)

    completed = len(top)
    histories_count = len(histories)
    active_count = len(active)
    not_started_count = len(not_started)
    planned = len(matrix)

    best = top.iloc[0]
    best_run_condition = condition.set_index("condition").loc[best.condition, "label"]
    best_run_model = model.set_index("model").loc[best.model, "label"]
    best_seed = int(best.seed)

    best_condition_row = condition.iloc[0]
    best_model_row = model.iloc[0]

    section = rf"""\section{{Results}}

\subsection{{Segmentation Results}}

The current benchmark state contains two complementary sources of evidence. First, {completed} completed checkpoints from the {planned}-run Cohort 2 inpainting semantic matrix have been evaluated on the locked C3-clean test set. These are held-out test metrics and are therefore the primary evidence for current empirical claims. Second, {histories_count} training histories are present, including {active_count} runs that are still active. Validation metrics are useful for monitoring training health, but they are not used as final evidence for cross-domain generalization.

\begin{{table}}[H]
  \centering
  \caption{{Current execution status for the Cohort 2 inpainting semantic benchmark. The matrix contains five inpainting conditions, five semantic segmentation models, and three seeds.}}
  \label{{tab:current-status}}
  \begin{{tabular}}{{lr}}
    \toprule
    \textbf{{Status item}} & \textbf{{Count}} \\
    \midrule
    Planned Cohort 2 inpainting runs & {planned} \\
    C3-clean evaluated completed checkpoints & {completed} \\
    Training histories present & {histories_count} \\
    Active training runs & {active_count} \\
    Not yet started runs & {not_started_count} \\
    \bottomrule
  \end{{tabular}}
\end{{table}}

Table~\ref{{tab:c2-condition-results}} summarizes held-out C3-clean performance by inpainting condition. {best_condition_row.label} has the strongest condition-level mean performance, with Dice {pm(best_condition_row.dice_mean_mean, best_condition_row.dice_mean_sd)} and IoU {pm(best_condition_row.iou_mean_mean, best_condition_row.iou_mean_sd)}. All five inpainting conditions have 15 evaluated runs, so the condition-level comparison is balanced across the five semantic segmentation models and three random seeds.

\begin{{table}}[H]
  \centering
  \caption{{Current C3-clean performance by Cohort 2 inpainting condition. Values are mean $\pm$ standard deviation over completed evaluated checkpoints only.}}
  \label{{tab:c2-condition-results}}
  \begin{{tabular}}{{lrrrr}}
    \toprule
    \textbf{{Condition}} & \textbf{{Runs}} & \textbf{{Dice}} & \textbf{{IoU}} & \textbf{{Boundary F1}} \\
    \midrule
{rows_condition(condition)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

Model choice has a larger effect than the current differences among completed inpainting conditions. Table~\ref{{tab:c2-model-results}} shows that {best_model_row.label} is the strongest completed model family, with mean Dice {pm(best_model_row.dice_mean_mean, best_model_row.dice_mean_sd)} and IoU {pm(best_model_row.iou_mean_mean, best_model_row.iou_mean_sd)}. This pattern suggests that decoder capacity, multiscale features, and model architecture remain central for small foreground segmentation under ecological clutter.

\begin{{table}}[H]
  \centering
  \caption{{Current C3-clean performance by segmentation model. Values are mean $\pm$ standard deviation over completed evaluated checkpoints only.}}
  \label{{tab:c2-model-results}}
  \begin{{tabular}}{{lrrrr}}
    \toprule
    \textbf{{Model}} & \textbf{{Runs}} & \textbf{{Dice}} & \textbf{{IoU}} & \textbf{{Boundary F1}} \\
    \midrule
{rows_model(model)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

The best completed single run is {best_run_condition} with {best_run_model} at seed {best_seed}, which achieves C3-clean Dice {fmt(best.dice_mean)} with a 95\% bootstrap confidence interval of {fmt(best.dice_ci_low)}--{fmt(best.dice_ci_high)} and IoU {fmt(best.iou_mean)} with a 95\% bootstrap confidence interval of {fmt(best.iou_ci_low)}--{fmt(best.iou_ci_high)}. These top-run results show that synthetic ecological inpainting can support useful C3-clean segmentation when paired with a strong segmentation architecture, while the spread across seeds and models cautions against relying on any single generator-model pairing.

\begin{{table}}[H]
  \centering
  \caption{{Training-validation summary by condition. These values monitor training health and checkpoint selection; they are not substitutes for held-out C3-clean metrics.}}
  \label{{tab:validation-condition-results}}
  \begin{{tabular}}{{lrrrr}}
    \toprule
    \textbf{{Condition}} & \textbf{{Runs}} & \textbf{{Best val Dice}} & \textbf{{Best val IoU}} & \textbf{{Last val Dice}} \\
    \midrule
{rows_validation(validation)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

The earlier single-condition smoke validation remains useful as an end-to-end check of the pipeline. This run trained a U-Net with a ResNet34 encoder using one seed on an SDXL-inpainted Cohort 2 synthetic training condition, then evaluated the resulting checkpoint on the 97-image Cohort 3 test set.

\begin{{table}}[H]
  \centering
  \caption{{Preliminary single-condition Cohort 3 validation metrics. This run validates the end-to-end evaluation workflow; the full benchmark is required for condition-level conclusions.}}
  \label{{tab:preliminary-validation}}
  \begin{{tabular}}{{lcc}}
    \toprule
    \textbf{{Metric}} & \textbf{{Mean}} & \textbf{{95\% bootstrap CI}} \\
    \midrule
    Dice & 0.067 & 0.041--0.096 \\
    Mask IoU & 0.041 & 0.024--0.061 \\
    Precision & 0.189 & -- \\
    Recall & 0.051 & -- \\
    Boundary F1 & 0.085 & -- \\
    Foreground area error & 0.015 & -- \\
    \bottomrule
  \end{{tabular}}
\end{{table}}

Because this smoke run uses one synthetic condition, one architecture, and one seed, we interpret it only as pipeline validation rather than final evidence for a synthetic-data benefit. The current C2 benchmark tables provide stronger evidence because they aggregate multiple inpainting conditions, models, and seeds. The difference between high validation Dice values in Table~\ref{{tab:validation-condition-results}} and lower C3-clean Dice values in Tables~\ref{{tab:c2-condition-results}}--\ref{{tab:c2-model-results}} is scientifically important: it shows that within-condition validation performance can overstate ecological generalization. The benchmark therefore reports C3-clean metrics as the primary outcome.
"""
    facts = {
        "completed": str(completed),
        "planned": str(planned),
        "best_condition": str(best_condition_row.label),
        "best_condition_dice": fmt(best_condition_row.dice_mean_mean),
        "best_condition_iou": fmt(best_condition_row.iou_mean_mean),
        "best_model": str(best_model_row.label),
        "best_model_dice": fmt(best_model_row.dice_mean_mean),
        "best_model_iou": fmt(best_model_row.iou_mean_mean),
        "best_run_condition": best_run_condition,
        "best_run_model": best_run_model,
        "best_run_seed": str(best_seed),
        "best_run_dice": fmt(best.dice_mean),
        "best_run_iou": fmt(best.iou_mean),
    }
    return section, facts


def sync_main(section: str, facts: dict[str, str]) -> None:
    INCLUDE.write_text(section, encoding="utf-8")
    tex = PAPER.read_text(encoding="utf-8")
    tex = re.sub(
        r"\\section\{(?:Current Benchmark Results|Results)\}.*?(?=\\section\{Discussion\})",
        lambda _: "\\input{c2_current_results}\n\n",
        tex,
        flags=re.DOTALL,
    )
    abstract_sentence = (
        f"Current completed inpainting results include {facts['completed']} evaluated checkpoints "
        f"from a {facts['planned']}-run semantic segmentation matrix. "
        f"Across those completed runs, {facts['best_condition']} currently gives the strongest "
        f"condition-level mean Dice and IoU ({facts['best_condition_dice']} and {facts['best_condition_iou']}), "
        f"while {facts['best_model']} gives the strongest model-level mean Dice and IoU "
        f"({facts['best_model_dice']} and {facts['best_model_iou']}). The best completed single run is "
        f"{facts['best_run_condition']} with {facts['best_run_model']} at seed {facts['best_run_seed']}, "
        f"with held-out Dice {facts['best_run_dice']} and IoU {facts['best_run_iou']}."
    )
    tex = re.sub(
        r"Current completed (?:Cohort 2 )?inpainting results include .*? with (?:C3-clean|held-out) Dice [0-9.]+ and IoU [0-9.]+\.",
        lambda _: abstract_sentence,
        tex,
        flags=re.DOTALL,
    )
    PAPER.write_text(tex, encoding="utf-8")


def main() -> None:
    section, facts = generated_section()
    sync_main(section, facts)
    print(f"Wrote {INCLUDE}")
    print(f"Updated {PAPER}")


if __name__ == "__main__":
    main()
