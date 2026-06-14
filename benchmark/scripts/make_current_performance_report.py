#!/usr/bin/env python
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
import heapq

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
C2 = RESULTS / "runs" / "c2_inpainting_5x5"
REPORTS = RESULTS / "reports" / "c2_inpainting_5x5"
RUN_MATRIX = RESULTS / "manifests" / "c2_inpainting_5x5" / "run_matrix.csv"
STATUS = RESULTS / "logs" / "benchmark_20260528_205702" / "status.csv"


CONDITION_LABELS = {
    "c2_sdxl_inpaint": "C2 SDXL inpaint",
    "c2_flux_inpaint": "C2 FLUX inpaint",
    "c2_sd2_inpaint": "C2 SD2 inpaint",
    "c2_sd2_fiber_inpaint": "C2 SD2 fiber inpaint",
    "c2_sdxl_texture_inpaint": "C2 SDXL texture inpaint",
}

MODEL_LABELS = {
    "smp_unet_resnet34": "U-Net ResNet34",
    "smp_unetpp_effb4": "U-Net++ EfficientNet-B4",
    "smp_deeplabv3plus_resnet50": "DeepLabV3+ ResNet50",
    "smp_fpn_effb3": "FPN EfficientNet-B3",
    "monai_unet": "MONAI U-Net",
}


def label(value: str, labels: dict[str, str]) -> str:
    return labels.get(value, value.replace("_", " "))


def split_run_id(run_id: str) -> dict[str, str | int]:
    condition, model, seed = run_id.split("__")
    return {"condition": condition, "model": model, "seed": int(seed.replace("seed", ""))}


def load_c3_summaries() -> pd.DataFrame:
    rows = []
    for path in C2.glob("*/c3_clean_metrics.summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_id = path.parent.name
        rows.append({"run_id": run_id, **split_run_id(run_id), **payload})
    return pd.DataFrame(rows)


def load_histories() -> pd.DataFrame:
    rows = []
    for path in C2.glob("*/history.jsonl"):
        run_id = path.parent.name
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not entries:
            continue
        best = max(entries, key=lambda row: row.get("dice_mean", -1.0))
        last = entries[-1]
        recent = entries[-5:] if len(entries) >= 5 else entries
        avg_recent_seconds = sum(row.get("seconds", 0.0) for row in recent) / len(recent)
        rows.append(
            {
                "run_id": run_id,
                **split_run_id(run_id),
                "epochs_logged": int(last["epoch"]),
                "last_val_dice": last.get("dice_mean"),
                "last_val_iou": last.get("iou_mean"),
                "best_val_dice": best.get("dice_mean"),
                "best_val_iou": best.get("iou_mean"),
                "best_val_epoch": int(best["epoch"]),
                "avg_recent_epoch_seconds": avg_recent_seconds,
                "estimated_remaining_hours": max(0, 80 - int(last["epoch"])) * avg_recent_seconds / 3600.0,
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(group_col)
    rows = []
    for name, group in grouped:
        row = {group_col: name, "label": label(str(name), CONDITION_LABELS if group_col == "condition" else MODEL_LABELS), "runs": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"{metrics[0]}_mean", ascending=False)


def status_summary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_csv(RUN_MATRIX)
    if STATUS.exists():
        status = pd.read_csv(STATUS)
        latest = status.drop_duplicates("run_id", keep="last")
    else:
        status = pd.DataFrame(columns=["run_id", "status"])
        latest = status
    not_started = matrix[~matrix["run_id"].isin(set(status["run_id"]))]
    active = latest[latest["status"] == "started"].copy()
    return matrix, active, not_started


def fmt(value: float) -> str:
    return f"{value:.3f}"


def markdown_table(frame: pd.DataFrame, cols: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    out = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in frame[cols].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value))
        out.append("|" + "|".join(values) + "|")
    return "\n".join(out)


def estimate_full_matrix_eta(
    matrix: pd.DataFrame,
    active: pd.DataFrame,
    not_started: pd.DataFrame,
    histories: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    if active.empty:
        return 0.0, pd.DataFrame()

    meta = {row.run_id: row for row in matrix.itertuples(index=False)}
    active_histories = histories[histories["run_id"].isin(set(active["run_id"]))].copy()
    active_remaining = {
        row.run_id: float(row.estimated_remaining_hours)
        for row in active_histories.itertuples(index=False)
    }

    completed_status = pd.read_csv(STATUS) if STATUS.exists() else pd.DataFrame()
    completed_status = completed_status[completed_status["status"] == "finished"].copy() if not completed_status.empty else completed_status
    if not completed_status.empty:
        completed_status = completed_status[completed_status["run_id"].isin(meta)]
        completed_status["model"] = completed_status["run_id"].map(lambda run_id: meta[run_id].model)
        model_hours = (completed_status.groupby("model")["seconds"].median() / 3600.0).to_dict()
    else:
        model_hours = {}

    # Fall back to active epoch rates when a queued model has no completed analogue.
    for row in active_histories.itertuples(index=False):
        model = meta[row.run_id].model
        estimated_full_run = float(row.avg_recent_epoch_seconds) * 80.0 / 3600.0
        model_hours.setdefault(model, estimated_full_run)

    heap: list[tuple[float, str]] = []
    for row in active.itertuples(index=False):
        remaining = active_remaining.get(row.run_id)
        if remaining is not None:
            heapq.heappush(heap, (remaining, row.worker))

    assignments = []
    for row in not_started.itertuples(index=False):
        if not heap:
            break
        start_hours, worker = heapq.heappop(heap)
        duration = float(model_hours.get(row.model, 28.0))
        done_hours = start_hours + duration
        assignments.append(
            {
                "run_id": row.run_id,
                "worker": worker,
                "start_after_hours": start_hours,
                "estimated_duration_hours": duration,
                "done_after_hours": done_hours,
            }
        )
        heapq.heappush(heap, (done_hours, worker))

    eta_hours = max([item[0] for item in heap], default=0.0)
    return eta_hours, pd.DataFrame(assignments)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    c3 = load_c3_summaries()
    histories = load_histories()
    matrix, active, not_started = status_summary()

    c3_condition = summarize(c3, "condition", ["dice_mean", "iou_mean", "precision_mean", "recall_mean", "boundary_f1_mean"])
    c3_model = summarize(c3, "model", ["dice_mean", "iou_mean", "precision_mean", "recall_mean", "boundary_f1_mean"])
    val_condition = summarize(histories, "condition", ["best_val_dice", "best_val_iou", "last_val_dice", "last_val_iou"])
    full_eta_hours, queued_schedule = estimate_full_matrix_eta(matrix, active, not_started, histories)

    c3.to_csv(REPORTS / "c3_completed_run_metrics.csv", index=False)
    c3_condition.to_csv(REPORTS / "c3_completed_by_condition.csv", index=False)
    c3_model.to_csv(REPORTS / "c3_completed_by_model.csv", index=False)
    histories.to_csv(REPORTS / "training_history_summary.csv", index=False)
    active.to_csv(REPORTS / "active_runs.csv", index=False)
    not_started.to_csv(REPORTS / "not_started_runs.csv", index=False)
    val_condition.to_csv(REPORTS / "validation_by_condition.csv", index=False)
    queued_schedule.to_csv(REPORTS / "queued_run_eta.csv", index=False)

    latest = datetime.now()
    active_histories = histories[histories["run_id"].isin(set(active["run_id"]))]
    next_active_hours = active_histories["estimated_remaining_hours"].min() if not active_histories.empty else 0.0
    max_active_hours = active_histories["estimated_remaining_hours"].max() if not active_histories.empty else 0.0

    best_text = "No C3-clean summaries are available yet."
    top_runs = pd.DataFrame()
    if not c3.empty:
        best = c3.sort_values("dice_mean", ascending=False).iloc[0]
        best_text = (
            f"Best completed C3-clean run: {label(best.condition, CONDITION_LABELS)} / "
            f"{label(best.model, MODEL_LABELS)} / seed {best.seed}, "
            f"Dice {fmt(best.dice_mean)}, IoU {fmt(best.iou_mean)}."
        )
        top_runs = c3.sort_values("dice_mean", ascending=False).head(10).copy()
        top_runs["condition_label"] = top_runs["condition"].map(lambda value: label(str(value), CONDITION_LABELS))
        top_runs["model_label"] = top_runs["model"].map(lambda value: label(str(value), MODEL_LABELS))

    report = f"""# Current Performance Report

Generated: {latest.strftime('%Y-%m-%d %H:%M:%S')}

## Run Status

- C2 run matrix size: {len(matrix)}
- C3-clean evaluated C2 checkpoints: {len(c3)}
- Training histories present: {len(histories)}
- Active runs: {len(active)}
- Not yet started: {len(not_started)}
- Soonest active completion estimate: {next_active_hours:.1f} hours ({(latest + timedelta(hours=float(next_active_hours))).strftime('%Y-%m-%d %H:%M')})
- Latest active completion estimate: {max_active_hours:.1f} hours ({(latest + timedelta(hours=float(max_active_hours))).strftime('%Y-%m-%d %H:%M')})
- Full 75-run matrix ETA estimate: {full_eta_hours:.1f} hours ({(latest + timedelta(hours=float(full_eta_hours))).strftime('%Y-%m-%d %H:%M')})

{best_text}

## C3-Clean Performance by Condition

{markdown_table(c3_condition, ['label', 'runs', 'dice_mean_mean', 'dice_mean_sd', 'iou_mean_mean', 'iou_mean_sd', 'precision_mean_mean', 'recall_mean_mean', 'boundary_f1_mean_mean'])}

## C3-Clean Performance by Model

{markdown_table(c3_model, ['label', 'runs', 'dice_mean_mean', 'dice_mean_sd', 'iou_mean_mean', 'iou_mean_sd', 'precision_mean_mean', 'recall_mean_mean', 'boundary_f1_mean_mean'])}

## Top Completed C3-Clean Runs

{markdown_table(top_runs, ['condition_label', 'model_label', 'seed', 'dice_mean', 'dice_ci_low', 'dice_ci_high', 'iou_mean', 'iou_ci_low', 'iou_ci_high'])}

## Validation Performance by Condition

These validation metrics come from training histories and are not a substitute for the locked C3-clean test set.

{markdown_table(val_condition, ['label', 'runs', 'best_val_dice_mean', 'best_val_dice_sd', 'best_val_iou_mean', 'best_val_iou_sd', 'last_val_dice_mean', 'last_val_iou_mean'])}

## Active Runs

{markdown_table(active_histories.sort_values('estimated_remaining_hours') if not active_histories.empty else active_histories, ['run_id', 'epochs_logged', 'last_val_dice', 'last_val_iou', 'best_val_dice', 'best_val_iou', 'estimated_remaining_hours'])}

## Queued Run ETA

{markdown_table(queued_schedule, ['run_id', 'worker', 'start_after_hours', 'estimated_duration_hours', 'done_after_hours'])}

## Interpretation

The completed C3-clean metrics should be treated as current, partial held-out test results for the completed checkpoints only. The C2 SDXL texture condition is still early in the queue, and the full 75-run C2 matrix is not yet complete. Validation metrics remain high for active and completed runs, but the C3-clean results are the authoritative generalization metrics for manuscript claims.
"""
    out = REPORTS / "current_performance_report.md"
    out.write_text(report, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
