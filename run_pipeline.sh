#!/usr/bin/env bash
# =============================================================================
# Microplastic Synthetic Data Augmentation — Full Pipeline Runner
#
# Runs the complete pipeline for every generative model, then executes the
# comprehensive evaluation framework.
#
# Usage:
#   ./run_pipeline.sh                    # run everything
#   MODELS="lama mat" ./run_pipeline.sh  # run only selected models
#   SKIP_EXISTING=1 ./run_pipeline.sh    # skip steps whose output already exists
#
# Configuration (all overridable via environment variables):
#   MODELS            space-separated list of generators to run (default: gan sd lama mat)
#   DEVICE            CUDA device string (default: cuda)
#   NUM_IMAGES        synthetic images to generate per model (default: 10000)
#   EPOCHS_GAN        GAN training epochs (default: 500)
#   EPOCHS_SD         SD training epochs (default: 100)
#   EPOCHS_LAMA       LaMa training epochs (default: 200)
#   EPOCHS_MAT        MAT training epochs (default: 200)
#   EPOCHS_SEG        segmentation training epochs (default: 100)
#   SAMPLES_PER_EPOCH samples per epoch for segmentation (default: 10000)
#   BATCH_SEG         batch size for segmentation training (default: 4)
#   SKIP_EXISTING     set to 1 to skip steps whose output already exists (default: 0)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS="${MODELS:-gan sd lama mat}"
read -ra MODELS_ARR <<< "$MODELS"   # split into array so loops work correctly
DEVICE="${DEVICE:-cuda}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
EPOCHS_GAN="${EPOCHS_GAN:-500}"
EPOCHS_SD="${EPOCHS_SD:-100}"
EPOCHS_LAMA="${EPOCHS_LAMA:-200}"
EPOCHS_MAT="${EPOCHS_MAT:-200}"
EPOCHS_SEG="${EPOCHS_SEG:-100}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-10000}"
BATCH_SEG="${BATCH_SEG:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
DATA="$ROOT/data"
CKPT="$ROOT/checkpoints"
SPLITS="$DATA/splits"
OUTPUTS="$ROOT/outputs"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

mkdir -p "$LOG_DIR" "$CKPT" "$OUTPUTS"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; BOLD='\033[1m'; RESET='\033[0m'

log()     { echo -e "${CYAN}[$(date '+%H:%M:%S')]${RESET} $*"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] DONE${RESET} $*"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] SKIP${RESET} $*"; }
err()     { echo -e "${RED}[$(date '+%H:%M:%S')] FAIL${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════${RESET}"; \
            echo -e "${BOLD}${CYAN}  $*${RESET}"; \
            echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${RESET}"; }

# Run a command, tee its output to a log file, and time it.
# Usage: run_logged <log_name> <description> <command...>
run_logged() {
    local log_name="$1"; local desc="$2"; shift 2
    local log_file="$LOG_DIR/${log_name}.log"
    log "$desc"
    local start=$SECONDS
    if "$@" 2>&1 | tee "$log_file"; then
        local elapsed=$(( SECONDS - start ))
        success "$desc  (${elapsed}s)  log: $log_file"
    else
        err "$desc FAILED — see $log_file"
        exit 1
    fi
}

# Check whether to skip a step based on a sentinel file/directory.
# Returns 0 (true) if we should skip, 1 (false) if we should run.
should_skip() {
    local sentinel="$1"
    [[ "$SKIP_EXISTING" == "1" ]] && [[ -e "$sentinel" ]]
}

# Map model name to its generator checkpoint path
ckpt_for() {
    case "$1" in
        gan)  echo "$CKPT/gan/generator.pth" ;;
        sd)   echo "$CKPT/sd/unet_final" ;;
        lama) echo "$CKPT/lama/generator.pth" ;;
        mat)  echo "$CKPT/mat/generator.pth" ;;
    esac
}

# Map model name to training epoch count
epochs_for() {
    case "$1" in
        gan)  echo "$EPOCHS_GAN" ;;
        sd)   echo "$EPOCHS_SD" ;;
        lama) echo "$EPOCHS_LAMA" ;;
        mat)  echo "$EPOCHS_MAT" ;;
    esac
}

# Batch size during generator training
batch_for() {
    case "$1" in
        gan)  echo "16" ;;
        sd)   echo "1" ;;
        lama) echo "8" ;;
        mat)  echo "4" ;;
    esac
}

# ---------------------------------------------------------------------------
# Print configuration summary
# ---------------------------------------------------------------------------
header "Pipeline configuration"
echo "  ROOT          : $ROOT"
echo "  MODELS        : $MODELS"
echo "  DEVICE        : $DEVICE"
echo "  NUM_IMAGES    : $NUM_IMAGES"
echo "  EPOCHS        : GAN=$EPOCHS_GAN  SD=$EPOCHS_SD  LaMa=$EPOCHS_LAMA  MAT=$EPOCHS_MAT"
echo "  EPOCHS_SEG    : $EPOCHS_SEG  (samples/epoch=$SAMPLES_PER_EPOCH)"
echo "  SKIP_EXISTING : $SKIP_EXISTING"
echo ""

# ---------------------------------------------------------------------------
# STEP 1 — Preprocess: dilate masks
# ---------------------------------------------------------------------------
header "Step 1 — Dilate masks"

if should_skip "$DATA/c1/masks_dilated"; then
    warn "masks_dilated already exists, skipping."
else
    run_logged "step1_preprocess" "Dilating c1 masks" \
        python "$ROOT/scripts/01_preprocess.py" \
            --src "$DATA/c1/masks" \
            --dst "$DATA/c1/masks_dilated" \
            --kernel_size 3 \
            --iterations 4
fi

# ---------------------------------------------------------------------------
# STEPS 2 + 3 — Train each generator, then generate synthetic images
# ---------------------------------------------------------------------------
for MODEL in "${MODELS_ARR[@]}"; do
    header "Model: $MODEL — Train generator"
    CKPT_PATH="$(ckpt_for "$MODEL")"
    EPOCHS="$(epochs_for "$MODEL")"
    BATCH="$(batch_for "$MODEL")"
    GEN_DIR="$DATA/c2/gen_${MODEL}"

    # --- Step 2: Train ---
    if should_skip "$CKPT_PATH"; then
        warn "Checkpoint $CKPT_PATH exists — skipping training."
    else
        EXTRA_ARGS=()
        if [[ "$MODEL" == "sd" ]]; then
            EXTRA_ARGS+=(--grad_accum 4 --mixed_precision fp16)
        elif [[ "$MODEL" == "lama" ]]; then
            EXTRA_ARGS+=(--ffc_blocks 9 --lambda_rec 10.0)
        elif [[ "$MODEL" == "mat" ]]; then
            EXTRA_ARGS+=(--style_dim 256 --embed_dim 512 --num_heads 8 --depth 6 --lambda_rec 10.0)
        fi

        run_logged "step2_train_${MODEL}" "Training $MODEL generator (epochs=$EPOCHS)" \
            python "$ROOT/scripts/02_train_generator.py" \
                --model      "$MODEL" \
                --image_dir  "$DATA/c1/imgs" \
                --mask_dir   "$DATA/c1/masks_dilated" \
                --output_dir "$CKPT/$MODEL" \
                --epochs     "$EPOCHS" \
                --batch_size "$BATCH" \
                --device     "$DEVICE" \
                "${EXTRA_ARGS[@]}"
    fi

    # --- Step 3: Generate synthetic images ---
    header "Model: $MODEL — Generate $NUM_IMAGES synthetic images"

    if should_skip "$GEN_DIR"; then
        warn "$GEN_DIR exists — skipping generation."
    else
        EXTRA_ARGS=()
        if [[ "$MODEL" == "sd" ]]; then
            EXTRA_ARGS+=(--inference_steps 50)
        fi

        run_logged "step3_generate_${MODEL}" "Generating synthetic images with $MODEL" \
            python "$ROOT/scripts/03_generate_synthetic.py" \
                --model      "$MODEL" \
                --checkpoint "$CKPT_PATH" \
                --image_dir  "$DATA/c2/imgs" \
                --mask_dir   "$DATA/c1/masks_dilated" \
                --output_dir "$GEN_DIR" \
                --num_images "$NUM_IMAGES" \
                --device     "$DEVICE" \
                "${EXTRA_ARGS[@]}"
    fi
done

# ---------------------------------------------------------------------------
# STEP 4 — Prepare data splits
# ---------------------------------------------------------------------------
header "Step 4 — Prepare data splits"

# Baseline split (real data only)
if should_skip "$SPLITS/baseline"; then
    warn "splits/baseline exists — skipping."
else
    run_logged "step4_split_baseline" "Preparing baseline split (real data only)" \
        python "$ROOT/scripts/04_prepare_data.py" \
            --real_imgs  "$DATA/c1/imgs" \
            --real_masks "$DATA/c1/masks" \
            --output_dir "$SPLITS/baseline" \
            --real_only
fi

# Augmented splits
for MODEL in "${MODELS_ARR[@]}"; do
    if should_skip "$SPLITS/$MODEL"; then
        warn "splits/$MODEL exists — skipping."
    else
        GEN_DIR="$DATA/c2/gen_${MODEL}"
        MASK_DIR="${GEN_DIR}_masks"
        run_logged "step4_split_${MODEL}" "Preparing $MODEL-augmented split" \
            python "$ROOT/scripts/04_prepare_data.py" \
                --real_imgs  "$DATA/c1/imgs" \
                --real_masks "$DATA/c1/masks" \
                --gen_imgs   "$GEN_DIR" \
                --gen_masks  "$MASK_DIR" \
                --output_dir "$SPLITS/$MODEL"
    fi
done

# ---------------------------------------------------------------------------
# STEP 5 — Train segmentation models
# ---------------------------------------------------------------------------
header "Step 5 — Train segmentation models"

# Baseline segmentation
if should_skip "$CKPT/seg_baseline/best_model.pth"; then
    warn "seg_baseline checkpoint exists — skipping."
else
    run_logged "step5_seg_baseline" "Training baseline segmentation" \
        python "$ROOT/scripts/05_train_segmentation.py" \
            --data_root         "$SPLITS/baseline" \
            --output_dir        "$CKPT/seg_baseline" \
            --epochs            "$EPOCHS_SEG" \
            --samples_per_epoch "$SAMPLES_PER_EPOCH" \
            --batch_size        "$BATCH_SEG" \
            --device            "$DEVICE"
fi

# Augmented segmentation models
for MODEL in "${MODELS_ARR[@]}"; do
    SEG_CKPT="$CKPT/seg_${MODEL}/best_model.pth"
    if should_skip "$SEG_CKPT"; then
        warn "seg_${MODEL} checkpoint exists — skipping."
    else
        run_logged "step5_seg_${MODEL}" "Training $MODEL-augmented segmentation" \
            python "$ROOT/scripts/05_train_segmentation.py" \
                --data_root         "$SPLITS/$MODEL" \
                --output_dir        "$CKPT/seg_${MODEL}" \
                --epochs            "$EPOCHS_SEG" \
                --samples_per_epoch "$SAMPLES_PER_EPOCH" \
                --batch_size        "$BATCH_SEG" \
                --device            "$DEVICE"
    fi
done

# ---------------------------------------------------------------------------
# STEP 6 + 7 — Per-model comparison + comprehensive evaluation
# ---------------------------------------------------------------------------
header "Step 6 — Per-model comparisons vs baseline"

for MODEL in "${MODELS_ARR[@]}"; do
    SEG_B="$CKPT/seg_baseline/best_model.pth"
    SEG_M="$CKPT/seg_${MODEL}/best_model.pth"
    OUT="$OUTPUTS/comparison_${MODEL}"

    if [[ ! -f "$SEG_M" ]]; then
        warn "seg_${MODEL} checkpoint not found — skipping comparison."
        continue
    fi

    if should_skip "$OUT"; then
        warn "comparison_${MODEL} exists — skipping."
    else
        run_logged "step6_compare_${MODEL}" "Comparing baseline vs $MODEL on test set" \
            python "$ROOT/scripts/06_evaluate.py" \
                --model_a    "$SEG_B" \
                --model_b    "$SEG_M" \
                --label_a    "Baseline" \
                --label_b    "$MODEL" \
                --data_root  "$DATA/c3" \
                --output_dir "$OUT" \
                --device     "$DEVICE"
    fi
done

header "Step 7 — Comprehensive evaluation (all models)"

run_logged "step7_evaluate_all" "Running full evaluation framework" \
    python "$ROOT/scripts/07_evaluate_all.py" \
        --data_root  "$DATA/c3" \
        --ckpt_dir   "$CKPT" \
        --output_dir "$OUTPUTS/evaluation" \
        --device     "$DEVICE"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header "Pipeline complete"
echo ""
echo "  Logs           : $LOG_DIR/"
echo "  Checkpoints    : $CKPT/"
echo "  Generated data : $DATA/c2/"
echo "  Splits         : $SPLITS/"
echo "  Evaluation     : $OUTPUTS/evaluation/"
echo "  Report         : $OUTPUTS/evaluation/report.html"
echo ""
success "All steps finished successfully."
