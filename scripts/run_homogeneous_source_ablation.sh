#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Homogeneous source efficiency ablation
#
# Step 1: Build per-source subsets (clustercov + random) at multiple budgets
# Step 2: Generate training configs from template
# Step 3: Train configs 2-at-a-time across 2 GPUs
# ============================================================================

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

MANIFEST="analysis/pooled_candidates_plus_pfpascal/manifest.jsonl"
OUTPUT_DIR="analysis/homogeneous_source_ablation"
CONFIG_OUTPUT_DIR="src/configs/CorrespondenceConfigs/homogeneous_ablation"
TEMPLATE_CONFIG="src/configs/CorrespondenceConfigs/pointodyssey_spair_pfpascal_pooled_multitarget_clustercov_k1024_norm_noshortlist_dedup_0p5pct_lr1e-4.yaml"

# Training settings
LR=5e-5
LR_BACKBONE=1e-6
EPOCHS=-1
STEP_MILESTONES="[]"
MAX_STEPS=20000
VALIDATION_INTERVAL=1000
CHECK_VAL_EVERY_N_EPOCH=1000000

# Scoring CSVs (all 5 benchmarks, existing)
SCORES=(
    "kitti2012:analysis/pooled_candidates_plus_pfpascal_vs_kitti2012_clustercov_k1024.csv"
    "kitti2015:analysis/pooled_candidates_plus_pfpascal_vs_kitti2015_clustercov_k1024.csv"
    "pfpascal:analysis/pooled_candidates_plus_pfpascal_vs_pfpascal_clustercov_k1024.csv"
    "pfwillow:analysis/pooled_candidates_plus_pfpascal_vs_pfwillow_clustercov_k1024.csv"
    "tss:analysis/pooled_candidates_plus_pfpascal_vs_tss_clustercov_k1024.csv"
)

# Budget fractions (relative to each source pool).
# Per-source fractions chosen so the smallest subset is still large enough to learn.
#   SPair   (53k pool): 2%, 5%, 10%, 25%, 50%  →  ~1k … 27k examples
#   PF-PASCAL (2.9k pool): 10%, 25%, 50%, 75%, 100%  →  ~294 … 2940 examples
FRACTIONS="0.02,0.05,0.10,0.25,0.50"  # default fallback
SOURCE_FRACTIONS=(
    "spair=0.02,0.05,0.10,0.25,0.50"
    "pfpascal=0.10,0.25,0.50,0.75,1.0"
)

# Sources to process for the semantic rerun.
SOURCES="spair,pfpascal"

# Minimum budget to actually run
MIN_BUDGET=50

# ── Step 1: Build subsets ──────────────────────────────────────────────────
echo "============================================"
echo "Step 1: Building per-source subsets"
echo "============================================"

python scripts/build_homogeneous_source_ablation_subsets.py \
    --manifest-path "$MANIFEST" \
    --scores "${SCORES[@]}" \
    --fractions "$FRACTIONS" \
    --source-fractions "${SOURCE_FRACTIONS[@]}" \
    --sources "$SOURCES" \
    --shortlist-count 0 \
    --alpha 1.0 \
    --normalize-by-n-valid \
    --min-budget "$MIN_BUDGET" \
    --seed 2021 \
    --output-dir "$OUTPUT_DIR"

# ── Step 2: Generate configs ───────────────────────────────────────────────
echo ""
echo "============================================"
echo "Step 2: Generating training configs"
echo "============================================"

python scripts/generate_homogeneous_configs.py \
    --summary "$OUTPUT_DIR/homogeneous_ablation_summary.json" \
    --template "$TEMPLATE_CONFIG" \
    --output-dir "$CONFIG_OUTPUT_DIR" \
    --sources "$SOURCES" \
    --lr "$LR" \
    --lr-backbone "$LR_BACKBONE" \
    --epochs "$EPOCHS" \
    --step-milestones "$STEP_MILESTONES" \
    --max-steps "$MAX_STEPS" \
    --validation-step-interval "$VALIDATION_INTERVAL" \
    --check-val-every-n-epoch "$CHECK_VAL_EVERY_N_EPOCH" \
    --disable-epoch-checkpoints

# ── Step 3: Train 2-at-a-time on 2 GPUs ───────────────────────────────────
echo ""
echo "============================================"
echo "Step 3: Training (2 GPUs, 2 runs at a time)"
echo "============================================"

# Collect all config paths into an array
CONFIGS=()
for cfg in "$CONFIG_OUTPUT_DIR"/homogeneous_*.yaml; do
    [ -f "$cfg" ] && CONFIGS+=("$cfg")
done

TOTAL=${#CONFIGS[@]}
echo "Found $TOTAL configs to train"
echo "Learning rate: $LR"
echo "Backbone learning rate: $LR_BACKBONE"
echo "Epochs per run: $EPOCHS"
echo "Step milestones: $STEP_MILESTONES"
echo "Max steps per run: $MAX_STEPS"
echo "Validation every N epochs: $CHECK_VAL_EVERY_N_EPOCH"
echo ""

DONE=0
IDX=0
while [ "$IDX" -lt "$TOTAL" ]; do
    # Launch on GPU 0
    CFG0="${CONFIGS[$IDX]}"
    echo "[$(date '+%H:%M:%S')] GPU 0: $(basename "$CFG0") ($((IDX+1))/$TOTAL)"
    CUDA_VISIBLE_DEVICES=0 python train_lightning.py --config "$CFG0" &
    PID0=$!

    # Launch on GPU 1 if there's another config
    PID1=""
    IDX1=$((IDX+1))
    if [ "$IDX1" -lt "$TOTAL" ]; then
        CFG1="${CONFIGS[$IDX1]}"
        echo "[$(date '+%H:%M:%S')] GPU 1: $(basename "$CFG1") ($((IDX1+1))/$TOTAL)"
        CUDA_VISIBLE_DEVICES=1 python train_lightning.py --config "$CFG1" &
        PID1=$!
    fi

    # Wait for both to finish
    wait "$PID0"
    EXIT0=$?
    DONE=$((DONE+1))
    if [ "$EXIT0" -ne 0 ]; then
        echo "WARNING: $(basename "$CFG0") exited with code $EXIT0"
    else
        echo "[$(date '+%H:%M:%S')] Done: $(basename "$CFG0")"
    fi

    if [ -n "$PID1" ]; then
        wait "$PID1"
        EXIT1=$?
        DONE=$((DONE+1))
        if [ "$EXIT1" -ne 0 ]; then
            echo "WARNING: $(basename "${CONFIGS[$IDX1]}") exited with code $EXIT1"
        else
            echo "[$(date '+%H:%M:%S')] Done: $(basename "${CONFIGS[$IDX1]}")"
        fi
        IDX=$((IDX+2))
    else
        IDX=$((IDX+1))
    fi

    echo "Progress: $DONE/$TOTAL complete"
    echo ""
done

echo "============================================"
echo "All $TOTAL runs complete."
echo "============================================"
