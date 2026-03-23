#!/usr/bin/env bash
set -euo pipefail

# Run all FPS descriptor variant training runs sequentially.
# Mirrors run_smoke_pointodyssey_5pct_local.sh — same env vars, same logic.
# Usage:
#   bash scripts/run_fps_smoke_overnight.sh
# Optional env:
#   PYTHON=python3
#   CUDA_DEVICES=0,1
#   LOG_DIR=./logs
#   CONTINUE_ON_ERROR=1
#   SLEEP_BETWEEN_RUNS=30
#   AUTO_RESUME=1
#   SKIP_COMPLETED=1
#   SNAPSHOTS_DIR=./snapshots

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-30}"
AUTO_RESUME="${AUTO_RESUME:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-$REPO_ROOT/snapshots}"

CONFIG_DIR="$REPO_ROOT/src/configs/CorrespondenceConfigs"
CONFIGS=(
  "pointodyssey_smoke_fps_bfv_5pct.yaml"
  "pointodyssey_smoke_fps_mean_mag_5pct.yaml"
  "pointodyssey_smoke_fps_median_mag_5pct.yaml"
  "pointodyssey_smoke_fps_p90_mag_5pct.yaml"
)

mkdir -p "$LOG_DIR"
mkdir -p "$SNAPSHOTS_DIR"

cd "$REPO_ROOT"
shopt -s nullglob

latest_snapshot_for_run() {
  local run_name="$1"
  local matches=("$SNAPSHOTS_DIR/${run_name}_"*)
  local latest=""
  local p
  for p in "${matches[@]}"; do
    [[ -d "$p" ]] || continue
    if [[ -z "$latest" || "$p" -nt "$latest" ]]; then latest="$p"; fi
  done
  echo "$latest"
}

latest_ckpt_in_snapshot() {
  local snap_dir="$1"
  local matches=("$snap_dir"/checkpoints/*.ckpt)
  local latest=""
  local p
  for p in "${matches[@]}"; do
    [[ -f "$p" ]] || continue
    if [[ -z "$latest" || "$p" -nt "$latest" ]]; then latest="$p"; fi
  done
  echo "$latest"
}

extract_planned_epochs() {
  local cfg_path="$1"
  "$PYTHON" - "$cfg_path" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
epochs = cfg.get("training", {}).get("epochs", None)
try:
    print(int(epochs))
except Exception:
    print(0)
PY
}

snapshot_is_completed() {
  local snap_dir="$1"
  local planned_epochs="$2"
  local summary_file="$snap_dir/training_summary.txt"

  if [[ -f "$summary_file" ]] && grep -q "STATUS: Training completed successfully" "$summary_file"; then
    return 0
  fi
  if [[ "$planned_epochs" -gt 0 && -f "$snap_dir/epoch_${planned_epochs}.pth" ]]; then
    return 0
  fi
  return 1
}

total_runs="${#CONFIGS[@]}"
run_idx=0
for cfg in "${CONFIGS[@]}"; do
  run_idx=$((run_idx + 1))
  cfg_path="$CONFIG_DIR/$cfg"
  if [[ ! -f "$cfg_path" ]]; then
    echo "Missing config: $cfg_path" >&2
    exit 1
  fi

  run_name="${cfg%.yaml}"
  planned_epochs="$(extract_planned_epochs "$cfg_path")"
  ts="$(date +%Y%m%d_%H%M%S)"
  log_path="$LOG_DIR/${run_name}_local_${ts}.log"
  latest_snapshot="$(latest_snapshot_for_run "$run_name")"
  resume_args=()

  if [[ -n "$latest_snapshot" ]]; then
    if [[ "$SKIP_COMPLETED" == "1" ]]; then
      if snapshot_is_completed "$latest_snapshot" "$planned_epochs"; then
        echo "===== Skipping $cfg (already completed) ====="
        echo "Found completed snapshot: $latest_snapshot"
        continue
      fi
    fi
    if [[ "$AUTO_RESUME" == "1" ]]; then
      latest_ckpt="$(latest_ckpt_in_snapshot "$latest_snapshot")"
      if [[ -n "$latest_ckpt" ]]; then
        resume_args=(--resume-ckpt "$latest_ckpt")
      fi
    fi
  fi

  echo "===== [$run_idx/$total_runs] Running $cfg ====="
  echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
  echo "Log: $log_path"
  if [[ "${#resume_args[@]}" -gt 0 ]]; then
    echo "Auto-resume checkpoint: ${resume_args[1]}"
  fi

  if ! CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON" -u train_lightning.py --config "$cfg_path" "${resume_args[@]}" 2>&1 | tee "$log_path"; then
    echo "Run failed: $cfg"
    if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
      echo "CONTINUE_ON_ERROR=1, continuing..."
      if [[ "$run_idx" -lt "$total_runs" && "$SLEEP_BETWEEN_RUNS" -gt 0 ]]; then
        sleep "$SLEEP_BETWEEN_RUNS"
      fi
      continue
    fi
    exit 1
  fi

  if [[ "$run_idx" -lt "$total_runs" && "$SLEEP_BETWEEN_RUNS" -gt 0 ]]; then
    echo "Sleeping ${SLEEP_BETWEEN_RUNS}s before next run..."
    sleep "$SLEEP_BETWEEN_RUNS"
  fi
done

echo "All FPS runs completed."
