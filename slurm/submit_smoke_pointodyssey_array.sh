#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash slurm/submit_smoke_pointodyssey_array.sh
#   bash slurm/submit_smoke_pointodyssey_array.sh _rc   # use remote-root smoke configs

SUFFIX="${1:-}"

export SMOKE_CFG_SUFFIX="$SUFFIX"
sbatch --export=SMOKE_CFG_SUFFIX "$PWD/slurm/run_smoke_pointodyssey_array.sbatch"
