#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 phase0|phase1|phase2|phase3|phase4" >&2
  exit 1
fi

PHASE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/generate_phase_jobs.py" --mode smoke --phase "${PHASE}" >/tmp/variable_flow_smoke_job.yaml

python3 - <<'PY'
from pathlib import Path
import subprocess
import yaml

manifest = yaml.safe_load(Path("/tmp/variable_flow_smoke_job.yaml").read_text())
item = manifest["jobs"][0]
result = subprocess.run(["sbatch", item["job"]], capture_output=True, text=True, check=True)
print(f"{item['phase']}: {result.stdout.strip()} log={item['log']}")
PY
