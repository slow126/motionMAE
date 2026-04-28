#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python3 "${SCRIPT_DIR}/generate_phase_jobs.py" --mode full >/tmp/variable_flow_full_jobs.yaml

python3 - <<'PY'
from pathlib import Path
import subprocess
import yaml

manifest = yaml.safe_load(Path("/tmp/variable_flow_full_jobs.yaml").read_text())
for item in manifest["jobs"]:
    result = subprocess.run(["sbatch", item["job"]], capture_output=True, text=True, check=True)
    print(f"{item['phase']}: {result.stdout.strip()} log={item['log']}")
PY
