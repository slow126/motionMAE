#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SLURM_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PACKAGE_ROOT / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_job(
    *,
    job_path: Path,
    job_name: str,
    log_path: Path,
    machine_cfg: dict[str, Any],
    config_path: Path,
    walltime: str,
) -> None:
    slurm_cfg = machine_cfg["slurm"]
    machine = machine_cfg["machine"]
    project_root = Path(machine["project_root"])
    account = str(slurm_cfg.get("account", "")).strip()
    qos = str(slurm_cfg.get("qos", "")).strip()
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_path}",
        f"#SBATCH --error={log_path}",
        f"#SBATCH --partition={slurm_cfg['partition']}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --nodes={slurm_cfg.get('nodes', 1)}",
        f"#SBATCH --ntasks={slurm_cfg.get('ntasks', 1)}",
        f"#SBATCH --cpus-per-task={slurm_cfg.get('cpus_per_task', 16)}",
        f"#SBATCH --gres=gpu:{slurm_cfg.get('gpus', 1)}",
        f"#SBATCH --mem={slurm_cfg.get('mem', '64g')}",
    ]
    if account:
        lines.append(f"#SBATCH --account={account}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {project_root}",
            f"source ~/.bashrc || true",
            f"conda activate {machine['conda_env']}",
            (
                "srun "
                f"{machine.get('python_path', 'python3')} -u "
                f"{project_root / 'scripts' / 'train_variable_flow_perceiver.py'} "
                f"--config {project_root / 'percieverIO_Exp' / 'configs' / config_path.name}"
            ),
            "",
        ]
    )
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text("\n".join(lines), encoding="utf-8")
    job_path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate phase-specific Slurm jobs for variable flow experiment.")
    parser.add_argument("--machine-config", default=str(SLURM_ROOT / "machine_configs" / "rc.yaml"))
    parser.add_argument("--mode", choices=["full", "smoke"], required=True)
    parser.add_argument("--phase", choices=[f"phase{i}" for i in range(5)], default=None)
    args = parser.parse_args()

    machine_cfg = load_yaml(Path(args.machine_config).resolve())
    phases = [args.phase] if args.phase else [f"phase{i}" for i in range(5)]
    snapshots_root = Path(machine_cfg["paths"]["snapshots_root"])
    logs_root = snapshots_root / "_slurm"
    jobs_root = SLURM_ROOT / "jobs" / args.mode
    walltime = str(machine_cfg["slurm"]["time_full" if args.mode == "full" else "time_smoke"])

    manifest: list[dict[str, str]] = []
    for phase in phases:
        config_name = f"{phase}_{'rc' if args.mode == 'full' else 'smoke'}.yaml"
        config_path = CONFIG_ROOT / config_name
        job_name = f"vf_{phase}_{args.mode}"
        log_path = logs_root / args.mode / f"{job_name}-%j.log"
        job_path = jobs_root / f"{job_name}.sbatch"
        write_job(
            job_path=job_path,
            job_name=job_name,
            log_path=log_path,
            machine_cfg=machine_cfg,
            config_path=config_path,
            walltime=walltime,
        )
        manifest.append({"phase": phase, "job": str(job_path), "log": str(log_path)})

    print(yaml.safe_dump({"mode": args.mode, "jobs": manifest}, sort_keys=False))


if __name__ == "__main__":
    main()
