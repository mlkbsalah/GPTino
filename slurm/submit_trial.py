import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import yaml


def submit_trial(command, env):
    result = subprocess.run(
        command,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = yaml.safe_load(manifest_path.read_text())

    for trial in manifest["trials"]:
        if trial["status"] != "planned":
            continue

        config_path = Path(trial["config_path"])
        trial_dir = config_path.parent
        log_dir = trial_dir / ".slurm"
        log_dir.mkdir(exist_ok=True)

        env = os.environ.copy()
        env["RUN_DIR"] = str(trial_dir)
        env["CHECKPOINT"] = ""

        command = [
            "sbatch",
            "--parsable",
            "--export=ALL",
            f"--output={log_dir}/slurm-%j.out",
            "slurm/train_multigpu.sh",
            "--config",
            str(config_path),
            "--run_dir",
            str(trial_dir),
        ]

        job_id = submit_trial(command, env)

        trial["job_id"] = job_id
        trial["status"] = "submitted"
        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=False)
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Submit planned trials to Slurm")
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the study manifest YAML file",
    )
    args = parser.parse_args()
    main(args.manifest)
