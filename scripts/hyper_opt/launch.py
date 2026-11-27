import json
import os
from pathlib import Path
from subprocess import run


def launch(sweep_path: Path):
    manifest_path = (sweep_path / "manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    entries = manifest["trials"]

    for entry in entries:
        if entry["status"] == "planned":
            run_dir = sweep_path / entry["trial_id"]
            (run_dir / ".slurm").mkdir(parents=True, exist_ok=True)
            command = [
                "sbatch",
                "--parsable",
                "--output", str(run_dir / ".slurm" / "slurm-%j.out"),
                "slurm/train_multigpu.sh",
                "--config", entry["config_path"],
                "--run_dir", str(run_dir),
            ]

            env = os.environ.copy()
            env["RUN_DIR"] = str(run_dir)
            result = run(
                command,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            entry["status"] = "submitted"
            entry["jobid"] = result.stdout.strip()
            manifest_path.write_text(json.dumps(manifest))

            print(result.stdout)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("sweep_path", type=Path)
    args = parser.parse_args()
    launch(args.sweep_path)
