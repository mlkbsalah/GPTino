import json
import os
from pathlib import Path
from subprocess import run


def launch(sweep_path: Path, *, resume_ids=None, token_budget=None):
    manifest_path = (sweep_path / "manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    entries = manifest["trials"]


    if resume_ids is None:
        selected = [entry for entry in entries if entry["status"] == "planned"]
    else:
        if token_budget is None:
            raise ValueError("--resume requires a positive --token-budget")
        entry_dict = {entry["trial_id"]: entry for entry in entries}
        selected = [entry_dict[trial_id] for trial_id in set(resume_ids)]
        for entry in selected:
            run_dir = sweep_path / entry["trial_id"]
            meta = json.loads((run_dir / "metadata" / "meta.json").read_text())
            jobid = str(entry["jobid"])
            if meta["slurm_job_id"] != jobid or meta["status"] != "completed":
                raise ValueError(f"Trial {entry['trial_id']}: job {jobid} has not completed")
            checkpoint = run_dir / "checkpoints" / "latest.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)

    for entry in selected:
        run_dir = sweep_path / entry["trial_id"]
        (run_dir / ".slurm").mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["RUN_DIR"] = str(run_dir)

        if resume_ids is None:
            train_args = ["--config", entry["config_path"], "--run_dir", str(run_dir)]
        else:
            checkpoint = str(run_dir / "checkpoints" / "latest.pt")
            env["CHECKPOINT"] = checkpoint
            train_args = ["--resume", checkpoint]

        if token_budget is not None:
            train_args.extend(["--set", f"training.run_token_budget={token_budget}"])
        command = [
            "sbatch", "--parsable",
            "--output", str(run_dir / ".slurm" / "slurm-%j.out"),
            "slurm/train_multigpu.sh", *train_args,
        ]
        result = run(command, env=env, check=True, capture_output=True, text=True)
        entry["status"] = "submitted"
        entry["jobid"] = result.stdout.strip()

        if resume_ids is not None:
            entry.setdefault("resumes", []).append({"jobid": entry["jobid"], "token_budget": token_budget})
        manifest_path.write_text(json.dumps(manifest, indent=4))
        print(result.stdout)



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("sweep_path", type=Path)
    parser.add_argument("--resume", nargs="+", metavar="TRIAL_ID", help="Resume selected completed trials")
    parser.add_argument("--token-budget", type=int, help="Total training token budget per trial (required with --resume)")
    args = parser.parse_args()
    launch(args.sweep_path, resume_ids=args.resume, token_budget=args.token_budget)
