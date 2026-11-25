import itertools
from dataclasses import dataclass
from pathlib import Path
from tkinter import Entry

import yaml

from ..config import TrainConfig, load_train_config, save_config, validate
from .config import hash_resolved_config, load_sweep_config


@dataclass
class Trial:
    trial_id: str
    config: TrainConfig


class Planner:
    def __init__(self, sweep_config_path: Path, study_path: Path):
        self.sweep_config = load_sweep_config(sweep_config_path)
        self.manifest = {"trials": []}
        self.trials = []

    def plan(self):
        trials = []
        grid = self.sweep_config.grid

        for values in itertools.product(*grid.values()):
            settings = {
                **self.sweep_config.fixed,
                **dict(zip(grid, values))
            }
            overrides = [
                f"{path}={value if value is not None else 'none'}"
                for path, value in settings.items()
            ]

            config = load_train_config(
                self.sweep_config.base_config,
                overrides=overrides
            )
            validate(config)

            trial_id = hash_resolved_config(config)
            trials.append(Trial(trial_id, config))

        self.trials = trials
        return trials

    def save_plan(self, output_dir: Path | str):
        assert self.trials, "No trials to spawn, run plan() first"

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "study_manifest.yaml"

        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = yaml.safe_load(f)
        else:
            manifest = {"trials": []}

        manifest_entries = {
            entry["trial_id"]: entry
            for entry in manifest["trials"]
        }

        for trial in self.trials:
            # This is from the created trials not the saved one
            trial_dir = output_dir / trial.trial_id
            config_path = trial_dir / "config.yaml"


            if config_path.exists(): # checking that config_path exists includes checking that trial dir exists
                existing_config = load_train_config(config_path)
                if existing_config != trial.config:
                    raise ValueError(f"Config mismatch for trial {trial.trial_id}")
            else:
                if trial.trial_id in manifest_entries:
                    raise ValueError("Trial recorded without config")

                trial_dir.mkdir(parents=True, exist_ok=True)
                save_config(trial.config, config_path)

            if trial.trial_id not in manifest_entries:
                manifest_entries[trial.trial_id] = {
                    "trial_id": trial.trial_id,
                    "status": "planned",
                    "config_path": str(config_path),
                    "job_id": None
                }
        manifest["trials"] = list(manifest_entries.values())

        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=False)
        )
