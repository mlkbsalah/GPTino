import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np

from gptino.config import TrainConfig, load_train_config, save_config, validate


def resolve_sweep_config(sweep_config:dict, seed: int):
    rng = np.random.default_rng(seed)
    for key, value in sweep_config.items():
        if value is None:
            raise ValueError(f"None for {key}")
        elif isinstance(value, (int, float, str)):
            sweep_config[key] = [value]
        elif isinstance(value, list):
            if len(value) == 0:
                raise ValueError(f"Empty list for {key}")
            sweep_config[key] = value
        else:
            if value["distribution"] == "uniform":
                min_val = value["min"]
                max_val = value["max"]
                repeats = value["repeats"]
                values = rng.uniform(min_val, max_val, repeats)
                sweep_config[key] = values.tolist()

            elif value["distribution"] == "log_uniform":
                min_val = value["min"]
                max_val = value["max"]
                repeats = value["repeats"]
                values = np.exp(rng.uniform(np.log(min_val), np.log(max_val), repeats))
                sweep_config[key] = values.tolist()

            else:
                raise ValueError(f"Unknown distribution: {value['distribution']}")

    return sweep_config


def create_trial_id(config: TrainConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def plan(config_path: Path, output_dir: Path, sweep_config: dict):
    trials = []
    base_config = load_train_config(config_path)
    for values in product(*sweep_config.values()):
        override_dict = dict(zip(sweep_config.keys(), values))
        config = deepcopy(base_config)
        config = replace(config, **override_dict)
        validate(config)

        trial_id =  create_trial_id(config)
        trials.append((trial_id, config, override_dict))


    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"trials":[]}

    entries = {entry["trial_id"]: entry for entry in manifest["trials"]}
    for trial_id, config, override_dict in trials:
        if trial_id not in entries:
            trial_dir = (output_dir / trial_id)
            trial_dir.mkdir(parents=True, exist_ok=False)
            config_path = trial_dir / "config.yaml"
            save_config(config, config_path)
            entry = {
                "trial_id": trial_id,
                "config_path": str(config_path.resolve()),
                "status": "planned",
                "jobid": None,
                "overrides": override_dict,
            }
            manifest["trials"].append(entry)
            entries[trial_id] = entry


    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    import yaml
    sweep_setup = yaml.safe_load(Path("config/sweep.yaml").read_text())

    output_dir = Path(sweep_setup["output_dir"])
    base_config_path = Path(sweep_setup["base_config"])
    sweep_config = sweep_setup["sweep_config"]
    seed = sweep_setup.get("seed")

    sweep_config = resolve_sweep_config(sweep_config, seed)
    plan(base_config_path, output_dir, sweep_config)
