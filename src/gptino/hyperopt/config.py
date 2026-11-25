import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class Objective:
    metric: str
    goal: Literal["minimize", "maximize"]

@dataclass
class SweepConfig:
    name: str
    base_config: str
    fixed: dict
    grid: dict
    objective: Objective


def load_sweep_config(sweep_config_path: Path) -> SweepConfig:
    with open(sweep_config_path) as f:
        sweep_dict = yaml.safe_load(f)

    objective_dict = sweep_dict["objective"]
    name = sweep_dict["name"]
    base_config = sweep_dict["base_config"]
    fixed = sweep_dict.get("fixed", {})
    grid = sweep_dict["grid"]

    objective = Objective(**objective_dict)
    config = SweepConfig(
        name=name,
        base_config=base_config,
        fixed=fixed,
        grid=grid,
        objective=objective)
    return config



def hash_resolved_config(config):
    text_dict = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(text_dict.encode("utf-8")).hexdigest()
