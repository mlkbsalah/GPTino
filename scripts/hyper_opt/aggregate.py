import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def aggregate(sweep_path: Path) -> pd.DataFrame:
    trials = json.loads((sweep_path / "manifest.json").read_text())["trials"]
    rows = []
    for trial in trials:
        trial_id = trial["trial_id"]
        metrics_path = sweep_path / trial_id / "metrics.jsonl"
        metrics = pd.read_json(metrics_path, lines=True)
        best = metrics.loc[metrics["Validation Loss"].idxmin()]
        final_val = metrics.iloc[-1]["Validation Loss"]
        rows.append((trial_id, best["Validation Loss"], best["Training Loss"], final_val))

    table = pd.DataFrame(rows, columns=["trial_id", "best_val_loss", "train_loss_at_best", "final_val_loss"])
    rank = table["best_val_loss"].rank(method="min").astype("Int64")
    table.insert(0, "rank", rank)
    return table.sort_values(["rank", "trial_id"], ignore_index=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("sweep_path", type=Path)
    args = parser.parse_args()
    print(aggregate(args.sweep_path).to_string(index=False))
