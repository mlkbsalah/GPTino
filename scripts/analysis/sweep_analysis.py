import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSSES = ("Training Loss", "Validation Loss")


def history(run, batch):
    series = defaultdict(dict)
    for filename in ("metrics.jsonl", "task_eval.jsonl"):
        path = run / filename
        if filename == "task_eval.jsonl" and not path.exists():
            continue
        with path.open() as log:
            for line in log:
                row = json.loads(line)
                tokens = row["Step"] * batch
                for key, value in row.items():
                    if key in LOSSES or key.endswith("/accuracy"):
                        series[key][tokens] = value
    return {key: sorted(points.items()) for key, points in series.items()}


@dataclass
class Run:
    name: str
    label: str
    value: float
    batch: int
    budget: int
    curves: dict


def load_run(folder, budget):
    label = folder.name.split("_")[-1]
    value = float(label.split("-", 1)[1])
    config = yaml.safe_load((folder / "config.yaml").read_text())
    batch = config["training"]["train_batch_tokens"]
    return Run(folder.name, label, value, batch, budget, history(folder, batch))


def load_sweep(sweep):
    rounds = {}
    runs = []
    unit = 1_000_000_000
    for folder in sorted(sweep.glob("round*-*")):
        suffix = folder.name.split("-")[-1]
        budget = int(float(suffix[:-1]) * unit)
        rounds[budget] = suffix
        for log in sorted(folder.glob("*/metrics.jsonl")):
            runs.append(load_run(log.parent, budget))
    if not runs:
        raise SystemExit(f"No run logs found at {sweep.resolve()}/round*-*/*/metrics.jsonl. Pass the directory containing the rounds.")
    return rounds, sorted(runs, key=lambda run: run.budget)


def runs_at_budget(runs, budget):
    selected = {}
    for run in runs:
        # Training can only reach the last whole batch before the cutoff.
        cutoff = budget // run.batch * run.batch
        train = run.curves.get("Training Loss", [])
        if run.budget >= budget and train and train[-1][0] >= cutoff:
            # Runs are ordered by budget: keep the nearest archive of each run.
            selected.setdefault(run.name, run)
    return sorted(selected.values(), key=lambda run: run.value)


def plot_metric(ax, run, metric, budget, color):
    """Plot the history through the cutoff and return its last point for the table."""
    points = [(t, v) for t, v in run.curves.get(metric, []) if t <= budget]
    if not points:
        return None
    tokens, values = zip(*points)
    marker = None if metric == "Training Loss" and len(points) > 1 else "."
    ax.plot([t / 1e9 for t in tokens], values, color=color, marker=marker, label=run.label)
    return points[-1]


def budget_report(sweep_name, budget, suffix, runs, colors, output_dir):
    report = [f"## {suffix} tokens", ""]
    if not runs:
        return report + ["No run reached this budget.", ""]

    tasks = sorted({key for run in runs for key in run.curves if key.endswith("/accuracy")})
    metrics = [*LOSSES, *tasks]
    image = f"budget-{suffix}.png"
    report.extend([f"![Curves at {suffix} tokens]({image})", ""])
    report.append("| Run | " + " | ".join(metrics) + " |")
    report.append("|" + "---|" * (len(metrics) + 1))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 2.8 * len(metrics)), sharex=True, layout="constrained")
    for run in runs:
        cells = [run.name]
        for ax, metric in zip(axes, metrics):
            ax.set_ylabel(metric)
            last = plot_metric(ax, run, metric, budget, colors[run.value])
            if last is None:
                cells.append("—")
            else:
                tokens, value = last
                cells.append(f"{value:.4f} ({tokens:,})")
        report.append("| " + " | ".join(cells) + " |")

    axes[0].set_title(f"{sweep_name}: {suffix} tokens")
    axes[0].legend()
    axes[-1].set_xlabel("Tokens processed (billions)")
    fig.savefig(output_dir / image, dpi=150)
    plt.close(fig)
    return report + [""]


def analyze(sweep):
    rounds, runs = load_sweep(sweep)
    output_dir = Path(f"outputs/sweep_analysis/{sweep.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    values = sorted({run.value for run in runs})
    colors = {v: plt.cm.viridis(i / len(values)) for i, v in enumerate(values)}
    report = [f"# {sweep.name}", "", "Cells show the last measurement at or before the budget (tokens in parentheses).", ""]
    for budget, suffix in sorted(rounds.items()):
        selected = runs_at_budget(runs, budget)
        report.extend(budget_report(sweep.name, budget, suffix, selected, colors, output_dir))
    path = output_dir / "report.md"
    path.write_text("\n".join(report))
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare sweep runs at each round's token budget.")
    parser.add_argument("sweep", type=Path)
    args = parser.parse_args()
    analyze(args.sweep)
