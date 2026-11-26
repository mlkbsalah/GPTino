from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

model_dict = {
    "swiglu": Path("runs/architecture-experiments/2026-11-26_01-24-26_swiglu"),
    "rmsnorm": Path("/home/m-ben-salah/repos/GPTino/runs/architecture-experiments/2026-11-26_01-43-41_rmsnorm"),
    "rmsnorm+swiglu": Path("runs/architecture-experiments/2026-11-26_01-44-39_swiglu+rmsnorm"),
    "base": Path("runs/gpt2-124M-adamw10b/tuned-model"),
}


fig, ax = plt.subplots(1, 4, figsize=(12, 6))
for architecture, dir in model_dict.items():
    train_metrics = pd.read_json(dir / "metrics.jsonl", lines=True)
    task_metrics = pd.read_json(dir / "task_eval.jsonl", lines=True)

    validation_loss = train_metrics.loc[~train_metrics["Validation Loss"].isna(), ["Step", "Validation Loss"]]
    ax[0].plot(validation_loss["Step"], validation_loss["Validation Loss"], label=architecture, alpha=0.5)
    ax[0].set_title("Validation Loss")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(task_metrics["Step"], task_metrics["hellaswag/accuracy"], label=architecture, alpha=0.5)
    ax[1].set_title("HellaSwag Accuracy")
    ax[1].grid(True)
    ax[1].legend()

    ax[2].plot(task_metrics["Step"], task_metrics["piqa/accuracy"], label=architecture, alpha=0.5)
    ax[2].set_title("PIQA Accuracy")
    ax[2].grid(True)
    ax[2].legend()

    ax[3].plot(train_metrics["Step"], train_metrics["Training Loss"], label=architecture, alpha=0.1)
    ax[3].set_title("Train Loss")
    ax[3].grid(True)
    ax[3].legend()

plt.tight_layout()
plt.savefig("swiglu_rmsnorm_analysis.png")
