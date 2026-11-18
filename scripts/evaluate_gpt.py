import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gptino.evaluation.hellaswag import HellaSwagDataset
from gptino.evaluation.multiple_choice import evaluate_multiple_choice_questions
from gptino.module.gpt import GPT


def main(checkpoint_path: Path, dataset_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_config = checkpoint["model_config"]
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = HellaSwagDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=None)
    results, accuracy = evaluate_multiple_choice_questions(model, dataloader, device)

    with open(checkpoint_path.parent.parent / "hellaswag_results.jsonl", "w") as f:
        f.writelines(json.dumps(result) + "\n" for result in results)

    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    HELLA_SWAG_PATH = Path("data/hellaswag/validation-00000-of-00001.parquet")

    parser = ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--dataset-path", default=HELLA_SWAG_PATH, type=Path)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args.checkpoint_path, args.dataset_path, args.device)
