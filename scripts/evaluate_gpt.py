from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gptino.data.hellaswag import HellaSwagDataset
from gptino.data.piqa import PIQADataset
from gptino.evaluation.multiple_choice_evaluator import MultipleChoiceQuestionEvaluator
from gptino.module.gpt import GPT

DATASET_MAP = {
    "hellaswag": HellaSwagDataset,
    "piqa": PIQADataset,
}

DATASET_PATH_MAP = {
    "hellaswag": Path("data/hellaswag/validation-00000-of-00001.parquet"),
    "piqa": Path("data/piqa/validation-00000-of-00001.parquet"),
}


def main(checkpoint_path: Path, dataset_names, device):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_config = checkpoint["model_config"]
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    datasets = [
        DATASET_MAP[dataset_name](DATASET_PATH_MAP[dataset_name])
        for dataset_name in dataset_names
    ]
    dataloaders = [DataLoader(dataset, batch_size=None) for dataset in datasets]
    evaluator = MultipleChoiceQuestionEvaluator(
        {
            dataset_name: dataloader
            for dataset_name, dataloader in zip(dataset_names, dataloaders)
        }
    )
    results = evaluator.evaluate(model, device)

    for dataset_name, metrics in results.items():
        print(f"{dataset_name}: {metrics['accuracy']:.2f}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--dataset", nargs="+", default=["hellaswag"], choices=DATASET_MAP.keys()
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args.checkpoint_path, args.dataset, args.device)
