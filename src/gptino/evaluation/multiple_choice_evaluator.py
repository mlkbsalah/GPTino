import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MultipleChoiceQuestionEvaluator:
    def __init__(
        self,
        dataloaders: dict[str, DataLoader],
        ddp_context=None,
    ):
        self.dataloaders = dataloaders
        self.ddp_context = ddp_context

    def evaluate(
        self,
        model: torch.nn.Module,
        device: torch.device | str,
    ) -> dict[str, float]:
        results = {}
        for name, dataloader in self.dataloaders.items():
            results[name] = self.evaluate_on_dataloader(model, device, dataloader)

        return results

    @torch.inference_mode()
    def evaluate_on_dataloader(
        self,
        model: torch.nn.Module,
        device: torch.device | str,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        model.to(device)
        model.eval()
        acc_sum = 0
        sample_count = 0
        distributed = self.ddp_context is not None and self.ddp_context.enabled

        for tokens, mask, label in dataloader:
            tokens = tokens.to(device)
            mask = mask.to(device)

            logits = model(tokens)  # [n_labels, sequence_length, vocab_size]
            logits = logits[
                :, :-1, :
            ].contiguous()  # we don't have next token for last token
            tokens = tokens[
                :, 1:
            ].contiguous()  # we don't have previous token for first token

            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), tokens.view(-1), reduction="none"
            )
            loss = loss.view(tokens.shape[0], -1) * mask[:, 1:]
            avg_loss = loss.sum(dim=1) / mask[:, 1:].sum(dim=1)

            prediction = avg_loss.argmin().item()
            acc_sum += int(prediction == label)
            sample_count += 1

        if distributed:
            counts = torch.tensor(
                [acc_sum, sample_count], device=device, dtype=torch.long
            )
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            acc_sum, sample_count = counts.tolist()
        accuracy = acc_sum / sample_count if sample_count else 0.0

        return {"accuracy": accuracy}
