import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.inference_mode()
def evaluate_multiple_choice_questions(model, dataloader, device):
    model.to(device)
    model.eval()
    results = []
    acc_sum = 0
    accuracy = 0.0

    for i, (tokens, mask, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
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

        row = {
            "sample_idx": i,
            "prediction": prediction,
            "label": label,
            "loss": avg_loss.tolist(),
        }

        accuracy = acc_sum / (i + 1)

        results.append(row)

    return results, accuracy
