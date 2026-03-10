import torch
from .losses import compute_loss
from .metrics import accuracy, multilabel_micro_f1

@torch.no_grad()
def eval_epoch(model, loader, device, topics_threshold: float = 0.5):
    model.eval()
    total_loss = 0.0
    n = 0

    sent_acc_sum = 0.0
    ctx_acc_sum = 0.0
    topics_f1_sum = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        topics_logits, sent_logits, ctx_logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["pixel_values"]
        )

        loss = compute_loss(topics_logits, sent_logits, ctx_logits, batch)

        total_loss += loss.detach().item()
        n += 1

        sent_acc_sum += accuracy(sent_logits, batch["sent_id"])
        ctx_acc_sum += accuracy(ctx_logits, batch["ctx_id"])
        topics_f1_sum += multilabel_micro_f1(topics_logits, batch["topic_vec"], threshold=topics_threshold)

    if n == 0:
        return 0.0, {"sent_acc": 0.0, "ctx_acc": 0.0, "topics_micro_f1": 0.0}

    metrics = {
        "sent_acc": sent_acc_sum / n,
        "ctx_acc": ctx_acc_sum / n,
        "topics_micro_f1": topics_f1_sum / n,
    }
    return total_loss / n, metrics