import torch
from .losses import compute_loss

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        topics_logits, sent_logits, ctx_logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["pixel_values"]
        )

        loss = compute_loss(topics_logits, sent_logits, ctx_logits, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        n += 1

    return total_loss / max(n, 1)