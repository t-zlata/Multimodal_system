import torch

@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()

@torch.no_grad()
def multilabel_micro_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).int()
    t = targets.int()

    tp = (preds & t).sum().item()
    fp = (preds & (1 - t)).sum().item()
    fn = ((1 - preds) & t).sum().item()

    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0