import torch.nn as nn

bce = nn.BCEWithLogitsLoss()
ce = nn.CrossEntropyLoss()

def compute_loss(topics_logits, sent_logits, ctx_logits, batch, w_topics=1.0, w_sent=1.0, w_ctx=1.0):
    loss_topics = bce(topics_logits, batch["topic_vec"])
    loss_sent = ce(sent_logits, batch["sent_id"])
    loss_ctx = ce(ctx_logits, batch["ctx_id"])
    return w_topics * loss_topics + w_sent * loss_sent + w_ctx * loss_ctx