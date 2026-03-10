import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path

from models.multimodal_clip import MultimodalCLIP
from datasets.label_maps import TOPIC2ID, SENT2ID, CTX2ID, TOPICS
from datasets.labelstudio_dataset import LabelStudioDataset, build_collate_fn
from training.train_epoch import train_epoch
from training.eval_epoch import eval_epoch
from training.early_stopping import EarlyStopping
from utils.seed import set_seed


def main():
    set_seed(42)

    DATA_PATH = "dataset.json"
    TEXT_MODEL_NAME = "xlm-roberta-base"
    IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"
    MAX_LENGTH = 128
    BATCH_SIZE = 8
    MAX_EPOCHS = 20

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    collate_fn = build_collate_fn(tokenizer)

    base_ds = LabelStudioDataset(
        ls_export_path=DATA_PATH,
        tokenizer=tokenizer,
        topics2id=TOPIC2ID,
        sent2id=SENT2ID,
        ctx2id=CTX2ID,
        max_length=MAX_LENGTH,
        skip_unlabeled=True,
        indices=None,
        image_model_name=IMAGE_MODEL_NAME,
    )

    n = len(base_ds)
    n_train = int(0.8 * n)
    perm = torch.randperm(n).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ds = LabelStudioDataset(
        ls_export_path=DATA_PATH,
        tokenizer=tokenizer,
        topics2id=TOPIC2ID,
        sent2id=SENT2ID,
        ctx2id=CTX2ID,
        max_length=MAX_LENGTH,
        skip_unlabeled=True,
        indices=train_idx,
        image_model_name=IMAGE_MODEL_NAME,
    )

    val_ds = LabelStudioDataset(
        ls_export_path=DATA_PATH,
        tokenizer=tokenizer,
        topics2id=TOPIC2ID,
        sent2id=SENT2ID,
        ctx2id=CTX2ID,
        max_length=MAX_LENGTH,
        skip_unlabeled=True,
        indices=val_idx,
        image_model_name=IMAGE_MODEL_NAME,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalCLIP(
        num_topics=len(TOPICS),
        num_sent=len(SENT2ID),
        num_ctx=len(CTX2ID),
        hidden_dim=512,
        text_model_name=TEXT_MODEL_NAME,
        image_model_name=IMAGE_MODEL_NAME,
    ).to(device)

    for p in model.text.parameters():
        p.requires_grad = False

    for p in model.image.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5,
        weight_decay=1e-4
    )

    PROJECT_ROOT = Path(__file__).resolve().parent
    save_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"

    stopper = EarlyStopping(
        patience=3,
        min_delta=1e-4,
        save_path=str(save_path),
        mode="min"
    )

    for epoch in range(MAX_EPOCHS):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        va_loss, va_metrics = eval_epoch(model, val_loader, device, topics_threshold=0.5)

        should_stop = stopper.step(va_loss, model)
        if should_stop:
            break

if __name__ == "__main__":
    main()