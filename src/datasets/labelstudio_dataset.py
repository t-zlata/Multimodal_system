import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from urllib.parse import urlparse, parse_qs

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor

class LabelStudioDataset(Dataset):
    def __init__(
        self,
        ls_export_path: str,
        tokenizer,
        topics2id=None,
        sent2id=None,
        ctx2id=None,
        max_length: int = 128,
        skip_unlabeled: bool = True,
        indices=None,
        image_model_name: str = "openai/clip-vit-base-patch32",
    ):
        self.tokenizer = tokenizer
        self.image_processor = AutoImageProcessor.from_pretrained(image_model_name)
        self.topics2id = topics2id
        self.sent2id = sent2id
        self.ctx2id = ctx2id
        self.max_length = max_length

        with open(ls_export_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.base_dir = Path(ls_export_path).resolve().parent
        filtered: List[Dict[str, Any]] = []

        for it in raw:
            data = it.get("data") or {}
            text = (data.get("text") or "").strip()
            image = data.get("image")
            labels = self._parse_labels(it)

            if not text or not image:
                continue
            if skip_unlabeled and labels is None:
                continue

            filtered.append(
                {
                    "text": text,
                    "image": image,
                    "labels": labels,
                }
            )

        if indices is not None:
            self.items = [filtered[i] for i in indices]
        else:
            self.items = filtered

    def _resolve_image_path(self, image_value: str) -> Path:
        if image_value.startswith("/data/local-files/"):
            parsed = urlparse(image_value)
            qs = parse_qs(parsed.query)
            rel = qs.get("d", [None])[0]
            if rel:
                return (self.base_dir / rel).resolve()

        p = Path(image_value)
        if p.is_absolute():
            return p
        return (self.base_dir / p).resolve()

    def _parse_labels(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        anns = item.get("annotations") or []
        if not anns:
            return None

        result = anns[0].get("result") or []

        topics = []
        sentiment = None
        context_type = None

        for r in result:
            from_name = r.get("from_name")
            value = r.get("value") or {}
            choices = value.get("choices") or []

            if from_name == "topics":
                topics = choices
            elif from_name == "sentiment":
                sentiment = choices[0] if choices else None
            elif from_name == "context_type":
                context_type = choices[0] if choices else None

        if sentiment is None or context_type is None:
            return None

        topic_vec = torch.zeros(len(self.topics2id), dtype=torch.float32)
        for t in topics:
            if t in self.topics2id:
                topic_vec[self.topics2id[t]] = 1.0

        sent_id = self.sent2id.get(sentiment, -1)
        ctx_id = self.ctx2id.get(context_type, -1)

        if sent_id == -1 or ctx_id == -1:
            return None

        return {
            "topic_vec": topic_vec,
            "sent_id": sent_id,
            "ctx_id": ctx_id,
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]

        text = item["text"]
        image_value = item["image"]
        labels = item["labels"]

        if labels is None:
            raise ValueError("Encountered unlabeled item. Use skip_unlabeled=True for training.")

        tok = self.tokenizer(
            text,
            return_attention_mask=True,
            return_tensors=None,
            truncation=True,
            max_length=self.max_length,
        )

        img_path = self._resolve_image_path(image_value)
        img = Image.open(img_path).convert("RGB")

        pixel_values = self.image_processor(
            images=img,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        return {
            "input_ids": torch.tensor(tok["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tok["attention_mask"], dtype=torch.long),
            "pixel_values": pixel_values,
            "topic_vec": labels["topic_vec"],
            "sent_id": torch.tensor(labels["sent_id"], dtype=torch.long),
            "ctx_id": torch.tensor(labels["ctx_id"], dtype=torch.long),
        }


def build_collate_fn(tokenizer):
    def collate_fn(batch):
        token_features = [
            {
                "input_ids": b["input_ids"],
                "attention_mask": b["attention_mask"],
            }
            for b in batch
        ]

        padded = tokenizer.pad(
            token_features,
            padding=True,
            return_tensors="pt"
        )

        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        topic_vec = torch.stack([b["topic_vec"] for b in batch], dim=0)
        sent_id = torch.stack([b["sent_id"] for b in batch], dim=0)
        ctx_id = torch.stack([b["ctx_id"] for b in batch], dim=0)

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "pixel_values": pixel_values,
            "topic_vec": topic_vec,
            "sent_id": sent_id,
            "ctx_id": ctx_id,
        }

    return collate_fn