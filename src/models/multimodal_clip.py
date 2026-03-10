import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel


class MultimodalCLIP(nn.Module):
    def __init__(
        self,
        num_topics: int,
        num_sent: int = 3,
        num_ctx: int = 4,
        hidden_dim: int = 512,
        text_model_name: str = "xlm-roberta-base",
        image_model_name: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__()

        self.text = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text.config.hidden_size

        self.image = CLIPVisionModel.from_pretrained(image_model_name)
        image_dim = self.image.config.hidden_size

        fusion_dim = text_dim + image_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.topic_head = nn.Linear(hidden_dim, num_topics)
        self.sent_head = nn.Linear(hidden_dim, num_sent)
        self.ctx_head = nn.Linear(hidden_dim, num_ctx)

    def forward(self, input_ids, attention_mask, pixel_values):

        text_outputs = self.text(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_feat = text_outputs.last_hidden_state[:, 0, :]

        image_outputs = self.image(pixel_values=pixel_values)
        image_feat = image_outputs.pooler_output

        fused = torch.cat([text_feat, image_feat], dim=-1)
        h = self.fusion(fused)

        topics_logits = self.topic_head(h)
        sent_logits = self.sent_head(h)
        ctx_logits = self.ctx_head(h)

        return topics_logits, sent_logits, ctx_logits