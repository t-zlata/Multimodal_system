import torch
from models.multimodal_clip import MultimodalCLIP
from datasets.label_maps import TOPICS, SENT2ID, CTX2ID

TEXT_MODEL_NAME = "xlm-roberta-base"
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultimodalCLIP(
    num_topics=len(TOPICS),
    num_sent=len(SENT2ID),
    num_ctx=len(CTX2ID),
    text_model_name=TEXT_MODEL_NAME,
    image_model_name=IMAGE_MODEL_NAME,
).to(device)

model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.eval()