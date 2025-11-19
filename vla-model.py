import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel

class SimpleVLA(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        
        # Vision Encoder (Pretrained ResNet18)
        vision_model = models.resnet18(pretrained=True)
        vision_model.fc = nn.Linear(vision_model.fc.in_features, embed_dim)
        self.vision_encoder = vision_model
        
        # Text Encoder (Tiny DistilBERT)
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_projection = nn.Linear(self.text_encoder.config.dim, embed_dim)
        
    def forward(self, images, input_ids, attention_mask):
        # Encode image
        img_embeds = self.vision_encoder(images)
        img_embeds = F.normalize(img_embeds, dim=-1)
        
        # Encode text
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = self.text_projection(text_out.last_hidden_state[:, 0, :])  # CLS token
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        return img_embeds, text_embeds

    def compute_loss(self, img_embeds, text_embeds):
        # Contrastive loss (CLIP-style)
        logits = img_embeds @ text_embeds.T
        labels = torch.arange(len(logits), device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2
