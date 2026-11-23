import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class MultimodalFakeNewsModel(nn.Module):
    """
    Frozen CLIP encoders + trainable fusion classifier.

    Fusion vector = text, image, absolute difference, elementwise product,
    and cosine similarity. The output logits are [Real, Fake].
    """

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        hidden_dim=512,
        dropout=0.25,
        freeze_backbone=True,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.projection_dim = self.clip.config.projection_dim

        if freeze_backbone:
            for parameter in self.clip.parameters():
                parameter.requires_grad = False

        fusion_size = self.projection_dim * 4 + 1
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_size),
            nn.Linear(fusion_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # Frozen feature extraction does not need gradient tracking.
        with torch.no_grad():
            outputs = self.clip(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=False,
            )
            text_features = F.normalize(outputs.text_embeds, dim=-1)
            image_features = F.normalize(outputs.image_embeds, dim=-1)

        similarity = F.cosine_similarity(
            text_features, image_features, dim=-1
        ).unsqueeze(-1)

        fused = torch.cat(
            [
                text_features,
                image_features,
                torch.abs(text_features - image_features),
                text_features * image_features,
                similarity,
            ],
            dim=-1,
        )
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "similarity": similarity.squeeze(-1),
        }
