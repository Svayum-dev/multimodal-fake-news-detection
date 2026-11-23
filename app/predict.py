import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor

import sys
from pathlib import Path

# Add the project root directory to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MultimodalFakeNewsModel


class FakeNewsPredictor:
    def __init__(
        self,
        checkpoint_path="checkpoints/best_model.pt",
        confidence_threshold=0.60,
        similarity_threshold=0.21961045,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_name = checkpoint.get("model_name", "openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = MultimodalFakeNewsModel(model_name=self.model_name).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Dashboard settings can be changed without reloading model weights.
        self.confidence_threshold = float(confidence_threshold)
        self.similarity_threshold = float(similarity_threshold)

    @torch.no_grad()
    def predict(self, text, image):
        if not text or not text.strip():
            raise ValueError("Please enter a headline or claim.")
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")

        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt",
            padding=True, truncation=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        result = self.model(
            inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"]
        )

        probabilities = F.softmax(result["logits"], dim=-1)[0]
        confidence, predicted_index = probabilities.max(dim=0)
        confidence_value = float(confidence.item())
        predicted_index = int(predicted_index.item())
        similarity = float(result["similarity"][0].item())

        raw_class = "Real" if predicted_index == 0 else "Fake"
        label = (
            raw_class if confidence_value >= self.confidence_threshold else "Uncertain"
        )

        # Consistency screening only; it is not a factual truth judgment.
        low_alignment = similarity < self.similarity_threshold
        consistency = (
            "Potential image-text mismatch" if low_alignment else "No low-alignment flag"
        )
        if label == "Uncertain" or low_alignment:
            overall_assessment = "Review required"
        else:
            overall_assessment = "No automatic warning — claim not fact-verified"

        return {
            "label": label,
            "raw_class": raw_class,
            "confidence": confidence_value,
            "real_probability": float(probabilities[0].item()),
            "fake_probability": float(probabilities[1].item()),
            "similarity": similarity,
            "similarity_threshold": self.similarity_threshold,
            "low_alignment": low_alignment,
            "consistency": consistency,
            "overall_assessment": overall_assessment,
        }
