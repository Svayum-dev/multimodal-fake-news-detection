import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor
from src.model import MultimodalFakeNewsModel


class FakeNewsPredictor:
    def __init__(
        self,
        checkpoint_path="checkpoints/best_model.pt",
        confidence_threshold=0.60,
        similarity_threshold=0.25,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_name = checkpoint.get("model_name", "openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = MultimodalFakeNewsModel(model_name=self.model_name).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        result = self.model(
            inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"]
        )

        probabilities = F.softmax(result["logits"], dim=-1)[0]
        confidence, predicted_index = probabilities.max(dim=0)
        confidence_value = float(confidence.item())
        predicted_index = int(predicted_index.item())
        similarity = float(result["similarity"][0].item())

        raw_class = "Real" if predicted_index == 0 else "Fake"
        label = raw_class if confidence_value >= self.confidence_threshold else "Uncertain"

        # This is a screening flag only. The threshold is a provisional heuristic,
        # not a calibrated probability or proof that an image is unrelated.
        low_alignment = similarity < self.similarity_threshold
        if low_alignment:
            consistency = "Potential image-text mismatch"
            overall_assessment = "Review required"
        else:
            consistency = "No low-alignment flag"
            overall_assessment = "Classifier output only — claim not fact-verified"

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
