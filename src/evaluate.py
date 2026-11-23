import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from src.dataset import FakedditDataset, collate_batch
from src.model import MultimodalFakeNewsModel


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint.get("model_name", "openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained(model_name)
    dataset = FakedditDataset(args.test, processor, args.image_dir, args.max_rows)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = MultimodalFakeNewsModel(model_name=model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels, predictions = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(batch["input_ids"], batch["attention_mask"], batch["pixel_values"])
        labels.extend(batch["labels"].cpu().tolist())
        predictions.extend(output["logits"].argmax(dim=1).cpu().tolist())

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "classification_report": classification_report(
            labels, predictions, target_names=["Real", "Fake"], zero_division=0, output_dict=True
        ),
    }

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot()
    plt.title("Fakeddit Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=160)
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print("\nClassification report:")
    print(classification_report(labels, predictions, target_names=["Real", "Fake"], zero_division=0))


if __name__ == "__main__":
    main()
