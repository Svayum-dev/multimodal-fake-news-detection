import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.dataset import FakedditDataset, collate_batch
from src.model import MultimodalFakeNewsModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_validation(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(
            batch["input_ids"], batch["attention_mask"], batch["pixel_values"]
        )
        loss = criterion(output["logits"], batch["labels"])
        total_loss += loss.item()
        preds = output["logits"].argmax(dim=1)
        all_labels.extend(batch["labels"].cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/best_model.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = CLIPProcessor.from_pretrained(args.model_name)
    train_ds = FakedditDataset(args.train, processor, args.image_dir, args.max_train_rows)
    val_ds = FakedditDataset(args.val, processor, args.image_dir, args.max_val_rows)

    print(f"Train usable pairs: {len(train_ds)} "
          f"(missing images: {train_ds.skipped_missing_image}, bad images: {train_ds.skipped_bad_image})")
    print(f"Validation usable pairs: {len(val_ds)} "
          f"(missing images: {val_ds.skipped_missing_image}, bad images: {val_ds.skipped_bad_image})")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_batch
    )

    model = MultimodalFakeNewsModel(model_name=args.model_name).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            result = model(
                batch["input_ids"], batch["attention_mask"], batch["pixel_values"]
            )
            loss = criterion(result["logits"], batch["labels"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = run_validation(model, val_loader, device)
        train_loss = running_loss / max(len(train_loader), 1)
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"val_loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['f1_macro']:.4f}"
        )

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "projection_dim": model.projection_dim,
                    "label_mapping": {"0": "Real", "1": "Fake"},
                    "validation_metrics": metrics,
                    "epoch": epoch + 1,
                    "seed": args.seed,
                },
                output_path,
            )
            print(f"Saved best checkpoint to {output_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
