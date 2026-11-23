
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


def resolve_image_path(row, image_dir: Path) -> Optional[Path]:

    for key in ("image_path", "local_image_path"):
        value = row.get(key)

        if isinstance(value, str) and value.strip():
            p = Path(value)

            if p.exists():
                return p

    image_id = row.get("id")

    if pd.notna(image_id):
        stem = str(image_id).strip()

        for suffix in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = image_dir / f"{stem}{suffix}"

            if candidate.exists():
                return candidate

    return None


class FakedditDataset(Dataset):

    def __init__(
        self,
        tsv_path: str,
        processor,
        image_dir: str = "data/images",
        max_rows: Optional[int] = None,
    ):

        self.tsv_path = tsv_path
        self.processor = processor
        self.image_dir = Path(image_dir)

        self.frame = pd.read_csv(
            tsv_path,
            sep="\t",
            low_memory=False
        )

        if max_rows is not None:
            self.frame = self.frame.head(max_rows)

        text_col = (
            "clean_title"
            if "clean_title" in self.frame.columns
            else "title"
        )

        required = [text_col, "2_way_label"]

        missing = [
            c for c in required
            if c not in self.frame.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required TSV columns: {missing}"
            )

        self.text_col = text_col
        self.samples = []

        self.skipped_missing_image = 0
        self.skipped_bad_image = 0

        for _, row in self.frame.iterrows():

            text = row.get(self.text_col)
            label = row.get("2_way_label")

            if pd.isna(text) or pd.isna(label):
                continue

            try:
                label = int(label)
            except (TypeError, ValueError):
                continue

            if label not in (0, 1):
                continue

            path = resolve_image_path(
                row,
                self.image_dir
            )

            if path is None:
                self.skipped_missing_image += 1
                continue

            try:
                with Image.open(path) as im:
                    im.verify()

            except (OSError, UnidentifiedImageError):
                self.skipped_bad_image += 1
                continue

            self.samples.append(
                (str(text), str(path), label)
            )

        if not self.samples:
            raise RuntimeError(
                f"No usable text-image rows in {tsv_path}. "
                "Check data/images."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        text, image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        # FIX: Every text sequence has exactly 77 tokens.
        encoded = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "label": label,
        }


def collate_batch(batch):

    return {
        "input_ids": torch.stack(
            [x["input_ids"] for x in batch]
        ),

        "attention_mask": torch.stack(
            [x["attention_mask"] for x in batch]
        ),

        "pixel_values": torch.stack(
            [x["pixel_values"] for x in batch]
        ),

        "labels": torch.tensor(
            [x["label"] for x in batch],
            dtype=torch.long
        ),
    }
