
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA = Path("data/raw")

def prepare_sample(source, destination, target, seed=42):

    print(f"\nPreparing {source}...")

    df = pd.read_csv(DATA / source, sep="\t", low_memory=False)

    # Keep rows with valid text, image URL and binary labels
    text_col = "clean_title" if "clean_title" in df.columns else "title"

    df = df.dropna(
        subset=[text_col, "image_url", "2_way_label", "id"]
    )

    df = df[
        df["image_url"].astype(str).str.startswith(
            ("http://", "https://")
        )
    ].copy()

    df["2_way_label"] = pd.to_numeric(
        df["2_way_label"], errors="coerce"
    )

    df = df[df["2_way_label"].isin([0, 1])]

    if len(df) < target:
        raise ValueError(
            f"Only {len(df)} eligible rows available; "
            f"need {target}."
        )

    sample, _ = train_test_split(
        df,
        train_size=target,
        stratify=df["2_way_label"],
        random_state=seed
    )

    sample.to_csv(
        DATA / destination,
        sep="\t",
        index=False
    )

    print(f"Saved: {destination}")
    print(f"Selected rows: {len(sample)}")
    print("Label distribution:")
    print(sample["2_way_label"].value_counts())


prepare_sample(
    "multimodal_train.tsv",
    "train_56000.tsv",
    56000,
    seed=42
)

prepare_sample(
    "multimodal_validate.tsv",
    "validate_6000.tsv",
    6000,
    seed=43
)

prepare_sample(
    "multimodal_test_public.tsv",
    "test_6000.tsv",
    6000,
    seed=44
)

print("\nAll samples prepared successfully!")
