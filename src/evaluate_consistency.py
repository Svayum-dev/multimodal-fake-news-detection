
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image
from transformers import CLIPProcessor

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.dataset import FakedditDataset
from src.model import MultimodalFakeNewsModel


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name = checkpoint.get(
        "model_name",
        "openai/clip-vit-base-patch32"
    )

    processor = CLIPProcessor.from_pretrained(model_name)

    model = MultimodalFakeNewsModel(
        model_name=model_name
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, processor


def create_mismatch_mapping(n, seed):
    if n < 2:
        raise ValueError("At least two samples are required.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    mapping = np.empty(n, dtype=int)
    mapping[order] = np.roll(order, 1)

    return mapping


@torch.no_grad()
def evaluate_split(
    split_name,
    tsv_path,
    model,
    processor,
    device,
    image_dir,
    batch_size,
    max_rows,
    seed
):
    dataset = FakedditDataset(
        tsv_path,
        processor,
        image_dir,
        max_rows
    )

    n = len(dataset)
    mapping = create_mismatch_mapping(n, seed)

    print(f"\nEvaluating {split_name}: {n} samples")

    records = []

    all_labels = []
    all_predictions = []

    matched_scores = []
    mismatched_scores = []

    for start in range(0, n, batch_size):

        end = min(start + batch_size, n)
        indices = list(range(start, end))

        texts = [
            dataset.samples[i][0]
            for i in indices
        ]

        labels = [
            dataset.samples[i][2]
            for i in indices
        ]

        matched_images = []
        mismatched_images = []

        for i in indices:

            matched_path = dataset.samples[i][1]
            mismatch_index = mapping[i]
            mismatch_path = dataset.samples[mismatch_index][1]

            with Image.open(matched_path) as image:
                matched_images.append(image.convert("RGB"))

            with Image.open(mismatch_path) as image:
                mismatched_images.append(image.convert("RGB"))

        matched_inputs = processor(
            text=texts,
            images=matched_images,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        )

        mismatched_inputs = processor(
            text=texts,
            images=mismatched_images,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        )

        matched_inputs = {
            k: v.to(device)
            for k, v in matched_inputs.items()
        }

        mismatched_inputs = {
            k: v.to(device)
            for k, v in mismatched_inputs.items()
        }

        matched_output = model(
            matched_inputs["input_ids"],
            matched_inputs["attention_mask"],
            matched_inputs["pixel_values"]
        )

        mismatched_output = model(
            mismatched_inputs["input_ids"],
            mismatched_inputs["attention_mask"],
            mismatched_inputs["pixel_values"]
        )

        probabilities = torch.softmax(
            matched_output["logits"],
            dim=1
        )

        predictions = probabilities.argmax(dim=1)
        confidence = probabilities.max(dim=1).values

        match_scores = (
            matched_output["similarity"].cpu().numpy()
        )

        mismatch_scores = (
            mismatched_output["similarity"].cpu().numpy()
        )

        matched_scores.extend(match_scores.tolist())
        mismatched_scores.extend(mismatch_scores.tolist())

        all_labels.extend(labels)
        all_predictions.extend(predictions.cpu().tolist())

        for j, i in enumerate(indices):

            text, image_path, label = dataset.samples[i]

            records.append({
                "split": split_name,
                "text": text,
                "true_label": label,
                "predicted_label": int(predictions[j].item()),
                "confidence": float(confidence[j].item()),
                "real_probability": float(probabilities[j, 0].item()),
                "fake_probability": float(probabilities[j, 1].item()),
                "matched_image": image_path,
                "matched_similarity": float(match_scores[j]),
                "mismatched_image": dataset.samples[mapping[i]][1],
                "mismatched_similarity": float(mismatch_scores[j])
            })

        print(f"Processed {end}/{n}")

    return {
        "records": records,
        "labels": np.array(all_labels),
        "predictions": np.array(all_predictions),
        "matched_scores": np.array(matched_scores),
        "mismatched_scores": np.array(mismatched_scores)
    }


def classification_metrics(labels, predictions):

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "precision_macro": float(
            precision_score(
                labels, predictions,
                average="macro",
                zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                labels, predictions,
                average="macro",
                zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                labels, predictions,
                average="macro",
                zero_division=0
            )
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["Real", "Fake"],
            zero_division=0,
            output_dict=True
        )
    }


def find_best_threshold(matched, mismatched):

    actual = np.concatenate([
        np.ones(len(matched)),
        np.zeros(len(mismatched))
    ])

    scores = np.concatenate([
        matched,
        mismatched
    ])

    fpr, tpr, thresholds = roc_curve(actual, scores)

    best_index = np.argmax(tpr - fpr)

    return float(thresholds[best_index])


def consistency_metrics(matched, mismatched, threshold):

    actual = np.concatenate([
        np.ones(len(matched), dtype=int),
        np.zeros(len(mismatched), dtype=int)
    ])

    scores = np.concatenate([
        matched,
        mismatched
    ])

    predicted = (scores >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(actual, scores)
        ),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).tolist()
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt"
    )

    parser.add_argument(
        "--image-dir",
        default="data/images"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output-dir",
        default="outputs/consistency"
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(
        args.checkpoint,
        device
    )

    # 1. Evaluate validation data.
    validation = evaluate_split(
        "validation",
        args.val,
        model,
        processor,
        device,
        args.image_dir,
        args.batch_size,
        args.max_val_rows,
        args.seed
    )

    # 2. Calibrate threshold using validation only.
    threshold = find_best_threshold(
        validation["matched_scores"],
        validation["mismatched_scores"]
    )

    validation_consistency = consistency_metrics(
        validation["matched_scores"],
        validation["mismatched_scores"],
        threshold
    )

    print("\nValidation threshold:", threshold)

    # 3. Evaluate untouched test data.
    test = evaluate_split(
        "test",
        args.test,
        model,
        processor,
        device,
        args.image_dir,
        args.batch_size,
        args.max_test_rows,
        args.seed + 1
    )

    test_classification = classification_metrics(
        test["labels"],
        test["predictions"]
    )

    test_consistency = consistency_metrics(
        test["matched_scores"],
        test["mismatched_scores"],
        threshold
    )

    # 4. Save individual predictions.
    test_frame = pd.DataFrame(test["records"])

    test_frame["matched_flag"] = (
        test_frame["matched_similarity"] >= threshold
    )

    test_frame["mismatched_flag"] = (
        test_frame["mismatched_similarity"] >= threshold
    )

    test_frame.to_csv(
        output_dir / "test_consistency_predictions.csv",
        index=False
    )

    pd.DataFrame(
        validation["records"]
    ).to_csv(
        output_dir / "validation_consistency_predictions.csv",
        index=False
    )

    # 5. Save report.
    report = {
        "note": (
            "Random image swaps are synthetic mismatches. "
            "Low similarity does not prove a claim is false."
        ),
        "threshold_selection": (
            "Maximum Youden J on validation data"
        ),
        "validation_threshold": threshold,
        "validation_consistency": validation_consistency,
        "test_classification": test_classification,
        "test_consistency": test_consistency
    }

    with open(
        output_dir / "consistency_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    # 6. Plot similarity distributions.
    plt.figure(figsize=(9, 5))

    plt.hist(
        test["matched_scores"],
        bins=40,
        alpha=0.6,
        label="Matched"
    )

    plt.hist(
        test["mismatched_scores"],
        bins=40,
        alpha=0.6,
        label="Random mismatches"
    )

    plt.axvline(
        threshold,
        linestyle="--",
        label=f"Threshold: {threshold:.3f}"
    )

    plt.xlabel("CLIP cosine similarity")
    plt.ylabel("Number of samples")
    plt.title("Image-text consistency on test data")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "similarity_distribution.png",
        dpi=160
    )

    plt.close()

    # 7. Plot confusion matrix.
    actual = np.concatenate([
        np.ones(len(test["matched_scores"]), dtype=int),
        np.zeros(len(test["mismatched_scores"]), dtype=int)
    ])

    scores = np.concatenate([
        test["matched_scores"],
        test["mismatched_scores"]
    ])

    predicted = (scores >= threshold).astype(int)

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)

    plt.xticks(
        [0, 1],
        ["Predicted mismatch", "Predicted match"],
        rotation=15
    )

    plt.yticks(
        [0, 1],
        ["Actual mismatch", "Actual match"]
    )

    plt.title("Image-text consistency confusion matrix")

    for (i, j), value in np.ndenumerate(cm):
        plt.text(
            j, i, str(value),
            ha="center",
            va="center"
        )

    plt.tight_layout()

    plt.savefig(
        output_dir / "consistency_confusion_matrix.png",
        dpi=160
    )

    plt.close()

    print("\n===== TEST CLASSIFICATION =====")

    print(json.dumps({
        key: value
        for key, value in test_classification.items()
        if key != "classification_report"
    }, indent=2))

    print("\n===== TEST IMAGE-TEXT CONSISTENCY =====")

    print(json.dumps(test_consistency, indent=2))

    print("\nSelected threshold:", threshold)
    print("Results saved in:", output_dir.resolve())


if __name__ == "__main__":
    main()
