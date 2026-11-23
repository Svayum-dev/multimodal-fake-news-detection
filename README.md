# Multimodal Fake News Detection Using Vision-Language Models

A research/academic implementation of the project proposal. It combines a news post's text and image using a pretrained CLIP vision-language encoder and a trainable multimodal fusion classifier.

## What this project does

1. Reads Fakeddit TSV splits.
2. Downloads associated images from the `image_url` field (where URLs are still accessible).
3. Encodes text and images with `openai/clip-vit-base-patch32`.
4. Fuses text and image features and trains a binary classifier.
5. Evaluates the model with accuracy, precision, recall, F1, and a confusion matrix.
6. Runs predictions through a Streamlit interface.
7. Flags low-confidence predictions as **Uncertain** and reports text-image similarity as a diagnostic signal.
8. Optionally retrieves Wikipedia context for a claim. Retrieved context is supplementary evidence, not proof that a claim is true or false.

> Important: Fakeddit's `2_way_label` is a dataset label, not a universal ground-truth fact-check. The model is a research prototype and must not be treated as an authoritative fact-checker.

## Architecture

```text
                Text / headline                Image
                      |                           |
                CLIP text encoder           CLIP image encoder
                      |                           |
                Text embedding              Image embedding
                      |                           |
                      +---- Multimodal fusion ----+
                                  |
                    [text, image, |t-i|, t*i,
                       cosine similarity]
                                  |
                         MLP classifier
                                  |
                      Real / Fake probability
                                  |
                 Confidence + mismatch diagnostic
                                  |
                  Optional contextual retrieval
```

The CLIP backbone is frozen in the initial implementation. The fusion classifier is trained on the selected Fakeddit split. This keeps training practical on a student laptop; an optional fine-tuning extension is described below.

## Repository structure

```text
multimodal-fake-news-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── images/
├── checkpoints/
├── outputs/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── evidence.py
└── scripts/
    ├── download_images.py
    └── inspect_dataset.py
```

## 1. Environment setup (VS Code)

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

PyTorch installation can differ depending on your CUDA setup. If the standard installation does not detect your NVIDIA GPU, install the appropriate CUDA-enabled PyTorch build from the official PyTorch installation selector.

## 2. Prepare Fakeddit

Download the Fakeddit multimodal TSV files from the dataset's official distribution and put them in `data/raw/`:

```text
data/raw/
├── multimodal_train.tsv
├── multimodal_validate.tsv
└── multimodal_test_public.tsv
```

The code expects these columns:
- `clean_title`: text input
- `image_url`: image URL
- `2_way_label`: binary target (0 = Real, 1 = Fake, as used by this project)
- `id`: post identifier, when present
- `hasImage`: optional image availability indicator

The loader also accepts `title` if `clean_title` is unavailable. Run:

```bash
python scripts/inspect_dataset.py --file data/raw/multimodal_train.tsv
```

Download images:

```bash
python scripts/download_images.py --file data/raw/multimodal_train.tsv
python scripts/download_images.py --file data/raw/multimodal_validate.tsv
python scripts/download_images.py --file data/raw/multimodal_test_public.tsv
```

By default, images are saved as `data/images/<id>.jpg`. The downloader skips existing images and records failed downloads. Image hosting links may be expired or rate-limited; therefore, the number of usable image-text pairs may be lower than the TSV row count.

For a quick smoke test, limit the download:

```bash
python scripts/download_images.py --file data/raw/multimodal_train.tsv --limit 500
```

## 3. Train

Start with a small subset to verify the complete pipeline:

```bash
python -m src.train --train data/raw/multimodal_train.tsv --val data/raw/multimodal_validate.tsv --max-train-rows 2000 --max-val-rows 500 --epochs 2
```

Then increase the sample size after confirming that the data and images load correctly. For a full run, omit the `--max-*` arguments:

```bash
python -m src.train --train data/raw/multimodal_train.tsv --val data/raw/multimodal_validate.tsv --epochs 5 --batch-size 16
```

The first run downloads the CLIP model weights from Hugging Face. An internet connection is required unless the model is already cached. Training saves the best checkpoint to:

```text
checkpoints/best_model.pt
```

The default batch size may need to be reduced if GPU memory is insufficient. Use `--batch-size 4` or `--batch-size 8` if you encounter CUDA out-of-memory errors.

## 4. Evaluate

```bash
python -m src.evaluate --test data/raw/multimodal_test_public.tsv --checkpoint checkpoints/best_model.pt
```

Metrics and a confusion matrix are saved in `outputs/`.

## 5. Launch the application

```bash
streamlit run app.py
```

Upload an image, enter its headline or claim, and run the prediction. The app shows:
- predicted class or Uncertain
- model confidence
- text-image cosine similarity
- a simple diagnostic explanation
- optional Wikipedia context lookup

The similarity score is not itself a truth score. A low score can indicate that the image and text are semantically different, but it does not establish that the post is fake.

## 6. Label and uncertainty behavior

The training target is binary: 0 = Real, 1 = Fake. The user interface adds an operational third state:
- **Real**: real-class probability is at least the confidence threshold.
- **Fake**: fake-class probability is at least the confidence threshold.
- **Uncertain**: neither class reaches the threshold.

This is a confidence-based abstention mechanism, not a third class learned from Fakeddit. Adjust the threshold in the Streamlit sidebar after validation.

## 7. GitHub

Before pushing, make sure raw data, images, model weights, and secrets are not committed. They are excluded by `.gitignore`.

```bash
git init
git add .
git commit -m "Initial multimodal fake news detection project"
git branch -M main
git remote add origin https://github.com/<your-username>/multimodal-fake-news-detection.git
git push -u origin main
```

## Limitations and next research steps

- Fakeddit labels can contain noise and do not cover every misinformation category.
- Some image URLs may no longer work.
- The baseline uses a frozen CLIP encoder and a trained fusion head.
- Text-image similarity is a mismatch clue, not a factual verification result.
- Wikipedia retrieval supplies context, not authoritative verification.
- The current model does not detect image manipulation or establish image recency.

Recommended extensions for the full research version:
1. Compare text-only, image-only, and multimodal baselines.
2. Add VERITE or NewsCLIPpings for image-context mismatch evaluation.
3. Add a dedicated image manipulation detector.
4. Add trustworthy news/search sources and source ranking for evidence retrieval.
5. Fine-tune CLIP or compare BLIP-2 / other vision-language models.
6. Add calibration and evaluate the uncertain/abstention rate.
7. Report per-class precision, recall, F1, macro-F1, and confusion matrix.

## Reproducibility

The training script sets random seeds and records the model name, label mapping, and validation metrics in the checkpoint. Results can still vary with hardware, package versions, dataset availability, and random initialization of the fusion head.
