# Evidence retrieval update

This update adds Google News RSS discovery and best-effort publisher-page extraction to the existing Streamlit project.

## Apply the update

Extract this update ZIP into the root of your existing `multimodal-fake-news-detection` project and allow it to replace:
- `app.py`
- `src/evidence.py`
- `requirements.txt`

It does not replace your trained checkpoints, dataset, image files, training code, or evaluation outputs.

## Install the additional dependency

```powershell
python -m pip install -r requirements.txt
```

If you already have the project's dependencies installed, the important new package is:

```powershell
python -m pip install beautifulsoup4
```

## Launch

```powershell
streamlit run app.py
```

1. Enter the news claim and upload its associated image.
2. Click **Analyze text and image**.
3. Click **Search related news coverage**.
4. Open the original publisher links and inspect the retrieved context.

## Important research limitation

Google News RSS is used as a discovery mechanism. Publisher pages may block automated access, require JavaScript, or provide only partial text. The app reports the source, publication date when available, snippet, extracted text preview, and a TF-IDF lexical relevance score.

The relevance score is NOT a truth score. The app does not automatically label evidence as supporting or contradicting a claim. Such a label requires claim-level reasoning and careful source/context evaluation. Treat results as candidate evidence for human review.
