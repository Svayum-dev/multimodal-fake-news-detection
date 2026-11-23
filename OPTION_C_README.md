# Combined Decision System update

This patch updates the existing Streamlit dashboard without retraining.

## Files
- `app/app.py` — combined dashboard and assessment.
- `app/predict.py` — uses the validation-calibrated similarity threshold by default.
- `app/evidence.py` — preserved from your uploaded project.

## Installation
1. Extract this ZIP in your project folder.
2. Replace the corresponding files inside your existing `app` folder.
3. Keep your existing `src/model.py`, `checkpoints/best_model.pt`, dataset, and other project files unchanged.
4. Open PowerShell in the project root (the folder containing `app`, `src`, and `checkpoints`).
5. Run:

   ```powershell
   streamlit run app/app.py
   ```

The dashboard imports `predict` and `evidence` from the same `app` folder. `app/predict.py` continues to import your existing model class from `src/model.py`.

## What changed
- Combined summary across classifier, image consistency, and evidence retrieval.
- Similarity threshold defaults to `0.21961045`, selected on validation data using maximum Youden J.
- Changing thresholds does not reload the model.
- Evidence relevance is explicitly described as topical relevance, not support/contradiction or truth.
- Existing checkpoint is not modified and no training is performed.

## Important
The 91.90% classification accuracy is a Fakeddit test-set result, not a guarantee of real-world fact-checking accuracy. Synthetic random image swaps are a consistency experiment; low similarity does not prove a claim is false.
