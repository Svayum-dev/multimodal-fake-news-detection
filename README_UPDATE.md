# Image-text consistency screening update

## Files
- `app.py` — updated Streamlit interface with a separate image-text screening section.
- `src/predict.py` — returns classifier output plus a low-alignment review flag.

## Install
1. Back up your current `app.py` and `src/predict.py`.
2. Extract this ZIP into the root of your existing project, preserving the `src` folder.
3. Replace those two files when prompted. Do not replace `src/model.py`, `src/train.py`, your dataset, or checkpoints.
4. Run:
   ```powershell
   streamlit run app.py
   ```

## Important interpretation
The classifier's Real/Fake output is unchanged. The CLIP cosine similarity now creates a separate
potential-mismatch review flag. The default threshold (0.25) is an experimental starting point,
not a validated universal cutoff. Calibrate it using a labeled validation set containing both
matching and deliberately mismatched image-text pairs. A low score is not proof of fake news;
a high score is not proof of truth.

No retraining is required for this software update.
