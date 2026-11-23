from pathlib import Path

import streamlit as st
from PIL import Image

from predict import FakeNewsPredictor
from evidence import search_news_evidence


CALIBRATED_SIMILARITY_THRESHOLD = 0.21961045


st.set_page_config(
    page_title="Multimodal Fake News Detector",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Multimodal Fake News Detection")
st.caption(
    "Combined research prototype: multimodal classification, image-text consistency "
    "screening, and external news evidence retrieval."
)


@st.cache_resource
def load_predictor(path: str, checkpoint_mtime: float):
    # Cache model weights independently of UI thresholds.
    return FakeNewsPredictor(
        checkpoint_path=path,
        confidence_threshold=0.60,
        similarity_threshold=CALIBRATED_SIMILARITY_THRESHOLD,
    )


def combined_assessment(prediction, evidence):
    """Conservative summary; it does not infer factual truth."""
    reasons = []
    if prediction["label"] == "Uncertain":
        reasons.append("The classifier confidence is below the selected threshold.")
    if prediction["low_alignment"]:
        reasons.append(
            "Image-text similarity is below the validation-calibrated review threshold."
        )

    if evidence is None:
        evidence_line = "External evidence has not been retrieved yet."
    elif evidence.get("results"):
        evidence_line = (
            f'{len(evidence["results"])} related coverage candidate(s) were retrieved. '
            "Their relevance scores measure topical similarity, not support or contradiction."
        )
    else:
        evidence_line = "No related news results were returned for this query."

    if reasons:
        headline = "REVIEW RECOMMENDED"
        explanation = " ".join(reasons)
    else:
        headline = "NO AUTOMATIC WARNING"
        explanation = (
            "The classifier returned a non-abstaining result and the consistency screen "
            "did not raise a low-similarity flag. This is not factual verification."
        )
    return headline, explanation, evidence_line


with st.sidebar:
    st.header("Analysis settings")
    confidence_threshold = st.slider(
        "Classifier confidence threshold",
        min_value=0.50, max_value=0.95, value=0.60, step=0.05,
        help="Below this confidence, the classifier displays Uncertain.",
    )
    similarity_threshold = st.number_input(
        "Image-text review threshold",
        min_value=0.05, max_value=0.50,
        value=CALIBRATED_SIMILARITY_THRESHOLD,
        step=0.005, format="%.5f",
        help=(
            "Default selected by maximum Youden J on validation data. "
            "It is a consistency screening threshold, not a truth probability."
        ),
    )
    st.caption(f"Validation-calibrated default: {CALIBRATED_SIMILARITY_THRESHOLD:.8f}")
    st.divider()
    st.caption(
        "The classifier, image consistency screen, and retrieved news coverage are "
        "separate signals. None alone establishes whether a claim is true."
    )

checkpoint_path = Path("checkpoints/best_model.pt")
if not checkpoint_path.exists():
    st.warning(
        "No trained checkpoint found at checkpoints/best_model.pt. "
        "Place your existing checkpoint there and relaunch the app."
    )
    st.stop()

try:
    predictor = load_predictor(str(checkpoint_path), checkpoint_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

st.subheader("1. Analyze a news post")
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    claim = st.text_area(
        "News headline or claim",
        placeholder="Enter the claim associated with the uploaded image...",
        height=130,
    )
    uploaded_image = st.file_uploader(
        "Upload associated image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    image = None
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read the uploaded image: {exc}")

    analyze = st.button("Analyze text and image", type="primary", use_container_width=True)

with right:
    st.markdown("**Three independent analysis signals**")
    st.markdown(
        """
        **1. AI classifier**  
        Predicts Real, Fake, or Uncertain using the Fakeddit-trained model.

        **2. Image consistency**  
        Uses CLIP cosine similarity to flag possible image-text mismatch.

        **3. External evidence**  
        Retrieves related news coverage for human inspection.
        """
    )
    st.info(
        "Low similarity does not prove a claim is false. News images can be illustrative, "
        "symbolic, or contextual rather than literal depictions."
    )

if analyze:
    if not claim.strip():
        st.error("Enter a headline or claim.")
    elif image is None:
        st.error("Upload a valid image.")
    else:
        with st.spinner("Running classifier and CLIP consistency check..."):
            try:
                predictor.confidence_threshold = float(confidence_threshold)
                predictor.similarity_threshold = float(similarity_threshold)
                prediction = predictor.predict(claim, image)
                st.session_state["last_prediction"] = prediction
                st.session_state["last_claim"] = claim.strip()
                st.session_state.pop("last_evidence", None)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]
    evidence = st.session_state.get("last_evidence")

    st.divider()
    st.subheader("2. Combined assessment")
    headline, explanation, evidence_line = combined_assessment(result, evidence)
    if headline == "REVIEW RECOMMENDED":
        st.warning(f"**{headline}**\n\n{explanation}")
    else:
        st.info(f"**{headline}**\n\n{explanation}")
    st.markdown(f"**External evidence status:** {evidence_line}")
    st.caption("This is a transparent review aid, not an automated fact-checking verdict.")

    st.divider()
    st.subheader("3. Signal-by-signal results")
    classifier_col, consistency_col = st.columns(2, gap="large")

    with classifier_col:
        st.markdown("#### AI classification")
        if result["label"] == "Real":
            st.success("Classifier prediction: REAL")
        elif result["label"] == "Fake":
            st.error("Classifier prediction: FAKE")
        else:
            st.warning("Classifier prediction: UNCERTAIN")
        st.metric("Classifier confidence", f"{result['confidence'] * 100:.1f}%")
        st.metric("Real probability", f"{result['real_probability'] * 100:.1f}%")
        st.metric("Fake probability", f"{result['fake_probability'] * 100:.1f}%")
        st.caption(
            "These probabilities come from the Fakeddit-trained classifier; "
            "they are not live fact-checking probabilities."
        )

    with consistency_col:
        st.markdown("#### Image-text consistency")
        st.metric("CLIP cosine similarity", f"{result['similarity']:.5f}")
        st.metric("Review threshold", f"{result['similarity_threshold']:.5f}")
        if result["low_alignment"]:
            st.warning(
                "Potential image-text mismatch. Inspect the image and its context "
                "before drawing a conclusion."
            )
        else:
            st.success("No low-similarity review flag was raised.")
        st.caption(
            "The threshold was selected on validation data using maximum Youden J. "
            "Similarity is not a probability that the image or claim is true."
        )

    st.divider()
    st.subheader("4. External news evidence")
    st.write(
        "Retrieve related coverage to help a person investigate the claim. "
        "The system ranks topical relevance only; it does not automatically label "
        "articles as supporting or contradicting."
    )

    if st.button("Search related news coverage", type="primary"):
        with st.spinner("Searching Google News RSS and retrieving available article text..."):
            try:
                evidence = search_news_evidence(
                    st.session_state["last_claim"], limit=6, extract_articles=True
                )
                st.session_state["last_evidence"] = evidence
            except Exception as exc:
                st.error(
                    "News retrieval failed. Check your internet connection and try again. "
                    f"Details: {exc}"
                )

    evidence = st.session_state.get("last_evidence")
    if evidence is not None:
        _, _, updated_evidence_line = combined_assessment(result, evidence)
        st.markdown(f"**Updated evidence status:** {updated_evidence_line}")
        st.caption(
            f"Provider: {evidence['provider']} · Retrieved: {evidence['retrieved_at']}"
        )
        st.warning(evidence["notice"])

        if not evidence["results"]:
            st.info("No related news results were returned for this query.")
        else:
            for index, item in enumerate(evidence["results"], start=1):
                title = item["title"] or "Untitled result"
                with st.expander(
                    f"{index}. {title} · topical relevance {item['relevance']:.3f}",
                    expanded=(index == 1),
                ):
                    st.markdown(f"**Source:** {item['source']}")
                    st.markdown(f"**Published:** {item['published']}")
                    st.markdown(f"**Retrieval:** {item['retrieval_status']}")
                    st.markdown(f"[Open original article]({item['url']})")
                    if item["snippet"]:
                        st.markdown("**Search snippet**")
                        st.write(item["snippet"])
                    if item["article_text"]:
                        st.markdown("**Extracted article text (preview)**")
                        st.write(
                            item["article_text"][:1800]
                            + ("..." if len(item["article_text"]) > 1800 else "")
                        )
                    else:
                        st.caption(
                            "Full article text could not be extracted. Open the publisher "
                            "link to inspect the original page."
                        )

    st.divider()
    st.caption(
        "Research limitations: retrieved coverage may be incomplete, biased, duplicated, "
        "or contextually unrelated. Relevance scores are lexical retrieval scores, not "
        "truth scores. The system does not infer support or contradiction automatically."
    )
