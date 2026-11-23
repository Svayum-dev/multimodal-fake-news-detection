from pathlib import Path
import streamlit as st
from PIL import Image

from src.predict import FakeNewsPredictor
from src.evidence import search_news_evidence

st.set_page_config(
    page_title="Multimodal Fake News Detector",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Multimodal Fake News Detection")
st.caption(
    "Research prototype combining a Fakeddit-trained classifier, "
    "CLIP image-text alignment screening, and external news coverage retrieval."
)

with st.sidebar:
    st.header("Model settings")
    threshold = st.slider(
        "Classifier confidence threshold",
        min_value=0.50, max_value=0.95, value=0.60, step=0.05,
        help="Below this confidence, the classifier abstains and displays Uncertain.",
    )
    similarity_threshold = st.slider(
        "Image-text review threshold (experimental)",
        min_value=0.05, max_value=0.50, value=0.25, step=0.01,
        help="A CLIP cosine similarity below this value raises a review flag. "
             "This is a provisional heuristic, not a calibrated truth threshold.",
    )
    st.divider()
    st.caption(
        "The classifier predicts dataset patterns. The alignment score can flag "
        "possible mismatches, but neither establishes whether a claim is true."
    )

checkpoint_path = Path("checkpoints/best_model.pt")
if not checkpoint_path.exists():
    st.warning(
        "No trained checkpoint found at checkpoints/best_model.pt. "
        "Train the model first, then relaunch the app."
    )
    st.stop()


@st.cache_resource
def load_predictor(path, confidence_threshold, similarity_threshold):
    return FakeNewsPredictor(path, confidence_threshold, similarity_threshold)


predictor = load_predictor(str(checkpoint_path), threshold, similarity_threshold)

st.subheader("1. Analyze a post")
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
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
    analyze = st.button("Analyze text and image", type="primary", use_container_width=True)

with right:
    st.markdown("**What the system checks**")
    st.markdown(
        "- Fakeddit-trained Real/Fake classification\n"
        "- CLIP text-image semantic alignment\n"
        "- Potential image-text mismatch screening\n"
        "- Related news coverage from Google News RSS"
    )
    st.info(
        "A low image-text similarity is a reason to review the post, not proof of "
        "misinformation. Images may be illustrative, symbolic, or contextual."
    )

if analyze:
    if not claim.strip():
        st.error("Enter a headline or claim.")
    elif image is None:
        st.error("Upload an image.")
    else:
        with st.spinner("Analyzing text and image..."):
            try:
                prediction = predictor.predict(claim, image)
                st.session_state["last_prediction"] = prediction
                st.session_state["last_claim"] = claim
                st.session_state.pop("last_evidence", None)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]
    label = result["label"]
    st.divider()
    st.subheader("2. Model prediction")

    if label == "Real":
        st.success("Classifier prediction: REAL")
    elif label == "Fake":
        st.error("Classifier prediction: FAKE")
    else:
        st.warning("Classifier prediction: UNCERTAIN")

    c1, c2, c3 = st.columns(3)
    c1.metric("Classifier confidence", f"{result['confidence'] * 100:.1f}%")
    c2.metric("Real probability", f"{result['real_probability'] * 100:.1f}%")
    c3.metric("Fake probability", f"{result['fake_probability'] * 100:.1f}%")

    st.subheader("3. Image-text consistency screening")
    c1, c2 = st.columns(2)
    c1.metric("CLIP cosine similarity", f"{result['similarity']:.3f}")
    c2.metric("Experimental review threshold", f"{result['similarity_threshold']:.2f}")

    if result["low_alignment"]:
        st.warning(
            "Potential image-text mismatch: the CLIP similarity is below the selected "
            "experimental threshold. Inspect whether the image actually depicts the "
            "event described by the claim."
        )
    else:
        st.info(
            "No low-alignment flag was raised. This does not establish that the image "
            "matches the claim or that the claim is true."
        )

    st.markdown(f"**Overall screening assessment:** {result['overall_assessment']}")
    st.caption(
        "Important: the Real/Fake probabilities come from the Fakeddit-trained "
        "classifier, not from live fact-checking. The similarity threshold is "
        "provisional and must be calibrated on a labeled image-text mismatch test set."
    )

    st.divider()
    st.subheader("4. External news evidence")
    st.write(
        "Search for recent and related news coverage. Results are candidate sources "
        "for manual verification, not automatic verdicts."
    )

    if st.button("Search related news coverage", type="primary"):
        with st.spinner("Searching news and retrieving available article text..."):
            try:
                evidence = search_news_evidence(
                    st.session_state["last_claim"],
                    limit=6,
                    extract_articles=True,
                )
                st.session_state["last_evidence"] = evidence
            except Exception as exc:
                st.error(
                    "News retrieval failed. Check your internet connection and try again. "
                    f"Details: {exc}"
                )

    if "last_evidence" in st.session_state:
        evidence = st.session_state["last_evidence"]
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
                    f"{index}. {title} · relevance {item['relevance']:.3f}",
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
                        st.write(item["article_text"][:1800] + (
                            "..." if len(item["article_text"]) > 1800 else ""
                        ))
                    else:
                        st.caption(
                            "Full article text could not be extracted. Open the publisher "
                            "link to inspect the original page."
                        )

    st.divider()
    st.caption(
        "Research limitation: retrieved coverage may be incomplete, biased, duplicated, "
        "or unrelated in context. Relevance scores are lexical retrieval scores, not "
        "probabilities of truth. This app does not automatically infer support or contradiction."
    )
