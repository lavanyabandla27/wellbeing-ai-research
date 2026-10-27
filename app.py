
import json
import re
from pathlib import Path

import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Well-Being AI Research Framework",
    page_icon="🧠",
    layout="wide",
)

st.title("Well-Being AI Research Framework")
st.subheader(
    "Stress Detection · Emotion Analysis · Evidence-Grounded Support"
)
st.warning(
    "Experimental research prototype only. Predictions may be incorrect; "
    "this app is not a diagnosis or a validated crisis detector."
)

ARTIFACTS = Path("artifacts")


@st.cache_resource
def load_models():
    stress_model = joblib.load(ARTIFACTS / "stress_model.joblib")
    emotion_artifact = joblib.load(ARTIFACTS / "emotion_model.joblib")

    if isinstance(emotion_artifact, dict):
        emotion_model = emotion_artifact["model"]
        emotion_labels = emotion_artifact.get("emotion_labels", [])
    else:
        emotion_model = emotion_artifact
        emotion_labels = []

    with open(ARTIFACTS / "knowledge_base.json", encoding="utf-8") as f:
        knowledge_base = json.load(f)

    return stress_model, emotion_model, emotion_labels, knowledge_base


def safety_route(text):
    patterns = [
        r"\bkill myself\b",
        r"\bend my life\b",
        r"\bsuicidal\b",
        r"\bhurt myself\b",
        r"\bhurting myself\b",
        r"\bharm myself\b",
        r"\bharming myself\b",
        r"\bself[- ]harm\b",
        r"\bwant to die\b",
    ]
    return any(re.search(p, text, re.I) for p in patterns)


def predict_stress(model, text):
    # Supports sklearn Pipeline models that expose predict_proba.
    label = int(model.predict([text])[0])
    score = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        classes = list(model.classes_)
        if 1 in classes:
            score = float(probabilities[classes.index(1)])

    return label, score



def predict_emotions(model, labels, text):
    if not hasattr(model, "predict"):
        return []

    prediction = model.predict([text])

    if hasattr(prediction, "toarray"):
        prediction = prediction.toarray()

    values = prediction[0]
    results = []

    for i, value in enumerate(values):
        if i >= len(labels):
            continue

        try:
            score = float(value)
        except (TypeError, ValueError):
            continue

        if score > 0:
            results.append({
                "label": str(labels[i]),
                "score": score
            })

    return sorted(
        results,
        key=lambda item: item["score"],
        reverse=True
    )[:5]



try:
    stress_model, emotion_model, emotion_labels, knowledge_base = load_models()
except Exception as exc:
    st.error(
        "Could not load the model files. Check that all three files are "
        "committed inside the artifacts/ folder and that dependencies match."
    )
    st.exception(exc)
    st.stop()


text = st.text_area(
    "Describe your experience",
    placeholder="Example: I am worried about my upcoming exams...",
    height=140,
)

if st.button("Analyze Text", type="primary"):
    if not text.strip():
        st.warning("Please enter text to analyze.")

    elif safety_route(text):
        st.error("High-risk support route")
        st.markdown(
            """
            Thank you for sharing this. Your safety matters.

            If you may act on thoughts of harming yourself or are in immediate
            danger, contact local emergency services or go to the nearest
            emergency department. If possible, reach out to someone you trust
            and ask them to stay with you while you get support.

            This prototype cannot assess your safety and is not a validated
            crisis detector.
            """
        )

    else:
        try:
            label, score = predict_stress(stress_model, text)

            st.markdown("### Stress Analysis")
            st.write(f"Predicted label: {label}")
            st.write(
                "Stress score: "
                + (f"{score:.4f}" if score is not None else "Unavailable")
            )
            st.caption(
                "The score is not a calibrated probability or diagnosis."
            )

            emotions = predict_emotions(
                emotion_model, emotion_labels, text
            )


            st.markdown("### Emotion Predictions")
            
            if emotions:
                st.json(emotions)
            
                # Each emotion is a dictionary: {"label": ..., "score": ...}
                names = [item["label"] for item in emotions][::-1]
                scores = [item["score"] for item in emotions][::-1]
            
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(names, scores)
                ax.set_xlabel("Model output score (not calibrated probability)")
                ax.set_title("Predicted Emotion Labels")
                ax.grid(axis="x", alpha=0.25)
                fig.tight_layout()
            
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No emotion scores were available.")


            st.markdown("### Evidence Resources")
            st.caption(
                "Resource relevance must be checked before showing a link. "
                "This starter version does not yet run your original retriever."
            )

        except Exception as exc:
            st.error("Analysis failed. Check model compatibility.")
            st.exception(exc)
