# app.py
import streamlit as st
import numpy as np
import pandas as pd
import re
import json
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------- SETTINGS: change paths if needed ----------
MODEL_PATH = "lstm_sentiment.h5"
TOKENIZER_PATH = "tokenizer.pickle"
LABEL_MAP_PATH = "label_map.json"
DEFAULT_LABELS = ["negative", "neutral", "positive"]  # adjust if your labels differ
# ----------------------------------------------------

st.set_page_config(page_title="Climate Tweet Sentiment Predictor", layout="wide")

st.title("Climate Tweet Sentiment Predictor")
#st.write("Paste a tweet or upload a CSV (column name 'tweet') to get sentiment and Nigeria-specific stakeholder recommendations.\nThis will predict the sentiment of the tweet and give recommendatons to relevant stakeholders.")
# Change the color of the text to white using markdown
st.markdown("<p style='color: black; '>Paste a tweet or upload a CSV (column name 'tweet') to get sentiment of the tweet and give recommendations to relevant stake holders</p>", unsafe_allow_html=True)


# ----------------- Utilities -----------------
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model = None
    tokenizer = None
    label_map = None
    # load model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model from {MODEL_PATH}: {e}")
    # load tokenizer
    try:
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        st.error(f"Could not load tokenizer from {TOKENIZER_PATH}: {e}")
    # load label mapping if exists
    if Path(LABEL_MAP_PATH).exists():
        try:
            with open(LABEL_MAP_PATH, "r") as f:
                label_map = json.load(f)
            # ensure it's a dict mapping str(index)->label
            # convert to list in correct order
            labels = [label_map[str(i)] for i in range(len(label_map))]
            label_map = labels
        except Exception as e:
            st.warning(f"Failed to load label_map.json: {e}. Falling back to defaults.")
            label_map = DEFAULT_LABELS
    else:
        st.warning("label_map.json not found — using default labels. Make sure order matches your training.")
        label_map = DEFAULT_LABELS

    # determine maxlen from model input shape if possible
    maxlen = None
    if model is not None:
        try:
            maxlen = model.input_shape[1]
        except Exception:
            maxlen = None
    return model, tokenizer, label_map, maxlen

def basic_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove urls, mentions, hashtags (or keep hashtag word?)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    # keep emojis? remove non-alphanumeric (except spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_texts(texts, model, tokenizer, maxlen=None):
    cleaned = [basic_clean(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    if maxlen is None:
        # fallback length
        maxlen = max(len(s) for s in seqs) if seqs else 50
    X = pad_sequences(seqs, maxlen=maxlen)
    preds = model.predict(X)
    return preds, cleaned

# Recommendations mapping (Nigeria-focused). Edit as needed.
RECOMMENDATIONS = {
    "negative": {
        "stakeholders": [
            "Federal Ministry of Environment (FMEnv)",
            "National Emergency Management Agency (NEMA)",
            "State Environmental Protection Agencies",
            "Local Government Environmental Departments",
            "Relevant NGOs and CSOs (e.g., Climate Action NGOs)",
            "Affected communities (farmers, fishers, informal settlements)"
        ],
        "actions": [
            "Investigate the reported issue urgently — collect location, time, and evidence (photos/videos).",
            "If life/property at risk -> escalate to NEMA & local emergency services immediately.",
            "Coordinate rapid response: mobilize clean-up, temporary shelters, or relief if needed.",
            "Open channels for community reporting and provide guidance on safe actions.",
            "Run localized awareness campaigns if misinformation is causing panic."
        ]},
    "neutral": {
        "stakeholders": [
            "FMEnv (for monitoring and policy info)",
            "State/local environmental offices",
            "Academic institutions for monitoring",
            "Climate NGOs"
        ],
        "actions": [
            "Log and monitor the issue — collect metadata and check for repeated reports.",
            "If pattern emerges, consider targeted research or community engagement.",
            "Share verified informational resources and guidance to the poster/community."
        ]},
    "positive": {
        "stakeholders": [
            "Federal Ministry of Environment (positive case = mitigation/adaptation success)",
            "State government (if local project)",
            "Community leaders",
            "NGOs & donors"
        ],
        "actions": [
            "Validate the claim and collect details for showcasing as a best practice.",
            "Amplify successful community-led initiatives to other communities.",
            "Engage funders or partners to scale the successful intervention."
        ]}
}

# ---------- App UI ----------
# Function to add background image
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-attachment: scroll;
            background-size: cover;
            background-color: rgba(0,0,0,0.1);
            
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local('2149217819.jpg')

model, tokenizer, labels, model_maxlen = load_model_and_tokenizer()

if model is None or tokenizer is None:
    st.error("model or tokenizer failed to load — fix file paths and restart. See instructions at top of the app.")
    st.stop()

st.sidebar.header("Options")
batch_mode = st.sidebar.checkbox("Batch mode (upload CSV)", value=False)
show_conf_threshold = st.sidebar.slider("Confidence threshold (for highlighting)", 0.5, 0.9, 0.7)

if batch_mode:
    uploaded = st.file_uploader("Upload CSV with a column named 'tweet'", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'tweet' not in df.columns:
            st.error("CSV must have a column named 'tweet'.")
        else:
            st.write("Preview:")
            st.dataframe(df.head())
            if st.button("Predict all"):
                preds, cleaned = predict_texts(df['tweet'].astype(str).tolist(), model, tokenizer, maxlen=model_maxlen)
                pred_labels = [labels[np.argmax(p)] for p in preds]
                confidences = [float(np.max(p)) for p in preds]
                df['cleaned'] = cleaned
                df['pred_label'] = pred_labels
                df['confidence'] = confidences
                st.success("Predictions complete.")
                st.dataframe(df[['tweet','cleaned','pred_label','confidence']])
                # show aggregated counts
                st.write("Counts by label:")
                st.bar_chart(df['pred_label'].value_counts())
                # export
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download results CSV", csv, file_name="predictions.csv", mime="text/csv")
else:
    tweet = st.text_area("Enter tweet text (one tweet).", height=150)
    if st.button("Analyze"):
        if not tweet.strip():
            st.warning("Please enter a tweet.")
        else:
            preds, cleaned = predict_texts([tweet], model, tokenizer, maxlen=model_maxlen)
            p = preds[0]
            idx = int(np.argmax(p))
            label = labels[idx] if idx < len(labels) else f"label_{idx}"
            confidence = float(np.max(p))
            st.markdown(f"**Predicted sentiment:** `{label}`  —  **confidence:** {confidence:.3f}")
            # highlight if above threshold
            if confidence >= show_conf_threshold:
                st.success("High confidence prediction")
            else:
                st.info("Prediction has low/medium confidence — consider manual review.")
            # recommendations
            rec = RECOMMENDATIONS.get(label, None)
            if rec:
                st.subheader("Recommended stakeholders")
                st.write(", ".join(rec['stakeholders']))
                st.subheader("Suggested immediate actions")
                for i, a in enumerate(rec['actions'], 1):
                    st.write(f"{i}. {a}")
                #st.subheader("Message template (editable)")
                #template = rec.get('message_template', "")
                #filled = template.format(location="[location]", date="[date]", evidence="[evidence]", details="[details]", description="[description]", phone="[contact]")
                #msg = st.text_area("Suggested message to send", value=filled, height=120)
                #st.download_button("Copy message", msg, file_name="recommendation.txt", mime="text/plain")
            else:
                st.write("No recommendation template for this label. Edit `RECOMMENDATIONS` in the app code to add one.")
