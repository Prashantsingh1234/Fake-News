import os
import re
import pickle
import torch
import numpy as np
import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from transformers import pipeline
generator = pipeline("text-generation", model="gpt2", framework="pt")

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ── DOWNLOAD NLTK RESOURCES ───────────────────────────────────
nltk.download("stopwords")
nltk.download("vader_lexicon")

# ── TEXT PREPROCESSING ────────────────────────────────────────
porter = PorterStemmer()
def stemming(text: str) -> str:
    # keep only letters, lowercase, split
    tokens = re.sub("[^a-zA-Z]", " ", text).lower().split()
    # remove stopwords and stem
    filtered = [porter.stem(w) for w in tokens if w not in stopwords.words("english")]
    return " ".join(filtered)

# ── FAKE‑NEWS MODEL SETUP ─────────────────────────────────────
MODEL_FILE      = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    # load pre‑trained
    with open(MODEL_FILE,      "rb") as f: model      = pickle.load(f)
    with open(VECTORIZER_FILE, "rb") as f: vectorizer = pickle.load(f)
else:
    # train from CSV
    df = pd.read_csv(r"train.csv").fillna("")
    df["content"] = (df["author"] + " " + df["title"]).apply(stemming)
    X_all = TfidfVectorizer().fit_transform(df["content"])
    y_all = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=2
    )
    model      = LogisticRegression().fit(X_train, y_train)
    vectorizer = TfidfVectorizer().fit(df["content"])

    # save
    with open(MODEL_FILE,      "wb") as f: pickle.dump(model,      f)
    with open(VECTORIZER_FILE, "wb") as f: pickle.dump(vectorizer, f)

# ── SENTIMENT ANALYZER ────────────────────────────────────────
sia = SentimentIntensityAnalyzer()

# ── MBTI CLASSIFIER ───────────────────────────────────────────
# Uses a DistilBERT‐based MBTI model from Hugging Face
mbti_pipe = pipeline(
    "text-classification",
    model="parka735/mbti-classifier",
    tokenizer="parka735/mbti-classifier",
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512
)

# ── STREAMLIT UI ──────────────────────────────────────────────
st.set_page_config(page_title="📰 Fake‑News + Sentiment + MBTI Detector")
st.title("📰 Sentimental and Fake-News Detector")

author_input = st.text_input("Author Name")
title_input  = st.text_area("News Title")

if st.button("Predict"):
    if not author_input or not title_input:
        st.warning("⚠️ Please enter both Author Name and News Title.")
    else:
        raw_text = f"{author_input} {title_input}"
        proc_text = stemming(raw_text)
        vect_text = vectorizer.transform([proc_text])

        # — Fake News Prediction —
        label = model.predict(vect_text)[0]
        if label == 0:
            st.success("✅ **Real** News")
        else:
            st.error("❌ **Fake** News")

        # — Sentiment Analysis —
        scores = sia.polarity_scores(raw_text)
        st.subheader("🔍 Sentiment Analysis")
        st.write(f"• Positive: {scores['pos']:.3f}")
        st.write(f"• Neutral : {scores['neu']:.3f}")
        st.write(f"• Negative: {scores['neg']:.3f}")
        st.write(f"• Compound: {scores['compound']:.3f}")
        # bar chart
        st.bar_chart(pd.DataFrame([scores]))

        # — MBTI Classification —
        #st.subheader("🧠 MBTI Personality Prediction")
        #mbti_scores = mbti_pipe(raw_text)[0]  # list of dicts: {"label": "INTJ", "score": 0.xxx}
        # convert to DataFrame, sort by score descending
        #df_mbti = pd.DataFrame(mbti_scores).sort_values("score", ascending=False)
        #df_mbti["score"] = df_mbti["score"].map(lambda x: f"{x:.3f}")
        #st.dataframe(df_mbti, use_container_width=True)


                # — MBTI Classification —
        st.subheader("🧠 MBTI Sentimental Prediction")
        mbti_scores = mbti_pipe(raw_text)[0]  # list of dicts: {"label":"INTJ","score":0.xxx}
        df_mbti = pd.DataFrame(mbti_scores).sort_values("score", ascending=False)

        # Keep a float copy for plotting
        df_mbti["score_float"] = df_mbti["score"]

        # Turn the original score into formatted strings
        df_mbti["score"] = df_mbti["score"].map(lambda x: f"{x:.3f}")

        # Show the table
        st.dataframe(df_mbti[["label", "score"]], use_container_width=True)

        # — MBTI Score Distribution Chart —
        st.subheader("📊 MBTI Score Distribution")
        chart_data = df_mbti.set_index("label")["score_float"]
        st.bar_chart(chart_data)
