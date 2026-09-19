"""
Streamlit UI for the SVM-Based Email Spam Detector.

Run with:
    streamlit run app.py

Requires mail_data.csv in the same folder (Category, Message columns).
The model trains once on startup (cached) -- no files are saved or loaded
from disk, consistent with the rest of the project.
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")


@st.cache_resource(show_spinner="Training model on startup...")
def load_model():
    df = pd.read_csv("mail_data.csv")
    data = df.where((pd.notnull(df)), " ")
    data["Category"] = data["Category"].map({"spam": 0, "ham": 1})

    X = data["Message"]
    Y = data["Category"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=3, stratify=Y
    )

    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
    X_train_features = vectorizer.fit_transform(X_train)

    # Best model from the project comparison: Linear SVM (spam F1 = 0.9514, AUC = 0.9944)
    model = SVC(kernel="linear", class_weight="balanced", probability=True)
    model.fit(X_train_features, Y_train)

    return model, vectorizer


model, vectorizer = load_model()

st.title("📧 SVM-Based Email Spam Detector")
st.write(
    "Paste a message below to check whether it's **spam** or **ham** (legitimate). "
    "Powered by a linear SVM trained on TF-IDF features -- the best-performing model "
    "from the project comparison."
)

message = st.text_area(
    "Message text",
    height=150,
    placeholder="Type or paste an email or SMS message here...",
)

if st.button("Check Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        vec = vectorizer.transform([message])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        spam_prob, ham_prob = proba[0], proba[1]

        if pred == 1:
            st.success("✅ This looks like **HAM** (legitimate)")
            st.metric("Confidence (ham)", f"{ham_prob:.1%}")
        else:
            st.error("🚫 This looks like **SPAM**")
            st.metric("Confidence (spam)", f"{spam_prob:.1%}")

        with st.expander("See full probability breakdown"):
            st.write(f"Ham probability: {ham_prob:.4f}")
            st.write(f"Spam probability: {spam_prob:.4f}")

st.divider()
st.caption(
    "Model: Linear SVM on TF-IDF features · Trained fresh each time the app starts "
    "(no saved model files) · Test-set performance: spam F1 = 0.9514, ROC-AUC = 0.9944"
)
