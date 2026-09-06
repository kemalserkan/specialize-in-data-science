"""Spam detection demo."""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Spam Demo", page_icon="📧")
st.title("📧 Spam Detection")

st.markdown("SMS metnini yaz; TF-IDF + Logistic Regression modeli spam/ham tahmin etsin.")

def _model(name):
    here = Path(__file__).resolve()
    for p in (here.parents[1] / "models" / name, here.parents[2] / "models" / name):
        if p.exists():
            return p
    return here.parents[2] / "models" / name

model_path = _model("classification_spam.joblib")
text = st.text_area("Mesaj", "Tebrikler! 1000 TL kazandınız, hemen tıklayın: http://bit.ly/odul")

if st.button("Tahmin et"):
    if not model_path.exists():
        st.error("Model bulunamadı.")
    else:
        import joblib

        pipe = joblib.load(model_path)
        pred = pipe.predict([text])[0]
        st.success(f"Tahmin: **{pred}**")
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba([text])[0]
            st.write(dict(zip(pipe.classes_, [round(float(p), 3) for p in proba])))
