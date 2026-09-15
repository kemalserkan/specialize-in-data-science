from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Tweet Demo", page_icon="💬")
st.title("💬 Tweet Sınıflandırma")

st.markdown("twitter.csv modeli: 0 hate, 1 offensive, 2 neither")

def _model(name):
    here = Path(__file__).resolve()
    for p in (here.parents[1] / "models" / name, here.parents[2] / "models" / name):
        if p.exists():
            return p
    return here.parents[2] / "models" / name

model_path = _model("nlp_twitter_sentiment.joblib")
text = st.text_area("Tweet", "This is a normal tweet about the weather.")
isim = {0: "hate", 1: "offensive", 2: "neither"}

if st.button("Tahmin"):
    if not model_path.exists():
        st.error("Model bulunamadı.")
    else:
        import joblib

        pipe = joblib.load(model_path)
        pred = pipe.predict([text])[0]
        st.success(f"Sınıf: **{isim.get(pred, pred)}**")
