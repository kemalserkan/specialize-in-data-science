"""
Specialize in Data Science — Streamlit demo hub.

Çalıştırma:
    streamlit run streamlit_app/app.py
"""

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

st.set_page_config(
    page_title="Specialize in Data Science",
    page_icon="📊",
    layout="wide",
)

st.title("Specialize in Data Science")
st.markdown("Eğittiğim modelleri buradan deniyorum. Soldaki sayfalardan aç.")

topics = [
    ("01_Regression", "Regression", 3),
    ("02_Classification", "Classification", 3),
    ("03_Clustering", "Clustering", 3),
    ("04_Computer_Vision", "Computer Vision", 3),
    ("05_NLP", "NLP", 3),
    ("06_Recommendation_Systems", "Recommendation Systems", 3),
    ("07_Time_Series", "Time Series", 3),
    ("08_Data_Visualization", "Data Visualization", 3),
    ("09_Deep_Learning", "Deep Learning", 3),
    ("10_AI_Agents", "AI Agents", 3),
]

st.subheader("Konu özeti")
cols = st.columns(2)
for i, (folder, label, n) in enumerate(topics):
    with cols[i % 2]:
        st.write(f"**{label}** (`{folder}`)")

st.subheader("Kayıtlı modeller")
model_files = sorted(
    p.name
    for p in MODELS_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".pkl", ".joblib", ".h5", ".pt", ".keras"}
) if MODELS_DIR.exists() else []

if model_files:
    st.write(model_files)
else:
    st.write("Model dosyası yok.")

st.caption("Hugging Face Space yayınlama: docs/HUGGINGFACE_DEPLOY.md")
