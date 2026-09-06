"""Iris classification demo — öğrenci demosu."""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Iris Demo", page_icon="🌸")
st.title("🌸 Iris Sınıflandırma")

st.markdown(
    "Bu sayfada Iris modelimi deniyorum. Notebook'ta Random Forest eğittim ve "
    "`classification_iris.joblib` olarak kaydettim."
)

def _model(name):
    here = Path(__file__).resolve()
    for p in (here.parents[1] / "models" / name, here.parents[2] / "models" / name):
        if p.exists():
            return p
    return here.parents[2] / "models" / name

model_path = _model("classification_iris.joblib")

sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

if st.button("Tahmin et"):
    if not model_path.exists():
        st.error(f"Model yok: {model_path.name}")
    else:
        import joblib

        model = joblib.load(model_path)
        pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        st.success(f"Tahmin: **{pred}**")
