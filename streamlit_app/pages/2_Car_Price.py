from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Car Price Demo", page_icon="🚗")
st.title("🚗 Araç Fiyat Tahmini")

st.markdown("CarPrice.csv ile eğittiğim Random Forest.")

def _model(name):
    here = Path(__file__).resolve()
    for p in (here.parents[1] / "models" / name, here.parents[2] / "models" / name):
        if p.exists():
            return p
    return here.parents[2] / "models" / name

model_path = _model("regression_car_price.joblib")

enginesize = st.slider("Motor hacmi (cc)", 60, 330, 120)
horsepower = st.slider("Beygir", 48, 288, 110)
citympg = st.slider("Şehir içi mpg", 13, 49, 25)
curbweight = st.slider("Ağırlık", 1488, 4066, 2550)
fueltype = st.selectbox("Yakıt", ["gas", "diesel"])
carbody = st.selectbox("Kasa", ["sedan", "hatchback", "wagon", "hardtop", "convertible"])

if st.button("Fiyat tahmin et"):
    if not model_path.exists():
        st.error("Model bulunamadı.")
    else:
        import joblib
        import pandas as pd

        model = joblib.load(model_path)
        row = pd.DataFrame(
            [
                {
                    "enginesize": enginesize,
                    "horsepower": horsepower,
                    "citympg": citympg,
                    "curbweight": curbweight,
                    "fueltype": fueltype,
                    "carbody": carbody,
                }
            ]
        )
        row = pd.get_dummies(row, drop_first=True)
        cols = getattr(model, "feature_names_in_", None)
        if cols is not None:
            row = row.reindex(columns=cols, fill_value=0)
        pred = model.predict(row)[0]
        st.success(f"Tahmini fiyat: **{pred:,.0f}**")
