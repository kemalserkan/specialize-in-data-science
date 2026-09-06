from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
import spaces


@spaces.GPU
def _gpu_ok():
    return "ok"

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def load(name):
    p = MODELS / name
    if not p.exists():
        return None
    return joblib.load(p)


iris_m = load("classification_iris.joblib")
car_m = load("regression_car_price.joblib")
spam_m = load("classification_spam.joblib")
tweet_m = load("nlp_twitter_sentiment.joblib")
bot_m = load("agent_chatbot.joblib")


def iris_tahmin(sl, sw, pl, pw):
    if iris_m is None:
        return "Iris modeli yok."
    return str(iris_m.predict([[sl, sw, pl, pw]])[0])


def araba_tahmin(enginesize, horsepower, citympg, curbweight, fueltype, carbody):
    if car_m is None:
        return "Araç modeli yok."
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
    cols = getattr(car_m, "feature_names_in_", None)
    if cols is not None:
        row = row.reindex(columns=cols, fill_value=0)
    return f"{car_m.predict(row)[0]:,.0f}"


def spam_tahmin(text):
    if spam_m is None:
        return "Spam modeli yok."
    return str(spam_m.predict([text])[0])


def tweet_tahmin(text):
    if tweet_m is None:
        return "Tweet modeli yok."
    isim = {0: "hate", 1: "offensive", 2: "neither"}
    pred = tweet_m.predict([text])[0]
    return isim.get(pred, str(pred))


def _intent_tag(text):
    t = (text or "").lower()
    kurallar = [
        ("goodbye", ["bye", "goodbye", "see you", "görüşürüz", "hoşça kal"]),
        ("thanks", ["thank", "sağol", "tesekkur", "teşekkür"]),
        ("items", ["sell", "item", "product", "coffee", "tea", "ne sat", "ürün"]),
        ("payments", ["pay", "card", "visa", "paypal", "cash", "ödeme", "kredi"]),
        ("delivery", ["deliver", "shipping", "kargo", "teslim"]),
        ("funny", ["joke", "funny", "şaka", "espri"]),
        ("greeting", ["hi", "hey", "hello", "selam", "merhaba", "günaydın"]),
    ]
    for tag, keys in kurallar:
        if any(k in t for k in keys):
            return tag
    pipe = bot_m.get("pipe")
    if pipe is None:
        return None
    tag = pipe.predict([text])[0]
    if hasattr(pipe, "decision_function"):
        skor = pipe.decision_function([text])[0]
        # tek boyutlu veya çok sınıflı
        try:
            import numpy as np

            skor = np.asarray(skor).ravel()
            if skor.size > 1:
                s = np.sort(skor)
                if s[-1] - s[-2] < 0.15:
                    return None
        except Exception:
            pass
    return tag


def chatbot(message, history):
    import random

    if bot_m is None:
        return "Chatbot modeli yok."
    tag = _intent_tag(message)
    intents = bot_m.get("intents", {})
    for it in intents.get("intents", []):
        if it.get("tag") == tag:
            resp = it.get("responses", ["ok"])
            return random.choice(resp) if isinstance(resp, list) else str(resp)
    return (
        "Bunu tam anlayamadım. Şunları deneyebilirsin: hello, what do you sell?, "
        "do you take credit cards?, how long does delivery take?, tell me a joke, thanks, bye"
    )


with gr.Blocks(title="Specialize in Data Science") as demo:
    gr.Markdown(
        """
# Specialize in Data Science

Notebook'larda eğittiğim modellerin demosu.
Hugging Face artık yeni Space'te Streamlit SDK ve ücretsiz Docker vermiyor; aynı modelleri Gradio ile yayınladım.
"""
    )
    with gr.Tab("Iris"):
        gr.Interface(
            fn=iris_tahmin,
            inputs=[
                gr.Slider(4.0, 8.0, 5.1, label="Sepal length"),
                gr.Slider(2.0, 4.5, 3.5, label="Sepal width"),
                gr.Slider(1.0, 7.0, 1.4, label="Petal length"),
                gr.Slider(0.1, 2.5, 0.2, label="Petal width"),
            ],
            outputs="text",
            flagging_mode="never",
        )
    with gr.Tab("Araç fiyatı"):
        gr.Interface(
            fn=araba_tahmin,
            inputs=[
                gr.Slider(60, 330, 120, label="Motor (cc)"),
                gr.Slider(48, 288, 110, label="Beygir"),
                gr.Slider(13, 49, 25, label="Şehir mpg"),
                gr.Slider(1488, 4066, 2550, label="Ağırlık"),
                gr.Dropdown(["gas", "diesel"], value="gas", label="Yakıt"),
                gr.Dropdown(
                    ["sedan", "hatchback", "wagon", "hardtop", "convertible"],
                    value="sedan",
                    label="Kasa",
                ),
            ],
            outputs="text",
            flagging_mode="never",
        )
    with gr.Tab("Spam"):
        gr.Interface(
            fn=spam_tahmin,
            inputs=gr.Textbox(value="WINNER free prize call now", label="Mesaj"),
            outputs="text",
            flagging_mode="never",
        )
    with gr.Tab("Tweet"):
        gr.Interface(
            fn=tweet_tahmin,
            inputs=gr.Textbox(value="This is a normal tweet about the weather.", label="Tweet"),
            outputs="text",
            flagging_mode="never",
        )
    with gr.Tab("Chatbot"):
        gr.Markdown(
            "Örnek: `hello` · `what do you sell?` · `do you take credit cards?` · "
            "`how long does delivery take?` · `tell me a joke` · `thanks` · `bye`"
        )
        gr.ChatInterface(fn=chatbot)

if __name__ == "__main__":
    demo.launch()
