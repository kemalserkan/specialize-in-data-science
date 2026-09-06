from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Chatbot Agent", page_icon="🤖")
st.title("🤖 Simple Chatbot Agent")

st.markdown("intents.json + LinearSVC")

def _model(name):
    here = Path(__file__).resolve()
    for p in (here.parents[1] / "models" / name, here.parents[2] / "models" / name):
        if p.exists():
            return p
    return here.parents[2] / "models" / name

model_path = _model("agent_chatbot.joblib")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_msg = st.chat_input("Mesajını yaz...")


def reply(message, bot):
    import random

    if not isinstance(bot, dict):
        return "Model formatı farklı."
    t = (message or "").lower()
    kurallar = [
        ("goodbye", ["bye", "goodbye", "see you", "görüşürüz"]),
        ("thanks", ["thank", "sağol", "teşekkür"]),
        ("items", ["sell", "item", "product", "coffee", "ne sat"]),
        ("payments", ["pay", "card", "visa", "paypal", "ödeme"]),
        ("delivery", ["deliver", "shipping", "kargo", "teslim"]),
        ("funny", ["joke", "funny", "şaka"]),
        ("greeting", ["hi", "hey", "hello", "selam", "merhaba"]),
    ]
    tag = None
    for ad, keys in kurallar:
        if any(k in t for k in keys):
            tag = ad
            break
    if tag is None:
        pipe = bot.get("pipe")
        tag = pipe.predict([message])[0] if pipe is not None else None
    intents = bot.get("intents", {})
    for it in intents.get("intents", []):
        if it.get("tag") == tag:
            resp = it.get("responses", ["ok"])
            return random.choice(resp) if isinstance(resp, list) else str(resp)
    return "Bunu tam anlayamadım. hello / what do you sell? / joke / thanks / bye dene."


if user_msg:
    st.session_state.chat_history.append(("user", user_msg))
    if model_path.exists():
        import joblib

        bot = joblib.load(model_path)
        st.session_state.chat_history.append(("bot", reply(user_msg, bot)))
    else:
        st.session_state.chat_history.append(("bot", "Model bulunamadı."))

for role, msg in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.write(msg)
