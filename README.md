# 14 — Specialize in Data Science

Her konudan 3 proje. Veriler orijinal setler.

Day2 / Day5 / Day14 ders kodunu kullandim:
- gorsellestirme: matplotlib, seaborn, plotly (model yok)
- deep learning: Keras Sequential, Conv2D, LSTM
- AI agents: OpenRouter + ollama + ChromaDB + arxiv

## Kaynaklar

- https://python.plainenglish.io/85-data-science-projects-c03c8750599e
- https://medium.com/coders-camp/230-machine-learning-projects-with-python-5d0c7abf8265
- Cogu csv: https://github.com/amankharwal/Website-data

## Projeler (30/30)

| # | Konu | Proje 1 | Proje 2 | Proje 3 |
|---|------|---------|---------|---------|
| 01 | Regression | Car Price | Student Marks | Insurance |
| 02 | Classification | Iris | SMS Spam | German Credit |
| 03 | Clustering | CC GENERAL | Mall Customers | Spotify |
| 04 | Computer Vision | OpenCV sayim | Maske foto | Digits |
| 05 | NLP | Twitter class | Fake/Real news | NER |
| 06 | Recommendation | Streaming movies | Goodbooks | Netflix filter |
| 07 | Time Series | AAPL | weatherHistory | Site trafigi |
| 08 | Visualization | Streaming EDA | Uber Sep 2014 | IPL 2022 |
| 09 | Deep Learning | MNIST CNN | AAPL LSTM | Flipkart Keras |
| 10 | AI Agents | LLM chatbot | Chroma RAG | arxiv+news agent |

Agent notebooklari icin `.env` icine `OPENROUTER_API_KEY` yaz. Yoksa ollama hucresi (dersdeki `llama3.2-vision`) calisir.

Keras notebooklari Day5/Day7 ile ayni import: `tensorflow.keras`. Colab'da da acilir.

## Streamlit

```bash
streamlit run streamlit_app/app.py
```

Hugging Face Space: https://huggingface.co/spaces/kemalserkany/specialize-in-data-science
