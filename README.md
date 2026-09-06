# 14 — Specialize in Data Science

Her konudan 3 proje seçtim. Verileri sentetik üretmedim, referanslardaki / bilinen orijinal setleri indirdim.
Her notebook’ta başlık, hedef, EDA, görselleştirme, boş veri, feature engineering, train/test, en az 3 model ve kısa sonuç var. Regression olanlarda feature importance + residual de çizdim.

## Kaynaklar

- https://python.plainenglish.io/85-data-science-projects-c03c8750599e
- https://medium.com/coders-camp/230-machine-learning-projects-with-python-5d0c7abf8265
- Çoğu csv: https://github.com/amankharwal/Website-data

## Projeler (30/30)

| # | Konu | Proje 1 | Proje 2 | Proje 3 |
|---|------|---------|---------|---------|
| 01 | Regression | Car Price | Student Marks | Insurance |
| 02 | Classification | Iris | SMS Spam | German Credit |
| 03 | Clustering | CC GENERAL | Mall Customers | Spotify |
| 04 | Computer Vision | OpenCV sayım | Maske fotoğraf | Digits |
| 05 | NLP | Twitter class | Fake/Real news | NER |
| 06 | Recommendation | Streaming movies | Goodbooks | Netflix filter |
| 07 | Time Series | AAPL | weatherHistory | Site trafiği |
| 08 | Visualization | Streaming EDA | Uber Sep 2014 | IPL 2022 |
| 09 | Deep Learning | Digits MLP | AAPL sequence | Flipkart |
| 10 | AI Agents | intents chatbot | BBC RAG | multi-tool |

Deep Learning’de TensorFlow takıldığı için sklearn MLP kullandım.

## Streamlit

```bash
streamlit run streamlit_app/app.py
```

Iris, araç fiyatı, spam, tweet, chatbot sayfaları var.

Hugging Face Space: https://huggingface.co/spaces/kemalserkany/specialize-in-data-science

Lokal Streamlit ayrı duruyor; HF ücretsiz hesapta Streamlit/Docker açılmadığı için Space’i Gradio ile yayınladım.
