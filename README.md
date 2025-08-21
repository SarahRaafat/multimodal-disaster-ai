# 🌍 Multimodal Disaster Response AI

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multimodal-disaster-ai-6xk9tytctynxctwld29qfn.streamlit.app/)

[![CI](https://github.com/<USER>/<REPO>/actions/workflows/ci.yml/badge.svg)](https://github.com/<USER>/<REPO>/actions/workflows/ci.yml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Coverage](https://img.shields.io/badge/coverage-run%20locally-informational)](#)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-FFCA28?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![codecov](https://codecov.io/gh/<USER>/<REPO>/branch/master/graph/badge.svg)](https://codecov.io/gh/<USER>/<REPO>)


> **Proof-of-concept AI system for flood disaster detection using both text (tweets) and images (drone photos).**

This project demonstrates how **Natural Language Processing (NLP)** and **Computer Vision (CV)** can be combined into a **multimodal AI pipeline** for disaster response scenarios.  
It was developed as a **portfolio-quality project** to showcase practical data science and engineering skills for internship applications (e.g., Google STEP, AI Residency).

---

## ✨ Features
- 📝 **Text Classification** – Distinguish disaster vs. non-disaster tweets (DistilBERT).
- 🖼️ **Image Classification** – Distinguish flooded vs. non-flooded drone photos (ResNet18).
- 🔗 **Fusion Model** – Combine text + image embeddings for improved accuracy.
- 🎯 **Balanced Datasets** – Stratified, deduplicated, reproducible splits.
- 🔒 **Reproducibility** – Config + seed utilities for deterministic runs.
- 💻 **Interactive Demo** – Streamlit app for hands-on exploration.

---

## ⚙️ How It Works
<p align="center">
  <img src="docs/how_it_works.png" width="600">
</p>

1. **Input**: Tweets and drone images.  
2. **Preprocessing**: Tokenization (text) + augmentation (images).  
3. **Models**: DistilBERT for text, ResNet18 for images.  
4. **Fusion**: Embeddings combined via MLP / late fusion.  
5. **Output**: Disaster vs. non-disaster prediction.  

---

## 🚀 Demo

Run locally:
```bash
streamlit run app.py

### Demo Screenshots
Single-sample demo:  
(docs/demo_single.png)

Batch demo:  
(docs/demo_batch.png)

## 📊 Results (Test Set)

| Model             | Accuracy | Precision | Recall | F1   |
|------------------ |----------|-----------|--------|------|
| Text (DistilBERT) | 81.2%    | 0.795     | 0.758  | 0.776 |
| Image (ResNet18)  | 96.6%    | 0.948     | 0.986  | 0.967 |
| Fusion (MLP)      | 83.9%    | 0.860     | 0.754  | 0.803 |
| Fusion (Late)     | 83.2%    | 0.833     | 0.769  | 0.800 |


### 🌍 Geo-Map Demo
We added a small visualization of model predictions on a map.  
*Note: coordinates are synthetic for demo purposes only.*  

(docs/map.png)



## 🔍 Interpretability

Text (DistilBERT – Saliency)

Highlights most influential tokens.

Disaster predictions highlight “storm”, “fire”, “drowned”.

False positives: sensational news/policy headlines.

👉 Explore: docs/interpretability/text_saliency.html.

Images (ResNet18 – Grad-CAM)

Heatmaps show regions driving predictions.

True positives: flooded roads, waterlines strongly activated.

False positives: wet asphalt / reflections confused as floods.

<p align="center"> <img src="docs/interpretability/gradcam_examples/gradcam_image_155.png" width="400"> <img src="docs/interpretability/gradcam_examples/gradcam_image_410.png" width="400"> </p>


🧪 Stress Tests & Limitations

Observed failure modes:

Figurative text (“concert was fire”, “song Hurricane”) – lower confidence.

Edge-case images (swimming pools, wet roads, reflections).

Borderline uncertainty – some non-disaster cases yield moderate disaster probability.

➡️ More in docs/limitations.md.

📜 License

MIT License © 2025 Sarah