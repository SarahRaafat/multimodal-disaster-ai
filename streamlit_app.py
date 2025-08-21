# app.py
import os
import json
import zipfile
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import matplotlib.pyplot as plt

# ---------------- Config / Paths ----------------
DISTIL_DIR = Path("artifacts/distilbert")
RESNET_CKPT = Path("artifacts/resnet18/resnet18.pt")
FUSION_DIR = Path("artifacts/fusion")  # for alpha.json (optional)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
MAX_LEN = 160


# ---------------- Load Models ----------------
@st.cache_resource
def load_text_model():
    tok = DistilBertTokenizerFast.from_pretrained(DISTIL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(DISTIL_DIR).to(DEVICE)
    model.eval()
    return tok, model


@st.cache_resource
def load_image_model():
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 2)
    state = torch.load(RESNET_CKPT, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    model.eval()
    tf = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    )
    return model, tf


@st.cache_resource
def load_alpha_threshold():
    alpha, thr = 0.5, 0.5
    try:
        d = json.loads((FUSION_DIR / "alpha.json").read_text())
        alpha = float(d.get("alpha", alpha))
        thr = float(d.get("threshold", thr))
    except Exception:
        pass
    return alpha, thr


# ---------------- Inference helpers ----------------
def softmax1(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


def predict_text_proba(tok, model, text: str) -> float:
    enc = tok(
        [text], truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    return float(softmax1(logits)[0])


def predict_image_proba(model, tf, image: Image.Image) -> float:
    x = tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
    return float(softmax1(logits)[0])


def fuse_probs(p_text: float, p_image: float, alpha: float) -> float:
    return float(alpha * p_text + (1.0 - alpha) * p_image)


def label_from_prob(p: float, thr: float) -> str:
    return "disaster" if p >= thr else "non_disaster"


def prob_bar_chart(p_text, p_image, p_fused):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(["Text", "Image", "Fused"], [p_text, p_image, p_fused])
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(disaster)")
    fig.tight_layout()
    st.pyplot(fig)


# ---------------- UI ----------------
st.set_page_config(
    page_title="Multimodal Disaster AI", page_icon="🌊", layout="centered"
)
st.title("🌍🔥 Multimodal Disaster Response — Demo")
st.caption(
    f"Device: **{DEVICE}** • DistilBERT + ResNet18 • Late-fusion α,thr from artifacts/fusion/alpha.json"
)

alpha_default, thr_default = load_alpha_threshold()

tab1, tab2 = st.tabs(["🔹 Single Sample", "📦 Batch"])

# ---------- Single Sample ----------
with tab1:
    st.subheader("Single Sample")
    col1, col2 = st.columns(2)
    with col1:
        text_input = st.text_area(
            "Text (optional)", height=140, placeholder="Enter a tweet or message..."
        )
    with col2:
        img_file = st.file_uploader(
            "Image (optional)", type=["jpg", "jpeg", "png", "webp", "bmp"]
        )

    alpha = st.slider(
        "Fusion weight α (text vs image)",
        0.0,
        1.0,
        value=float(alpha_default),
        step=0.05,
    )
    thr = st.slider(
        "Decision threshold", 0.05, 0.95, value=float(thr_default), step=0.01
    )

    if st.button("Predict", type="primary"):
        tok, text_model = load_text_model()
        img_model, img_tf = load_image_model()

        p_text = None
        p_image = None

        if text_input.strip():
            p_text = predict_text_proba(tok, text_model, text_input)
        if img_file is not None:
            image = Image.open(img_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            p_image = predict_image_proba(img_model, img_tf, image)

        if (p_text is None) and (p_image is None):
            st.warning("Please provide text and/or an image.")
        else:
            if p_text is None:
                p_text = 0.0
            if p_image is None:
                p_image = 0.0
            p_fused = fuse_probs(p_text, p_image, alpha)

            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Text p(disaster)", f"{p_text:.3f}", help="DistilBERT")
                st.write("**Pred:**", label_from_prob(p_text, thr))
            with colB:
                st.metric("Image p(disaster)", f"{p_image:.3f}", help="ResNet18")
                st.write("**Pred:**", label_from_prob(p_image, thr))
            with colC:
                st.metric(
                    "Fused p(disaster)", f"{p_fused:.3f}", help="α·text + (1-α)·image"
                )
                st.write("**Pred:**", label_from_prob(p_fused, thr))

            prob_bar_chart(p_text, p_image, p_fused)

# ---------- Batch ----------
with tab2:
    st.subheader("Batch Inference")
    st.write(
        "Upload a CSV with columns: **text** (optional) and **image** (optional path/filename). Optionally upload a **ZIP** of images whose filenames match the `image` column."
    )

    csv_file = st.file_uploader("CSV", type=["csv"], key="csv")
    zip_file = st.file_uploader("Images ZIP (optional)", type=["zip"], key="zip")

    alpha_b = st.slider(
        "Fusion weight α (text vs image) [batch]",
        0.0,
        1.0,
        value=float(alpha_default),
        step=0.05,
        key="alpha_b",
    )
    thr_b = st.slider(
        "Decision threshold [batch]",
        0.05,
        0.95,
        value=float(thr_default),
        step=0.01,
        key="thr_b",
    )

    if st.button("Run Batch", type="primary", key="runbatch") and csv_file is not None:
        df = pd.read_csv(csv_file)
        if ("text" not in df.columns) and ("image" not in df.columns):
            st.error("CSV must contain at least one of: text, image")
        else:
            tok, text_model = load_text_model()
            img_model, img_tf = load_image_model()

            temp_dir = None
            img_lookup = {}
            if zip_file is not None:
                temp_dir = tempfile.TemporaryDirectory()
                with zipfile.ZipFile(zip_file) as z:
                    z.extractall(temp_dir.name)
                # build filename->fullpath map (basename only)
                for root, _, files in os.walk(temp_dir.name):
                    for f in files:
                        img_lookup[f] = os.path.join(root, f)

            probs_text, probs_image, probs_fused, preds = [], [], [], []
            for _, row in df.iterrows():
                t = (
                    str(row["text"])
                    if "text" in df.columns and not pd.isna(row["text"])
                    else None
                )
                img_path = None
                if "image" in df.columns and not pd.isna(row["image"]):
                    cand = str(row["image"])
                    # prefer ZIP-extracted match by basename
                    if cand in img_lookup:
                        img_path = img_lookup[cand]
                    elif os.path.exists(cand):
                        img_path = cand

                pt = 0.0
                pi = 0.0
                if t and t.strip():
                    pt = predict_text_proba(tok, text_model, t)
                if img_path:
                    try:
                        img = Image.open(img_path)
                        pi = predict_image_proba(img_model, img_tf, img)
                    except Exception:
                        pi = 0.0

                pf = fuse_probs(pt, pi, alpha_b)
                probs_text.append(pt)
                probs_image.append(pi)
                probs_fused.append(pf)
                preds.append(label_from_prob(pf, thr_b))

            df_out = df.copy()
            df_out["p_text"] = probs_text
            df_out["p_image"] = probs_image
            df_out["p_fused"] = probs_fused
            df_out["pred_fused"] = preds

            st.dataframe(df_out.head(25))
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )

            if temp_dir is not None:
                temp_dir.cleanup()

st.caption("© Multimodal Disaster Response AI — demo for internship application")
