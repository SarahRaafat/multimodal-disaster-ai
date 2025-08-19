#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

def softmax1(logits):
    return torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()

def predict_probs(df, text_col, model_dir, max_len=160, batch_size=32, device="cpu"):
    tok = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            enc = tok(batch[text_col].astype(str).tolist(),
                      truncation=True, padding=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs.append(softmax1(logits))
    return np.concatenate(probs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", type=str, default="text_data/test.csv")
    ap.add_argument("--model-dir", type=str, default="artifacts/distilbert")  # or artifacts/distilbert_ft
    ap.add_argument("--text-col", type=str, default="text")
    ap.add_argument("--lat-col", type=str, default="lat")
    ap.add_argument("--lon-col", type=str, default="lon")
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Decision threshold; if omitted, tries to read best_threshold.json from model-dir, else 0.5")
    ap.add_argument("--out-html", type=str, default="docs/map.html")
    args = ap.parse_args()

    csv_path = Path(args.split_csv)
    df = pd.read_csv(csv_path)

    # Validate expected columns
    for c in [args.text_col, args.lat_col, args.lon_col]:
        if c not in df.columns:
            raise SystemExit(f"[error] '{c}' column not found in {csv_path}. "
                             f"Add lat/lon columns or pass --lat-col/--lon-col to match your file.")

    # Keep only rows with valid coords
    df = df.copy()
    df = df[np.isfinite(df[args.lat_col]) & np.isfinite(df[args.lon_col])]
    if len(df) == 0:
        raise SystemExit("[error] No rows with valid lat/lon in the chosen split.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Threshold
    thr = args.threshold
    if thr is None:
        try:
            d = json.loads((Path(args.model_dir)/"best_threshold.json").read_text())
            thr = float(d.get("threshold", 0.5))
        except Exception:
            thr = 0.5

    # Predict
    df["p_disaster"] = predict_probs(df, args.text_col, args.model_dir,
                                     max_len=args.max_len, batch_size=args.batch_size, device=device)
    df["pred"] = (df["p_disaster"] >= thr).map({True: "disaster", False: "non_disaster"})

    # Center map at mean lat/lon
    lat0 = float(df[args.lat_col].mean()); lon0 = float(df[args.lon_col].mean())
    m = folium.Map(location=[lat0, lon0], zoom_start=4, tiles="cartodbpositron")
    mc = MarkerCluster().add_to(m)

    # Color by prediction; size by confidence
    def color_row(p):
        return "#d62728" if p >= thr else "#1f77b4"
    def radius_row(p):
        return 6 + int(6 * abs(p - 0.5) * 2)  # 6..18 based on confidence

    for _, row in df.iterrows():
        lat, lon = float(row[args.lat_col]), float(row[args.lon_col])
        p = float(row["p_disaster"])
        txt = str(row[args.text_col])[:240].replace("\n", " ")
        pred = "disaster" if p >= thr else "non_disaster"
        folium.CircleMarker(
            location=(lat, lon),
            radius=radius_row(p),
            color=color_row(p),
            fill=True, fill_color=color_row(p), fill_opacity=0.8,
            tooltip=f"{pred} • p={p:.2f}",
            popup=folium.Popup(html=f"<b>{pred}</b> (p={p:.2f})<br/><hr/>{txt}", max_width=360),
        ).add_to(mc)

    out = Path(args.out_html); out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"[saved] {out}  (open in a browser)")

if __name__ == "__main__":
    main()
