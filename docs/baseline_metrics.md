# Baseline Metrics – Multimodal Disaster Response AI

## Text — DistilBERT (class-weighted, best ckpt, threshold tuned)
Val (best epoch=1): Acc 85.4%, F1 0.828
Test (thr=0.58): Acc 81.2%, Prec 0.795, Rec 0.758, F1 0.776
Confusion matrix: [[TN 243, FP 42], [FN 52, TP 163]]
Notes: CPU, 3 epochs, class weights from train counts; threshold tuned on val to maximize F1.

## Image — ResNet18 (head-only, ImageNet weights)
Val (best): F1 0.961
Test: Acc 96.6%, Prec 0.948, Rec 0.986, F1 0.967
Confusion matrix: [[TN 73, FP 1], [FN 4, TP 71]]
Notes: CPU, light aug (flip/rotate/color jitter), 4 epochs.

## Error Analysis (Test)

### Text (DistilBERT, thr=0.58 tuned on val)
- Test: Prec 0.795 • Rec 0.758 • F1 0.776
- Common FP patterns: headlines/sensational words (“storm vortex”, “catastrophe”) without real incidents; policy/insurance/legal posts.
- Common FN patterns: short/slang/ambiguous phrasing; jokes/sarcasm; benign “storm/free” mentions; non-incident uses of “fire/slaughter”.

### Image (ResNet18, thr≈0.30 tuned on val)
- Test: Prec 0.938 • Rec 1.000 • F1 0.968
- No FNs observed on test; a handful of FPs (see grid).
- FP patterns: wet/reflective surfaces; roads/waterlines that resemble flooding.

### Fusion (Late, α≈0.50, thr≈0.46)
- Test: Acc 83.2% • Prec 0.833 • Rec 0.769 • F1 0.800
- Notes: fusion uses synthetic index pairing (unpaired datasets); retains solid performance but calibration mainly tied to image stream.

Artifacts:
- `docs/errors/image_fp_grid.png`
- `docs/errors/error_summary.json`

## Embeddings
Saved to artifacts/embeddings:
- text_{train,val,test}.npz → (N, 768)
- image_{train,val,test}.npz → (N, 512)

## Fusion Baselines

### Early Fusion (Concat → MLP)
- Test Accuracy: 83.9%
- Precision: 0.860
- Recall: 0.754
- F1: 0.803
- Notes: Uses joint embeddings (768 text + 512 image). Outperformed text-only, slightly below image-only.

### Late Fusion (Calibrated Probs)
- Validation: best α=0.50, threshold=0.46, F1=0.855
- Test Results:
  - Accuracy: 83.2%
  - Precision: 0.833
  - Recall: 0.769
  - F1: 0.800
  - Confusion Matrix: [[74,10],[15,50]]
- Calibration (ECE ↓ better, Brier ↓ better):
  - Text: ECE=0.443, Brier=0.130
  - Image: ECE=0.146, Brier=0.246
  - Fused: ECE=0.399, Brier=0.152
- Notes: Image model dominated performance; fusion matched image on F1 but improved Brier score. High label mismatches (synthetic pairing) limit interpretation.