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

## Embeddings
Saved to artifacts/embeddings:
- text_{train,val,test}.npz → (N, 768)
- image_{train,val,test}.npz → (N, 512)

## Fusion — Early MLP (Embeddings → MLP)
Settings: 1280-d input [768 text CLS; 512 image penultimate], MLP 1280→512→2, AdamW lr=1e-3, wd=1e-2, dropout=0.2, class weights on.

- concat_naive:  Val F1=0.821 • Test Acc=0.839 • Test F1=0.803
- text_only_pad: Val F1=0.823 • Test Acc=0.810 • Test F1=0.774
- image_only_pad: Val F1=0.987 • Test Acc=0.973 • Test F1=0.973

Notes:
- Datasets are not truly paired; concat_naive uses index-based alignment (ablation).
- Fusion improves over text-only, but image-only remains best on this data.
- We will use late-fusion + calibration next to target parity with image while improving calibration.
