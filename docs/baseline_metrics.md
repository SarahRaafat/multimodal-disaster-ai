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
