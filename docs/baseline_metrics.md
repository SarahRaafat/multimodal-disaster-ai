# Baseline Metrics – Multimodal Disaster Response AI

## Text — DistilBERT (class-weighted, tuned threshold)
- Val best (0.50): Acc 85.4%, F1 0.828
- Val tuned threshold: 0.58 → F1 0.830
- Test (thr=0.58): Acc 81.2%, Prec 0.795, Rec 0.758, F1 0.776
- CM: [[TN 243, FP 42], [FN 52, TP 163]]
- Notes: CPU training, 3 epochs, class weights. Tuned threshold to maximize val F1.

## Image — ResNet18 (head-only, ImageNet weights)
- Val best: F1 0.961
- Test: Acc 96.6%, Prec 0.948, Rec 0.986, F1 0.967
- CM: [[73, 1], [4, 71]]
- Notes: CPU training, light aug (flip/rotate/color jitter).

These baselines are **not final models**.  
They serve as quick checks to confirm data loaders, preprocessing, and training pipelines work.
