# Baseline Metrics – Multimodal Disaster Response AI

## Text – DistilBERT (class-weighted)
- Acc: 81.0%
- Prec: 0.778
- Rec:  0.781
- F1:   0.780
- Notes: Best on epoch 1; weights balanced FN/FP.

## Image – ResNet18 (head-only, ImageNet weights)
- Val F1: 0.961 (best)
- Test: Acc 96.6%, Prec 0.948, Rec 0.986, F1 0.967
- Confusion matrix: [[73, 1], [4, 71]]
- Notes: Head-only training on CPU. Light aug (flip/rotate/color jitter).


These baselines are **not final models**.  
They serve as quick checks to confirm data loaders, preprocessing, and training pipelines work.
