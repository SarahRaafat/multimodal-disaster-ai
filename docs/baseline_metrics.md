# Baseline Metrics – Multimodal Disaster Response AI

## Text – DistilBERT (class-weighted)
- Acc: 81.0%
- Prec: 0.778
- Rec:  0.781
- F1:   0.780
- Notes: Best on epoch 1; weights balanced FN/FP.


## Image (ResNet18 frozen backbone)
- Sanity check loss (1 batch): 1.37
- Notes: Confirms image pipeline works. Needs full training to report accuracy.


These baselines are **not final models**.  
They serve as quick checks to confirm data loaders, preprocessing, and training pipelines work.
