# Fine-tuning Metrics

## Text — DistilBERT (unfreeze top-2 layers + head)
Args: epochs=4, batch=16, head LR=2e-5, base LR=5e-6, max_len=192, class-weights, grad clip=1.0  
**Val (best):** Acc 81.0% | F1 0.779  
**Test (thr≈0.46):** Acc **80.8%** | Prec **0.774** | Rec **0.781** | **F1 0.778**  
**Δ vs baseline (F1 0.776):** **+0.0016**  
Notes: Small lift. If you want to try squeezing more: epochs=5, head LR=3e-5, base LR≈7e-6, or max_len=256. Also try without class weights as an ablation.

## Image — ResNet18 (unfreeze layer4 after warmup=2)
Args: epochs=6, head LR=1e-3, layer4 LR=1e-4, wd=1e-4  
**Val (best):** Acc 98.7% | **F1 0.986**  
**Test:** Acc **97.99%** | Prec **0.9863** | Rec **0.9730** | **F1 0.9796**  
**Δ vs baseline (F1 0.9669):** **+0.0127**  
**Δ vs baseline (Acc 96.64%):** **+1.34 pts**  
Notes: Clear benefit from fine-tuning the last block. Confusion matrix improved to [[TN 72, FP 2], [FN 1, TP 74]].
