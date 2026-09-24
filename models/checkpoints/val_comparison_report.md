# Model Comparison Report (Val Set)

**Date:** 2026-09-25  
**Evaluation Split:** `val`  
**Dataset:** strict_clean_knee_osteoporosis (Validation n=134)  
**Primary Model Selection Metric:** Validation Macro F1-Score  

## Comparison Summary

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall | ROC-AUC | ECE | Epochs |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| custom_cnn | 0.7836 | 0.7841 | 0.7965 | 0.8049 | 0.8942 | 0.1010 | 35 |
| resnet18 | 0.7910 | 0.7913 | 0.8065 | 0.8109 | 0.9127 | 0.1196 | 36 |
| efficientnet_b0 | 0.7537 | 0.7542 | 0.7701 | 0.7738 | 0.8919 | 0.0507 | 27 |
| densenet121 | 0.7910 | 0.7912 | 0.7965 | 0.8017 | 0.9007 | 0.1205 | 53 |

## Model Selection Recommendation

**Top-performing model by Validation Macro F1:** `resnet18` (Val Macro F1 = 0.7913)

> [!IMPORTANT]
> All candidate comparisons above are performed STRICTLY on the **Validation Set (n=134)**.
> The **Held-Out Test Set (n=135)** remains completely untouched and blinded during model exploration.
> Test set evaluation should only be performed once the final architecture is locked and approved.
