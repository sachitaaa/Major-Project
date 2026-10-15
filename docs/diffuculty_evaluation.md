   # Difficulty prediction baseline

   `train_improved.py` checks how well text features predict the LLM-assigned
   difficulty label on 211 questions (`final2/master_dataset_final.csv`).

   Run: `python train_improved.py`

   Method: stratified 5-fold CV repeated over 5 shuffles, scaling inside a
   pipeline, quadratic weighted kappa.

   | Model | Accuracy | Macro-F1 | Weighted kappa |
   |---|---|---|---|
   | Baseline (always Easy) | 0.512 | 0.226 | 0.000 |
   | Logistic regression | 0.557 | 0.481 | 0.391 |
   | Random forest | 0.589 | 0.517 | 0.395 |

   The three difficulty labels in the dataset agree only weakly
   (kappa 0.18 to 0.32).

   Limitations: labels come from an LLM or a formula, only 12 questions
   are "Hard", and the sample is small.