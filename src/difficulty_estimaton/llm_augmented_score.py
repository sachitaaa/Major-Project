import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from scipy.stats import spearmanr

# Load the dataset
df = pd.read_csv('/home/sujal/Programming/MajorProject/evaluation/master_dataset.csv')

# Ensure the relevant columns exist
required = ['subject', 'self_reported_difficulty', 'llm_response_augmented_difficulty']
for col in required:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in CSV.")

# Drop rows with missing labels (if any)
df_clean = df[required].dropna()

# Map labels to numeric for correlation and kappa (Easy=1, Medium=2, Hard=3)
label_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
df_clean['self_num'] = df_clean['self_reported_difficulty'].map(label_map)
df_clean['llm_num'] = df_clean['llm_response_augmented_difficulty'].map(label_map)

# Remove any rows where mapping failed (should not happen)
df_clean = df_clean.dropna(subset=['self_num', 'llm_num'])

# Extract arrays
y_true = df_clean['self_num'].astype(int)
y_pred = df_clean['llm_num'].astype(int)
y_true_labels = df_clean['self_reported_difficulty']
y_pred_labels = df_clean['llm_response_augmented_difficulty']

# ---- Overall metrics ----
print("="*60)
print("Overall comparison: Self‑Reported vs. LLM‑Response‑Augmented Difficulty")
print("="*60)

# Confusion matrix (rows: self-reported, cols: llm-response-augmented)
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=['Easy', 'Medium', 'Hard'])
print("\nConfusion matrix (rows = Self‑Reported, cols = LLM‑Response‑Augmented):")
print(pd.DataFrame(cm, index=['Easy', 'Medium', 'Hard'], columns=['Easy', 'Medium', 'Hard']))

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {acc:.3f} ({acc*100:.1f}%)")

# Cohen's kappa (unweighted)
kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's κ (unweighted): {kappa:.3f}")

# Linear‑weighted kappa (same as in paper)
kappa_linear = cohen_kappa_score(y_true, y_pred, weights='linear')
print(f"Linear‑weighted κ: {kappa_linear:.3f}")

# Spearman rank correlation
rho, p_value = spearmanr(y_true, y_pred)
print(f"Spearman ρ: {rho:.3f} (p‑value: {p_value:.2e})")

# ---- Per‑subject breakdown ----
print("\n" + "="*60)
print("Per‑subject agreement (Self‑Reported vs. LLM‑Response‑Augmented)")
print("="*60)

subjects = df_clean['subject'].unique()
for subj in sorted(subjects):
    subj_df = df_clean[df_clean['subject'] == subj]
    y_true_sub = subj_df['self_num'].astype(int)
    y_pred_sub = subj_df['llm_num'].astype(int)
    acc_sub = accuracy_score(y_true_sub, y_pred_sub)
    kappa_sub = cohen_kappa_score(y_true_sub, y_pred_sub)
    print(f"{subj:20s}  Accuracy: {acc_sub:.3f} ({acc_sub*100:.1f}%)  κ: {kappa_sub:.3f}")

# Optional: print number of questions per subject
print("\nNumber of questions per subject:")
print(df_clean['subject'].value_counts().sort_index())