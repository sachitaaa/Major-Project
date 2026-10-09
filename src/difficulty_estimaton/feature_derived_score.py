import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from scipy.stats import spearmanr

df = pd.read_csv('/home/sujal/Programming/MajorProject/evaluation/master_dataset.csv')

required = ['subject', 'self_reported_difficulty', 'feature_derived_difficulty']
for col in required:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in CSV.")

df_clean = df[required].dropna()

label_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
df_clean['self_num'] = df_clean['self_reported_difficulty'].map(label_map)
df_clean['feat_num'] = df_clean['feature_derived_difficulty'].map(label_map)
df_clean = df_clean.dropna(subset=['self_num', 'feat_num'])

y_true = df_clean['self_num'].astype(int)
y_pred = df_clean['feat_num'].astype(int)
y_true_labels = df_clean['self_reported_difficulty']
y_pred_labels = df_clean['feature_derived_difficulty']

print("="*60)
print("Overall comparison: Self-Reported vs. Feature-Derived Difficulty")
print("="*60)
print(f"\nTotal N: {len(df_clean)}")

cm = confusion_matrix(y_true_labels, y_pred_labels, labels=['Easy', 'Medium', 'Hard'])
print("\nConfusion matrix (rows = Self-Reported, cols = Feature-Derived):")
print(pd.DataFrame(cm, index=['Easy', 'Medium', 'Hard'], columns=['Easy', 'Medium', 'Hard']))

acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {acc:.3f} ({acc*100:.1f}%)")

kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's kappa (unweighted): {kappa:.3f}")

kappa_linear = cohen_kappa_score(y_true, y_pred, weights='linear')
print(f"Linear-weighted kappa: {kappa_linear:.3f}")

rho, p_value = spearmanr(y_true, y_pred)
print(f"Spearman rho: {rho:.3f} (p-value: {p_value:.2e})")

print("\n" + "="*60)
print("Per-subject agreement (Self-Reported vs. Feature-Derived)")
print("="*60)

for subj in sorted(df_clean['subject'].unique()):
    subj_df = df_clean[df_clean['subject'] == subj]
    y_true_sub = subj_df['self_num'].astype(int)
    y_pred_sub = subj_df['feat_num'].astype(int)
    acc_sub = accuracy_score(y_true_sub, y_pred_sub)
    kappa_sub = cohen_kappa_score(y_true_sub, y_pred_sub)
    print(f"{subj:20s}  Accuracy: {acc_sub:.3f} ({acc_sub*100:.1f}%)  kappa: {kappa_sub:.3f}")

print("\nSelf-reported distribution:")
print(df_clean['self_reported_difficulty'].value_counts().reindex(['Easy','Medium','Hard']))
print("\nFeature-derived distribution:")
print(df_clean['feature_derived_difficulty'].value_counts().reindex(['Easy','Medium','Hard']))