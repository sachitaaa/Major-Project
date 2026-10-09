# """
# Print all remaining table values for the paper from master_dataset_cleaned.csv.
# """

# import pandas as pd
# import numpy as np
# from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
# from scipy.stats import spearmanr, chi2_contingency

# df = pd.read_csv('/home/sujal/Programming/MajorProject/evaluation/master_dataset.csv')
# LABELS = ['Easy', 'Medium', 'Hard']
# LMAP = {'Easy': 1, 'Medium': 2, 'Hard': 3}
# BLOOM = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
# POS = ['A', 'B', 'C', 'D']
# N = len(df)

# print("=" * 72)
# print(f"TOTAL N = {N}")
# print("=" * 72)
# print(df['subject'].value_counts().sort_index())
# print()


# # =========================================================
# print("\n" + "=" * 72)
# print("TABLE V: Feature Profile by Subject (Mean ± SD) + Overall")
# print("=" * 72)
# cols_v = [
#     ('Stem words', 'stem_word_count', 2),
#     ('Option words', 'avg_option_word_count', 2),
#     ('Pairwise Jaccard', 'pairwise_avg_jaccard', 3),
#     ('Distractor similarity', 'avg_distractor_jaccard_to_correct', 3),
#     ('Explanation words', 'explanation_word_count', 2),
#     ('Expl->Question Jaccard', 'expl_to_question_jaccard', 3),
#     ('Expl->Correct Jaccard', 'expl_to_correct_jaccard', 3),
#     ('New information ratio', 'expl_new_info_ratio', 3),
#     ('Flesch-Kincaid grade', 'flesch_kincaid_grade', 2),
# ]
# for name, col, prec in cols_v:
#     line = f"{name:28s}"
#     for s in ['ai', 'computernetworks', 'datamining', 'softengineering']:
#         sub = df[df['subject'] == s][col]
#         line += f"  {sub.mean():.{prec}f}±{sub.std():.{prec}f}"
#     line += f"  | Overall: {df[col].mean():.{prec}f}±{df[col].std():.{prec}f}"
#     print(line)

# # Verbatim copy as percentage
# line = f"{'Verbatim copy (%)':28s}"
# for s in ['ai', 'computernetworks', 'datamining', 'softengineering']:
#     sub = df[df['subject'] == s]['expl_contains_correct_verbatim']
#     line += f"  {sub.mean()*100:.2f}±{sub.std()*100:.2f}"
# line += f"  | Overall: {df['expl_contains_correct_verbatim'].mean()*100:.2f}±{df['expl_contains_correct_verbatim'].std()*100:.2f}"
# print(line)


# # =========================================================
# print("\n" + "=" * 72)
# print("TABLE VII: Bloom's Taxonomy Distribution by Subject (%)")
# print("=" * 72)
# bloom_ct = pd.crosstab(df['subject'], df['bloom_taxonomy_level'], normalize='index') * 100
# bloom_ct = bloom_ct.reindex(columns=BLOOM, fill_value=0)
# print(bloom_ct.round(1))


# # =========================================================
# print("\n" + "=" * 72)
# print("TABLE VIII: Correct Answer Position by Subject (%)")
# print("=" * 72)
# pos_ct = pd.crosstab(df['subject'], df['correct_answer'], normalize='index') * 100
# pos_ct = pos_ct.reindex(columns=POS, fill_value=0)
# print(pos_ct.round(1))
# overall_pos = df['correct_answer'].value_counts(normalize=True).reindex(POS) * 100
# print("\nOverall:")
# print(overall_pos.round(1))

# obs = df['correct_answer'].value_counts().reindex(POS, fill_value=0).values
# expected = np.full(4, N / 4)
# chi2, p, _, _ = chi2_contingency([obs, expected])
# print(f"\nChi-square = {chi2:.2f}, p = {p:.2e}")


# # =========================================================
# print("\n" + "=" * 72)
# print("TABLE IX + stats: Self-Reported vs Feature-Derived")
# print("=" * 72)
# cm_fd = confusion_matrix(df['self_reported_difficulty'],
#                          df['feature_derived_difficulty'], labels=LABELS)
# print(pd.DataFrame(cm_fd, index=LABELS, columns=LABELS))
# y_true = df['self_reported_difficulty'].map(LMAP)
# y_fd = df['feature_derived_difficulty'].map(LMAP)
# print(f"\nAccuracy: {accuracy_score(y_true, y_fd)*100:.1f}% "
#       f"({int(accuracy_score(y_true, y_fd)*N)}/{N})")
# print(f"Cohen's kappa:   {cohen_kappa_score(y_true, y_fd):.3f}")
# print(f"Weighted kappa:  {cohen_kappa_score(y_true, y_fd, weights='linear'):.3f}")
# r, pv = spearmanr(y_true, y_fd)
# print(f"Spearman rho:    {r:.3f}  (p = {pv:.2e})")
# print("\nFeature-derived distribution:")
# print(df['feature_derived_difficulty'].value_counts().reindex(LABELS))
# print("\nSelf-reported distribution:")
# print(df['self_reported_difficulty'].value_counts().reindex(LABELS))


# # =========================================================
# print("\n" + "=" * 72)
# print("TABLE X: Feature Correlations with Feature-Derived Score")
# print("=" * 72)
# features_x = [
#     ('Stem word count', 'stem_word_count'),
#     ('Flesch-Kincaid grade', 'flesch_kincaid_grade'),
#     ('Avg. distractor Jaccard to correct', 'avg_distractor_jaccard_to_correct'),
#     ('New information ratio (explanation)', 'expl_new_info_ratio'),
#     ('Avg. option word count', 'avg_option_word_count'),
#     ('Explanation word count', 'explanation_word_count'),
# ]
# for name, col in features_x:
#     r = df[col].corr(df['feature_score_scaled'])
#     print(f"{name:45s} {r:+.3f}")


# # =========================================================
# print("\n" + "=" * 72)
# print("TABLE XI: Summary Stats by Feature-Derived Category")
# print("=" * 72)
# for cat in LABELS:
#     sub = df[df['feature_derived_difficulty'] == cat]
#     print(f"\n{cat} (n={len(sub)}):")
#     print(f"  Stem words:     {sub['stem_word_count'].mean():.2f} ± {sub['stem_word_count'].std():.2f}")
#     print(f"  Distractor sim: {sub['avg_distractor_jaccard_to_correct'].mean():.3f} "
#           f"± {sub['avg_distractor_jaccard_to_correct'].std():.3f}")
#     print(f"  New info ratio: {sub['expl_new_info_ratio'].mean():.3f} "
#           f"± {sub['expl_new_info_ratio'].std():.3f}")


# # =========================================================
# print("\n" + "=" * 72)
# print("TABLE XII: Bloom's Level by Feature-Derived Category (% within)")
# print("=" * 72)
# bloom_by_diff = pd.crosstab(df['feature_derived_difficulty'],
#                              df['bloom_taxonomy_level'],
#                              normalize='index') * 100
# bloom_by_diff = bloom_by_diff.reindex(index=LABELS, columns=BLOOM, fill_value=0)
# print(bloom_by_diff.round(1))


# # =========================================================
# print("\n" + "=" * 72)
# print("TABLE XIII + stats: Self-Reported vs LLM-Augmented")
# print("=" * 72)
# cm_llm = confusion_matrix(df['self_reported_difficulty'],
#                           df['llm_response_augmented_difficulty'], labels=LABELS)
# print(pd.DataFrame(cm_llm, index=LABELS, columns=LABELS))
# y_llm = df['llm_response_augmented_difficulty'].map(LMAP)
# print(f"\nAccuracy: {accuracy_score(y_true, y_llm)*100:.1f}% "
#       f"({int(accuracy_score(y_true, y_llm)*N)}/{N})")
# print(f"Cohen's kappa:   {cohen_kappa_score(y_true, y_llm):.3f}")
# print(f"Weighted kappa:  {cohen_kappa_score(y_true, y_llm, weights='linear'):.3f}")
# r2, pv2 = spearmanr(y_true, y_llm)
# print(f"Spearman rho:    {r2:.3f}  (p = {pv2:.2e})")


# # =========================================================
# print("\n" + "=" * 72)
# print("TABLE XIV: Per-Subject Agreement")
# print("=" * 72)
# rows = []
# for subj in ['ai', 'computernetworks', 'datamining', 'softengineering']:
#     sub = df[df['subject'] == subj]
#     yt = sub['self_reported_difficulty'].map(LMAP)
#     yf = sub['feature_derived_difficulty'].map(LMAP)
#     yl = sub['llm_response_augmented_difficulty'].map(LMAP)
#     rows.append({
#         'Subject': subj,
#         'N': len(sub),
#         'FD_Acc': round(accuracy_score(yt, yf), 3),
#         'FD_kappa': round(cohen_kappa_score(yt, yf), 3),
#         'LLM_Acc': round(accuracy_score(yt, yl), 3),
#         'LLM_kappa': round(cohen_kappa_score(yt, yl), 3),
#     })
# print(pd.DataFrame(rows).to_string(index=False))

# print("\n" + "=" * 72)
# print("Done. Copy these values into the paper tables.")
# print("=" * 72)

import pandas as pd

df = pd.read_csv('/home/sujal/Programming/MajorProject/evaluation/master_dataset.csv')

# Normalize case
df['bloom'] = df['bloom_taxonomy_level'].str.capitalize()

BLOOM = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
LABELS = ['Easy', 'Medium', 'Hard']

print("=" * 72)
print("TABLE VII: Bloom's Taxonomy Distribution by Subject (%)")
print("=" * 72)
bloom_ct = pd.crosstab(df['subject'], df['bloom'], normalize='index') * 100
bloom_ct = bloom_ct.reindex(columns=BLOOM, fill_value=0)
print(bloom_ct.round(1))

print("\n" + "=" * 72)
print("TABLE XII: Bloom's Level by Feature-Derived Difficulty (% within)")
print("=" * 72)
b = pd.crosstab(df['feature_derived_difficulty'], df['bloom'],
                normalize='index') * 100
b = b.reindex(index=LABELS, columns=BLOOM, fill_value=0)
print(b.round(1))