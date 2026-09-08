import pandas as pd
import numpy as np

CSV_PATH = '/home/sujal/Programming/MajorProject/evaluation/all_features_combined.csv'
OUTPUT_DIR = '/home/sujal/Programming/MajorProject/evaluation'

df = pd.read_csv(CSV_PATH)

metrics = [
    'stem_word_count',
    'avg_option_word_count',
    'pairwise_avg_jaccard',
    'avg_distractor_jaccard_to_correct',
    'explanation_word_count',
    'expl_to_question_jaccard',
    'expl_to_correct_jaccard',
    'expl_new_info_ratio',
    'expl_contains_correct_verbatim',
    'flesch_kincaid_grade'
]

pretty_names = {
    'stem_word_count': 'Stem words',
    'avg_option_word_count': 'Option words',
    'pairwise_avg_jaccard': 'Pairwise Jaccard',
    'avg_distractor_jaccard_to_correct': 'Distractor similarity',
    'explanation_word_count': 'Explanation words',
    'expl_to_question_jaccard': 'Expl→Question Jaccard',
    'expl_to_correct_jaccard': 'Expl→Correct Jaccard',
    'expl_new_info_ratio': 'New information ratio',
    'expl_contains_correct_verbatim': 'Verbatim copy (%)',
    'flesch_kincaid_grade': 'Flesch–Kincaid grade'
}

df['expl_contains_correct_verbatim'] = df['expl_contains_correct_verbatim'] * 100

subjects = sorted(df['subject'].unique())
subject_stats = {}
for sub in subjects:
    sub_df = df[df['subject'] == sub]
    subject_stats[sub] = {m: (sub_df[m].mean(), sub_df[m].std()) for m in metrics}

overall_stats = {m: (df[m].mean(), df[m].std()) for m in metrics}

rows = []
for m in metrics:
    row = {'Metric': pretty_names[m]}
    for sub in subjects:
        mean, std = subject_stats[sub][m]
        row[f'{sub}_mean'] = round(mean, 3)
        row[f'{sub}_std'] = round(std, 3)
    mean, std = overall_stats[m]
    row['Overall_mean'] = round(mean, 3)
    row['Overall_std'] = round(std, 3)
    rows.append(row)

summary_df = pd.DataFrame(rows)
cols = ['Metric'] + [f'{s}_mean' for s in subjects] + ['Overall_mean'] + [f'{s}_std' for s in subjects] + ['Overall_std']
summary_df = summary_df[cols]

summary_csv = f"{OUTPUT_DIR}/stats_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Summary stats saved to: {summary_csv}")

corr_metrics = [
    'explanation_word_count',
    'expl_new_info_ratio',
    'expl_to_correct_jaccard',
    'expl_to_question_jaccard',
    'avg_option_word_count',
    'stem_word_count'
]
corr_df = df[corr_metrics].corr()
corr_df.columns = [pretty_names[c] for c in corr_df.columns]
corr_df.index = [pretty_names[c] for c in corr_df.index]

corr_csv = f"{OUTPUT_DIR}/correlation_matrix.csv"
corr_df.round(3).to_csv(corr_csv)
print(f"Correlation matrix saved to: {corr_csv}")

print("\n" + "=" * 80)
print("MEAN ± STD PER SUBJECT")
print("=" * 80)
print(f"{'Metric':<30}", end="")
for sub in subjects:
    print(f"{sub:>12}", end="")
print(f"{'Overall':>10}")
print("-" * 80)

for m in metrics:
    print(f"{pretty_names[m]:<30}", end="")
    for sub in subjects:
        mean, std = subject_stats[sub][m]
        print(f"{mean:>6.2f} ± {std:>4.2f}", end="")
    mean, std = overall_stats[m]
    print(f"{mean:>6.2f} ± {std:>4.2f}")
print("=" * 80)

print("\n" + "=" * 80)
print("PEARSON CORRELATION MATRIX")
print("=" * 80)
print(corr_df.round(3))
print("=" * 80)

print("\nKey Correlations:")
pairs = [
    ('explanation_word_count', 'expl_new_info_ratio'),
    ('explanation_word_count', 'expl_to_correct_jaccard'),
    ('expl_new_info_ratio', 'expl_to_correct_jaccard'),
    ('stem_word_count', 'explanation_word_count'),
]
for a, b in pairs:
    r = corr_df.loc[pretty_names[a], pretty_names[b]]
    print(f"  {pretty_names[a]} vs {pretty_names[b]}: r = {r:.3f}")