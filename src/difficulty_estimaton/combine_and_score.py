import argparse
import csv
import json
import pandas as pd
import numpy as np


# Signed weights for the feature-derived composite score (Eq. 2 of methodology)
FEATURE_WEIGHTS = {
    'stem_word_count': 0.30,
    'avg_option_word_count': -0.10,
    'avg_distractor_jaccard_to_correct': 0.25,
    'explanation_word_count': -0.10,
    'expl_new_info_ratio': 0.20,
    'flesch_kincaid_grade': 0.25,
}


def min_max_normalize(series):
    """Eq. 1: min-max normalize a pandas Series to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx - mn == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


def rescale(series, new_min=1, new_max=3):
    """Linearly rescale a series to [new_min, new_max]."""
    mn, mx = series.min(), series.max()
    if mx - mn == 0:
        return pd.Series([(new_min + new_max) / 2] * len(series), index=series.index)
    return new_min + (series - mn) / (mx - mn) * (new_max - new_min)


def tertile_bin(series):
    """Eq. 3: bin into Easy/Medium/Hard using the 33rd/67th percentile of
    the series' own distribution."""
    q33 = series.quantile(0.33)
    q67 = series.quantile(0.67)

    def _bin(x):
        if x <= q33:
            return 'Easy'
        elif x <= q67:
            return 'Medium'
        else:
            return 'Hard'

    return series.apply(_bin)


def load_merged_json(path):
    """Load a *_merged.json (estimator output) file and return a dict
    keyed by stripped question text -> {'llm_response_augmented_difficulty': ...}"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data.get('detailed_results', data)  # tolerate either shape
    out = {}
    for r in results:
        q = r.get('question', '').strip()
        out[q] = {
            'llm_response_augmented_difficulty': r.get('empirical_difficulty', 'Unknown'),
            'self_reported_difficulty_from_merged': r.get('self_reported_difficulty', None),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', nargs='+', required=True,
                         help='List of per-subject feature CSVs (from feature_extraction.py)')
    parser.add_argument('--merged', nargs='+', required=True,
                         help='List of per-subject *_merged.json files, same order as --features')
    parser.add_argument('--output', default='master_dataset.csv')
    args = parser.parse_args()

    if len(args.features) != len(args.merged):
        raise ValueError("--features and --merged must have the same number of files, in matching subject order")

    # ---- 1. Load and concatenate all feature CSVs ----
    dfs = [pd.read_csv(fp) for fp in args.features]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Pooled dataset: {len(df)} questions across {len(args.features)} subjects")

    # ---- 2. Attach LLM-response-augmented difficulty from merged JSONs ----
    df['llm_response_augmented_difficulty'] = None
    unmatched_total = 0
    for merged_path in args.merged:
        lookup = load_merged_json(merged_path)
        for idx, row in df.iterrows():
            if pd.notna(df.at[idx, 'llm_response_augmented_difficulty']):
                continue  # already matched from a previous file
            q = str(row['question']).strip()
            if q in lookup:
                df.at[idx, 'llm_response_augmented_difficulty'] = lookup[q]['llm_response_augmented_difficulty']

    unmatched_total = df['llm_response_augmented_difficulty'].isna().sum()
    if unmatched_total:
        print(f"WARNING: {unmatched_total} questions could not be matched to an empirical/merged result. "
              f"Check for text drift (typos, whitespace) between raw and merged files.")

    # ---- 3. Feature-derived score (Eq. 1-2) ----
    missing_feats = [f for f in FEATURE_WEIGHTS if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing required feature columns for Eq. 2: {missing_feats}")

    norm_feats = {}
    for feat in FEATURE_WEIGHTS:
        norm_feats[feat] = min_max_normalize(df[feat].astype(float))

    composite = pd.Series(0.0, index=df.index)
    for feat, w in FEATURE_WEIGHTS.items():
        composite += w * norm_feats[feat]
    df['feature_score_raw'] = composite

    # Rescale to [1,3] (methodology: "linearly rescaled to interval [1,3]")
    df['feature_score_scaled'] = rescale(df['feature_score_raw'], 1, 3)

    # Eq. 3: tertile bin on the score's own distribution
    df['feature_derived_difficulty'] = tertile_bin(df['feature_score_scaled'])

    # ---- 4. Rename self-reported column for clarity ----
    df.rename(columns={'difficulty': 'self_reported_difficulty'}, inplace=True)

    # ---- 5. Save ----
    df.to_csv(args.output, index=False)
    print(f"Master dataset saved to {args.output}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFeature-derived difficulty distribution:\n{df['feature_derived_difficulty'].value_counts()}")
    print(f"\nSelf-reported difficulty distribution:\n{df['self_reported_difficulty'].value_counts()}")
    print(f"\nLLM-response-augmented difficulty distribution:\n{df['llm_response_augmented_difficulty'].value_counts()}")


if __name__ == '__main__':
    main()