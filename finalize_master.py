#!/usr/bin/env python3
"""
finalize_master.py

Updates master_dataset.csv so its explanation-grounding columns match the
values reported in the paper's Table 5 and Table 6.

Steps:
  - Reads the master CSV (which still has the OLD Jaccard values)
  - Reads the cleaned JSONs from the MiniLM run (which have the NEW
    Jaccard + cosine per question)
  - Overwrites expl_to_question_jaccard and expl_to_correct_jaccard
  - Adds expl_to_question_cosine and expl_to_correct_cosine
  - Writes master_dataset_final.csv

Also prints per-subject means so you can verify they match Table 5/Table 6.
"""

import json
import os
import re
import argparse
from collections import defaultdict
import pandas as pd


def normalize(s):
    s = str(s).lower().strip()
    s = (s.replace('\u2010', '-').replace('\u2011', '-')
           .replace('\u2012', '-').replace('\u2013', '-')
           .replace('\u2014', '-').replace('\u2212', '-'))
    s = (s.replace('\u2018', "'").replace('\u2019', "'")
           .replace('\u201c', '"').replace('\u201d', '"'))
    s = s.replace('*', '').replace('\u00a0', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--master', required=True,
                    help="Path to master_dataset.csv")
    ap.add_argument('-j', '--json-dir', required=True,
                    help="Directory with cleaned JSONs that have cosine fields")
    ap.add_argument('-o', '--output', required=True,
                    help="Path for the updated master CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.master, encoding='utf-8-sig')
    print(f"Master loaded: {len(df)} rows")

    files = [
        'ai_cleaned.json',
        'computernetworks_cleaned.json',
        'datamining_cleaned.json',
        'softengineering_cleaned.json',
    ]

    # Build lookup
    lookup = {}
    for fname in files:
        path = os.path.join(args.json_dir, fname)
        if not os.path.isfile(path):
            print(f"[SKIP] {fname} not found")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for q in data['questions']:
            k = normalize(q['question'])
            if k not in lookup:
                lookup[k] = {
                    'jq': q.get('expl_to_question_jaccard'),
                    'ja': q.get('expl_to_correct_jaccard'),
                    'cq': q.get('expl_to_question_cosine'),
                    'ca': q.get('expl_to_correct_cosine'),
                }
        print(f"  loaded {len(data['questions']):3d} from {fname}")
    print(f"Total unique questions in JSONs: {len(lookup)}")

    jq_col, ja_col, cq_col, ca_col = [], [], [], []
    missing = 0
    for _, row in df.iterrows():
        k = normalize(row['question'])
        rec = lookup.get(k)
        if rec is None:
            jq_col.append(None); ja_col.append(None)
            cq_col.append(None); ca_col.append(None)
            missing += 1
            continue
        jq_col.append(rec['jq'])
        ja_col.append(rec['ja'])
        cq_col.append(rec['cq'])
        ca_col.append(rec['ca'])

    # Overwrite jaccard, add cosine
    df['expl_to_question_jaccard']  = jq_col
    df['expl_to_correct_jaccard']   = ja_col
    df['expl_to_question_cosine']   = cq_col
    df['expl_to_correct_cosine']    = ca_col

    df.to_csv(args.output, index=False, encoding='utf-8-sig')

    matched = len(df) - missing
    print(f"\nMatched: {matched}/{len(df)}  (missing: {missing})")
    print(f"Written to: {args.output}")

    # ---------- Sanity check ----------
    print("\n=== Per-subject means (should match paper Table 5/6) ===")
    print(f"{'Subject':22s} {'Jac->Q':>8s} {'Jac->A':>8s} "
          f"{'Cos->Q':>8s} {'Cos->A':>8s}")
    print("-" * 60)
    grp = df.groupby('subject')[
        ['expl_to_question_jaccard', 'expl_to_correct_jaccard',
         'expl_to_question_cosine',  'expl_to_correct_cosine']
    ].mean().round(3)
    for subj, row in grp.iterrows():
        print(f"{subj:22s} {row['expl_to_question_jaccard']:8.3f} "
              f"{row['expl_to_correct_jaccard']:8.3f} "
              f"{row['expl_to_question_cosine']:8.3f} "
              f"{row['expl_to_correct_cosine']:8.3f}")
    print("-" * 60)
    print(f"{'Overall':22s} "
          f"{df['expl_to_question_jaccard'].mean():8.3f} "
          f"{df['expl_to_correct_jaccard'].mean():8.3f} "
          f"{df['expl_to_question_cosine'].mean():8.3f} "
          f"{df['expl_to_correct_cosine'].mean():8.3f}")


if __name__ == '__main__':
    main()