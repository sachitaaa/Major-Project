#!/usr/bin/env python3
"""
cosine_similarity.py

Computes cosine similarity between each question's explanation and:
  - the question text
  - the correct answer option text

Also computes Jaccard (word overlap) for the same pairs, then compares
all four metrics against the paper's reported Table 6 values.

Default model: sentence-transformers/all-MiniLM-L6-v2 (no prefix).

Usage:
    python cosine_similarity.py -i <cleaned_dir> -o <output_dir>
"""

import json
import os
import re
import argparse
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------------
# Jaccard (word overlap)
# ------------------------------------------------------------------
def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a, b):
    A, B = tokenize(a), tokenize(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def correct_answer_text(item):
    """Return the full text of the correct option."""
    letter = item.get('correct_answer', '')
    return item.get('options', {}).get(letter, '')


# ------------------------------------------------------------------
# Paper's reported Table 6
# ------------------------------------------------------------------
PAPER_TABLE6 = {
    'ai':               {'jac_a': 0.200, 'cos_a': 0.628,
                         'jac_q': 0.172, 'cos_q': 0.682},
    'computernetworks': {'jac_a': 0.221, 'cos_a': 0.700,
                         'jac_q': 0.159, 'cos_q': 0.658},
    'datamining':       {'jac_a': 0.198, 'cos_a': 0.656,
                         'jac_q': 0.143, 'cos_q': 0.670},
    'softengineering':  {'jac_a': 0.211, 'cos_a': 0.648,
                         'jac_q': 0.153, 'cos_q': 0.646},
}

TOL_JAC = 0.02
TOL_COS = 0.05


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
                    help="Directory containing *_cleaned.json")
    ap.add_argument('-o', '--output', required=True,
                    help="Directory for JSONs with cosine fields added")
    ap.add_argument('--model',
                    default='sentence-transformers/all-MiniLM-L6-v2',
                    help="SentenceTransformer model name")
    ap.add_argument('--prefix', default='',
                    help="Task prefix for models that need one.")
    ap.add_argument('--no-normalize', action='store_true',
                    help="Skip L2 normalization of embeddings")
    ap.add_argument('--answer-mode', choices=['full', 'letter'],
                    default='full',
                    help="Use full option text ('full') or just the "
                         "letter ('letter') as the 'correct answer'.")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = [
        'ai_cleaned.json',
        'computernetworks_cleaned.json',
        'datamining_cleaned.json',
        'softengineering_cleaned.json',
    ]

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, trust_remote_code=True)
    print(f"Prefix      : {args.prefix!r}")
    print(f"Normalize   : {not args.no_normalize}")
    print(f"Answer mode : {args.answer_mode}\n")

    def embed(texts):
        if args.prefix:
            texts = [args.prefix + t for t in texts]
        return model.encode(texts,
                            normalize_embeddings=(not args.no_normalize),
                            show_progress_bar=False)

    def get_answer(item):
        if args.answer_mode == 'letter':
            return item.get('correct_answer', '')
        return correct_answer_text(item)

    per_subject = defaultdict(lambda: {'cos_q': [], 'cos_a': [],
                                       'jac_q': [], 'jac_a': []})

    for fname in files:
        path = os.path.join(args.input, fname)
        if not os.path.isfile(path):
            print(f"[SKIP] {fname} not found in {args.input}")
            continue

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        qs = data['questions']
        usable = [q for q in qs if q.get('explanation')]

        if not usable:
            print(f"[SKIP] {fname}: no questions with explanations")
            continue

        explanations = [q['explanation'] for q in usable]
        questions    = [q['question'] for q in usable]
        answers      = [get_answer(q) for q in usable]

        E_expl = embed(explanations)
        E_q    = embed(questions)

        cos_q = (E_expl * E_q).sum(axis=1)

        if args.answer_mode == 'letter' or not any(answers):
            cos_a = np.zeros(len(usable))
        else:
            E_ans = embed(answers)
            cos_a = (E_expl * E_ans).sum(axis=1)

        jac_q = [jaccard(e, q) for e, q in zip(explanations, questions)]
        jac_a = [jaccard(e, a) for e, a in zip(explanations, answers)]

        for i, item in enumerate(usable):
            item['expl_to_question_cosine']  = float(cos_q[i])
            item['expl_to_correct_cosine']   = float(cos_a[i])
            item['expl_to_question_jaccard'] = float(jac_q[i])
            item['expl_to_correct_jaccard']  = float(jac_a[i])

        out_path = os.path.join(args.output, fname)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        subj = data.get('subject', fname.replace('_cleaned.json', ''))
        per_subject[subj]['cos_q'].extend(cos_q.tolist())
        per_subject[subj]['cos_a'].extend(cos_a.tolist())
        per_subject[subj]['jac_q'].extend(jac_q)
        per_subject[subj]['jac_a'].extend(jac_a)

        print(f"{fname:34s}  n={len(usable):3d}  "
              f"cos->correct={np.mean(cos_a):.3f}  "
              f"cos->question={np.mean(cos_q):.3f}  "
              f"jac->correct={np.mean(jac_a):.3f}  "
              f"jac->question={np.mean(jac_q):.3f}")

    # ------------------------------------------------------------------
    # Per-subject summary (Table 6 format)
    # ------------------------------------------------------------------
    print("\n=== Per-subject summary (Table 6 format) ===")
    header = (f"{'Subject':22s} {'Jac->Correct':>12s} {'Cos->Correct':>12s} "
              f"{'Jac->Question':>14s} {'Cos->Question':>14s}")
    print(header)
    print("-" * len(header))

    all_cos_q, all_cos_a, all_jac_q, all_jac_a = [], [], [], []
    for subj in sorted(per_subject):
        d = per_subject[subj]
        print(f"{subj:22s} "
              f"{np.mean(d['jac_a']):12.3f} {np.mean(d['cos_a']):12.3f} "
              f"{np.mean(d['jac_q']):14.3f} {np.mean(d['cos_q']):14.3f}")
        all_cos_a.extend(d['cos_a']); all_cos_q.extend(d['cos_q'])
        all_jac_a.extend(d['jac_a']); all_jac_q.extend(d['jac_q'])

    print("-" * len(header))
    print(f"{'Overall':22s} "
          f"{np.mean(all_jac_a):12.3f} {np.mean(all_cos_a):12.3f} "
          f"{np.mean(all_jac_q):14.3f} {np.mean(all_cos_q):14.3f}")

    # ------------------------------------------------------------------
    # Consistency check vs paper Table 6
    # ------------------------------------------------------------------
    print("\n=== Consistency check vs. paper Table 6 ===")
    header2 = (f"{'Subject':22s} {'Metric':14s} "
               f"{'Paper':>8s} {'Got':>8s} {'Delta':>8s} {'Status':>10s}")
    print(header2)
    print("-" * len(header2))

    all_match = True
    for subj in sorted(per_subject):
        d = per_subject[subj]
        ref = PAPER_TABLE6.get(subj)
        if ref is None:
            continue

        got = {
            'jac_a': float(np.mean(d['jac_a'])),
            'cos_a': float(np.mean(d['cos_a'])),
            'jac_q': float(np.mean(d['jac_q'])),
            'cos_q': float(np.mean(d['cos_q'])),
        }

        for key, label, tol in [
            ('jac_a', 'Jac->Correct',  TOL_JAC),
            ('cos_a', 'Cos->Correct',  TOL_COS),
            ('jac_q', 'Jac->Question', TOL_JAC),
            ('cos_q', 'Cos->Question', TOL_COS),
        ]:
            delta = got[key] - ref[key]
            ok = abs(delta) <= tol
            if not ok:
                all_match = False
            status = 'OK' if ok else 'MISMATCH'
            print(f"{subj:22s} {label:14s} "
                  f"{ref[key]:8.3f} {got[key]:8.3f} {delta:+8.3f} "
                  f"{status:>10s}")

    print("-" * len(header2))
    if all_match:
        print("RESULT: All values are consistent with the paper's Table 6.")
    else:
        print("RESULT: Mismatches found. Interpretation:")
        print()
        print("  If BOTH Jac and Cos mismatch -> correct-answer text is wrong.")
        print("    Try: --answer-mode letter")
        print()
        print("  If ONLY Cos mismatches -> model or prefix differs.")
        print("    Try: --model nomic-ai/nomic-embed-text-v1.5 --prefix 'search_document: '")
        print("    Or:  --model BAAI/bge-small-en-v1.5 --prefix ''")
        print()
        print("  If ONLY one subject mismatches -> possible duplicate or outlier.")

    print(f"\nFiles with cosine fields written to: {args.output}")


if __name__ == '__main__':
    main()