#!/usr/bin/env python3
"""
add_explanation.py

Merges explanations and citation page references from the raw generation
JSON files into the cleaned per-subject JSON files.

Usage:
    python add_explanation.py -i <input_dir> -o <output_dir>
"""

import json
import re
import os
import argparse
import sys


# ------------------------------------------------------------------
# Citation extraction
#   Handles: Page 19 | page 183 | Page: 2 | Page 2:19 | page 13:32
#            pages 185-186 | pages 185‑186 | (pages 61, 5)
#            [Page 95] | [Page 36, 37] | [Page: 2]
# ------------------------------------------------------------------
CITATION = re.compile(
    r'\[?\s*[Pp]ages?[:]?\s+([0-9]+(?:\s*[:,\-‑–]\s*[0-9]+)*)'
)


def extract_citations(text):
    """Return sorted unique list of page numbers found after 'page(s)'."""
    if not text:
        return []
    pages = set()
    for m in CITATION.finditer(text):
        for n in re.split(r'[:,\-‑–]', m.group(1)):
            n = n.strip()
            if n.isdigit():
                pages.add(int(n))
    return sorted(pages)


def clean_explanation(text):
    """Strip trailing '---' separators and collapse whitespace."""
    if not text:
        return text
    t = re.sub(r'\s*-{3,}\s*$', '', text)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


# ------------------------------------------------------------------
# Question text normalizer (used for matching cleaned <-> raw)
# ------------------------------------------------------------------
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


def load_raw(path):
    """Return list of question dicts from a raw generation file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'questions' in data:
        return data['questions']
    if isinstance(data, list):
        return data
    raise ValueError(f"Unknown raw format: {path}")


PAIRS = {
    'ai_cleaned.json':               'ai.json',
    'computernetworks_cleaned.json': 'computernetworks.json',
    'datamining_cleaned.json':       'datamining.json',
    'softengineering_cleaned.json':  'softengineering.json',
}


# ------------------------------------------------------------------
# Self-tests
# ------------------------------------------------------------------
def run_tests():
    tests = [
        ("as shown on Page 19. DFS is not...",    [19]),
        ("described on page 183",                  [183]),
        ("see the complexity notes on Page 2:19",  [2, 19]),
        ("The CNF steps listed on page 13:32",     [13, 32]),
        ("pages 185-186",                          [185, 186]),
        ("pages 185‑186",                          [185, 186]),
        ("(pages 61, 5)",                          [5, 61]),
        ("[Page 95]",                              [95]),
        ("[Page 36, 37]",                          [36, 37]),
        ("[Page: 2]",                              [2]),
    ]
    ok = True
    for text, expect in tests:
        got = extract_citations(text)
        status = "OK  " if got == sorted(expect) else "FAIL"
        if got != sorted(expect):
            ok = False
        print(f"  {status} {text!r} -> {got}")
    return ok


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',  required=True,
                    help="Directory containing *_cleaned.json and raw *.json")
    ap.add_argument('-o', '--output', required=True,
                    help="Directory for the updated *_cleaned.json files")
    ap.add_argument('--skip-tests', action='store_true')
    args = ap.parse_args()

    if not args.skip_tests:
        print("=== Regex sanity tests ===")
        if not run_tests():
            print(">>> TESTS FAILED - aborting <<<")
            sys.exit(1)
        print("All tests passed.\n")

    if not os.path.isdir(args.input):
        print(f"Input dir not found: {args.input}")
        sys.exit(1)
    os.makedirs(args.output, exist_ok=True)

    print(f"Input : {args.input}")
    print(f"Output: {args.output}\n")

    for cleaned_name, raw_name in PAIRS.items():
        cpath = os.path.join(args.input, cleaned_name)
        rpath = os.path.join(args.input, raw_name)

        if not os.path.isfile(cpath):
            print(f"[SKIP] {cleaned_name} not found in {args.input}")
            continue
        if not os.path.isfile(rpath):
            print(f"[SKIP] {raw_name} not found in {args.input}")
            continue

        with open(cpath, 'r', encoding='utf-8') as f:
            cleaned = json.load(f)
        raw_qs = load_raw(rpath)

        # Build normalized lookup: question text -> raw entry (first wins)
        lookup = {}
        for q in raw_qs:
            k = normalize(q.get('question', ''))
            if k not in lookup:
                lookup[k] = q

        matched = unmatched = 0
        cited = 0
        for item in cleaned['questions']:
            k = normalize(item['question'])
            r = lookup.get(k)
            if r is None:
                item['explanation'] = None
                item['citation_pages'] = []
                item['citation_present'] = False
                unmatched += 1
                continue
            expl = clean_explanation(r.get('explanation', ''))
            item['explanation'] = expl
            item['citation_pages'] = extract_citations(expl)
            item['citation_present'] = len(item['citation_pages']) > 0
            if item['citation_present']:
                cited += 1
            matched += 1

        out_path = os.path.join(args.output, cleaned_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)

        n = len(cleaned['questions'])
        pct = 100 * cited / n if n else 0
        print(f"{cleaned_name:34s}  n={n:3d}  "
              f"matched={matched:3d}  unmatched={unmatched:3d}  "
              f"cited={cited:3d}/{n:3d} ({pct:5.1f}%)")

    print("\nDone.")


if __name__ == '__main__':
    main()