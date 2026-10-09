import json
import re
import os
import argparse
import pandas as pd


def normalize(s):
    """Normalize question text for matching."""
    s = str(s).lower().strip()
    s = (s.replace('\u2010', '-').replace('\u2011', '-')
           .replace('\u2012', '-').replace('\u2013', '-')
           .replace('\u2014', '-').replace('\u2212', '-'))
    s = (s.replace('\u2018', "'").replace('\u2019', "'")
           .replace('\u201c', '"').replace('\u201d', '"'))
    s = s.replace('*', '')
    s = s.replace('\u00a0', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s


def main():
    parser = argparse.ArgumentParser(
        description="Clean subject-wise JSON files and merge difficulty labels from master_dataset.csv"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Directory containing master_dataset.csv and the *_merged.json files"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Directory where cleaned JSON files will be written"
    )
    parser.add_argument(
        "--master", default="master_dataset.csv",
        help="Filename of the master CSV (default: master_dataset.csv)"
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    master_path = os.path.join(input_dir, args.master)

    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(master_path):
        raise SystemExit(f"Master CSV not found: {master_path}")

    # ---------- Load master ----------
    master = pd.read_csv(master_path, encoding='utf-8-sig')
    master_lookup = {}
    for _, row in master.iterrows():
        master_lookup[normalize(row['question'])] = row
    print(f"Master loaded: {len(master_lookup)} unique questions from {master_path}\n")

    # ---------- Subject config ----------
    subjects = {
        'ai':               ('ai_merged.json',              'AI', 56),
        'computernetworks': ('computerNetowrk_merged.json', 'CN', 50),
        'datamining':       ('datamining_merged.json',      'DM', 55),
        'softengineering':  ('softengineering_merged.json', 'SE', 50),
    }

    # ---------- Process each subject ----------
    summary = []
    for subject, (json_name, prefix, expected) in subjects.items():
        json_path = os.path.join(input_dir, json_name)
        if not os.path.isfile(json_path):
            print(f"[SKIP] {subject}: file not found -> {json_path}")
            summary.append((subject, 0, expected, "MISSING FILE"))
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        seen = set()
        cleaned = []
        counter = 1
        missing_from_master = 0
        duplicates = 0

        for item in data.get('detailed_results', []):
            q_text = item.get('question', '')
            key = normalize(q_text)

            if key in seen:
                duplicates += 1
                print(f"  [{subject}] duplicate removed: {q_text[:70]}...")
                continue
            seen.add(key)

            mrow = master_lookup.get(key)
            feature_diff = None
            llm_diff = None
            if mrow is not None:
                fd = mrow.get('feature_derived_difficulty')
                ld = mrow.get('llm_response_augmented_difficulty')
                feature_diff = None if pd.isna(fd) else fd
                llm_diff = None if pd.isna(ld) else ld
            else:
                missing_from_master += 1
                print(f"  [{subject}] NOT in master: {q_text[:70]}...")

            cleaned.append({
                "question_id": f"{prefix}_{counter:03d}",
                "question": q_text,
                "options": item.get('options', {}),
                "correct_answer": item.get('correct_answer'),
                "bloom_level": item.get('bloom_level'),
                "self_reported_difficulty": item.get('self_reported_difficulty'),
                "feature_derived_difficulty": feature_diff,
                "llm_response_augmented_difficulty": llm_diff,
            })
            counter += 1

        output = {
            "subject": subject,
            "total_questions": len(cleaned),
            "expected_total": expected,
            "questions": cleaned,
        }

        out_path = os.path.join(output_dir, f"{subject}_cleaned.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        status = "OK" if len(cleaned) == expected else "MISMATCH"
        extra = []
        if duplicates:
            extra.append(f"dupes removed: {duplicates}")
        if missing_from_master:
            extra.append(f"missing in master: {missing_from_master}")
        extra_str = (" | " + ", ".join(extra)) if extra else ""

        print(f"{subject:18s} -> {len(cleaned):3d} questions "
              f"(expected {expected}) [{status}]{extra_str}")

        summary.append((subject, len(cleaned), expected, status))

    # ---------- Summary ----------
    print("\n=== Summary ===")
    total_out = 0
    total_expected = 0
    for s, got, exp, status in summary:
        print(f"  {s:18s}  {got:3d} / {exp:3d}  [{status}]")
        total_out += got
        total_expected += exp
    print(f"  {'TOTAL':18s}  {total_out:3d} / {total_expected:3d}")

    print(f"\nCleaned files written to: {output_dir}")


if __name__ == "__main__":
    main()