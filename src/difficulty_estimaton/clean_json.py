import json
import re
import pandas as pd

def normalize(s):
    """Normalize question text for matching."""
    s = str(s).lower().strip()
    # Normalize unicode dashes and quotes
    s = (s.replace('\u2010', '-').replace('\u2011', '-')
           .replace('\u2012', '-').replace('\u2013', '-')
           .replace('\u2014', '-').replace('\u2212', '-'))
    s = (s.replace('\u2018', "'").replace('\u2019', "'")
           .replace('\u201c', '"').replace('\u201d', '"'))
    s = s.replace('*', '')                  # strip markdown bold
    s = re.sub(r'\s+', ' ', s)              # collapse whitespace
    s = s.replace('\u00a0', ' ')            # non-breaking space
    return s

# ---------- Load master ----------
master = pd.read_csv('master_dataset.csv', encoding='utf-8-sig')
master_lookup = {}
for _, row in master.iterrows():
    master_lookup[normalize(row['question'])] = row
print(f"Master loaded: {len(master_lookup)} unique questions")

# ---------- Subject config ----------
subjects = {
    'ai':              ('ai_merged.json',                 'AI'),
    'computernetworks':('computerNetowrk_merged.json',    'CN'),
    'datamining':      ('datamining_merged.json',         'DM'),
    'softengineering': ('softengineering_merged.json',    'SE'),
}

expected_counts = {
    'ai': 56,
    'computernetworks': 50,
    'datamining': 55,
    'softengineering': 50,
}

# ---------- Process each subject ----------
for subject, (json_path, prefix) in subjects.items():
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    seen = set()
    cleaned = []
    counter = 1
    missing_from_master = 0

    for item in data['detailed_results']:
        q_text = item['question']
        key = normalize(q_text)

        # Deduplicate
        if key in seen:
            print(f"  [{subject}] duplicate removed: {q_text[:70]}...")
            continue
        seen.add(key)

        # Pull difficulty labels from master
        mrow = master_lookup.get(key)
        feature_diff = None
        llm_diff = None
        if mrow is not None:
            feature_diff = mrow.get('feature_derived_difficulty')
            llm_diff = mrow.get('llm_response_augmented_difficulty')
            if pd.isna(feature_diff): feature_diff = None
            if pd.isna(llm_diff): llm_diff = None
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
        "expected_total": expected_counts[subject],
        "questions": cleaned,
    }

    out_path = f"{subject}_cleaned.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    status = "OK" if len(cleaned) == expected_counts[subject] else "MISMATCH"
    print(f"{subject:18s} -> {len(cleaned):3d} questions "
          f"(expected {expected_counts[subject]}) [{status}]"
          f"{' | missing from master: '+str(missing_from_master) if missing_from_master else ''}")

print("\nDone. Files written:")
for s in subjects:
    print(f"  {s}_cleaned.json")