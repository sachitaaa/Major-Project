import json
import csv
import os
import argparse
from pathlib import Path
from collections import defaultdict

def extract_question_data(mcq, subject):
    """
    Extract relevant fields from a single MCQ entry.
    Returns a dict with the desired metrics.
    """
    # Basic info
    row = {
        'subject': subject,
        'question': mcq.get('question', ''),
        'correct_answer': mcq.get('correct_answer', ''),
        'bloom_level': mcq.get('bloom_level', ''),
        'empirical_difficulty': mcq.get('empirical_difficulty', ''),
        'completion_time': mcq.get('completion_time', None),
        'tokens': mcq.get('tokens', None),
        'correct': mcq.get('correct', None),
        'predicted': mcq.get('predicted', ''),
        'latency': mcq.get('latency', None),
        'justification': mcq.get('justification', ''),
    }
    # Optional raw reasoning (may be long, skip by default)
    row['raw_reasoning'] = mcq.get('raw_reasoning', '')

    # Extract features dict if present
    features = mcq.get('features', {})
    if features:
        row['stem_word_count'] = features.get('stem_word_count', None)
        row['correct_word_count'] = features.get('correct_word_count', None)
        row['avg_distractor_word_count'] = features.get('avg_distractor_word_count', None)
        row['stem_sentence_count'] = features.get('stem_sentence_count', None)
        row['avg_word_length'] = features.get('avg_word_length', None)
        row['distractor_count'] = features.get('distractor_count', None)
        row['avg_distractor_overlap'] = features.get('avg_distractor_overlap', None)
        row['max_distractor_similarity'] = features.get('max_distractor_similarity', None)
        row['avg_distractor_similarity'] = features.get('avg_distractor_similarity', None)
        row['content_word_density'] = features.get('content_word_density', None)
        row['negation_presence'] = features.get('negation_presence', None)
        row['modal_presence'] = features.get('modal_presence', None)
        row['quantitative_flag'] = features.get('quantitative_flag', None)

        # Readability (if present)
        for r in ['flesch_reading_ease', 'flesch_kincaid_grade', 'ari', 'gunning_fog', 'coleman_liau', 'smog']:
            if r in features:
                row[r] = features[r]
    return row

def process_json_file(file_path, subject=None):
    """
    Load a JSON file and extract all MCQ entries.
    If subject is None, use the file stem as subject.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Determine the list of MCQs
    if 'detailed_results' in data:
        mcq_list = data['detailed_results']
    else:
        # fallback: maybe it's a list directly
        mcq_list = data if isinstance(data, list) else []

    if not mcq_list:
        print(f"Warning: No 'detailed_results' found in {file_path}")
        return []

    # Get subject from filename if not provided
    if subject is None:
        subject = Path(file_path).stem

    rows = []
    for mcq in mcq_list:
        row = extract_question_data(mcq, subject)
        rows.append(row)
    return rows

def main():
    parser = argparse.ArgumentParser(description='Extract empirical difficulty, tokens, completion time from JSON result files.')
    parser.add_argument('--input_dir', required=True, help='Directory containing the JSON files (datamining.json, ai_full_difficulty.json, etc.)')
    parser.add_argument('--output', default='empirical_metrics.csv', help='Output CSV file for per-question details')
    parser.add_argument('--summary', default='empirical_summary.csv', help='Output CSV for summary statistics per subject')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return

    all_rows = []
    for json_file in input_dir.glob('*.json'):
        print(f"Processing {json_file.name}...")
        rows = process_json_file(json_file)
        all_rows.extend(rows)

    if not all_rows:
        print("No data extracted. Check JSON structure.")
        return

    # Write detailed per-question CSV
    fieldnames = list(all_rows[0].keys())
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Per-question data saved to {args.output}")

    # Compute summary statistics per subject
    subjects = defaultdict(list)
    for row in all_rows:
        subjects[row['subject']].append(row)

    summary_rows = []
    for subj, rows in subjects.items():
        n = len(rows)
        # Filter numeric fields
        numeric_fields = ['completion_time', 'tokens', 'latency', 'stem_word_count', 'correct_word_count',
                          'avg_distractor_word_count', 'stem_sentence_count', 'avg_word_length',
                          'avg_distractor_overlap', 'max_distractor_similarity', 'avg_distractor_similarity',
                          'content_word_density', 'flesch_reading_ease', 'flesch_kincaid_grade',
                          'ari', 'gunning_fog', 'coleman_liau', 'smog']
        agg = {'subject': subj, 'n': n}
        for field in numeric_fields:
            # Get all non-None values
            vals = [r[field] for r in rows if r.get(field) is not None]
            if vals:
                agg[f'avg_{field}'] = sum(vals) / len(vals)
                agg[f'std_{field}'] = (sum((x - agg[f'avg_{field}'])**2 for x in vals) / len(vals)) ** 0.5
            else:
                agg[f'avg_{field}'] = None
                agg[f'std_{field}'] = None
        # Accuracy: correct / total
        corrects = [r['correct'] for r in rows if r.get('correct') is not None]
        if corrects:
            agg['accuracy'] = sum(corrects) / len(corrects)
        # Difficulty distribution
        diffs = [r['empirical_difficulty'] for r in rows if r.get('empirical_difficulty')]
        if diffs:
            for d in ['Easy', 'Medium', 'Hard']:
                agg[f'pct_{d}'] = diffs.count(d) / len(diffs) * 100
        summary_rows.append(agg)

    summary_fieldnames = list(summary_rows[0].keys())
    with open(args.summary, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary statistics saved to {args.summary}")

if __name__ == '__main__':
    main()