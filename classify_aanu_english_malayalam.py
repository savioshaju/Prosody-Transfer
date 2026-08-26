"""
classify_aanu_english_malayalam.py — Clean Structural Aanu vs Copula Classification for Datasets/english_malayalam.csv.
"""

import os
import sys
import csv
import json
from collections import Counter, defaultdict

# Ensure UTF-8 output encoding for Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "test"))
from analyze_dataset_emphasis import analyze_sentence_emphasis


def run_classification():
    csv_path = os.path.join("Datasets", "english_malayalam.csv")
    if not os.path.exists(csv_path):
        print(f"Error: File not found at '{csv_path}'")
        return

    print("=" * 80)
    print(" REFINED AANU vs COPULA CLASSIFICATION ON Datasets/english_malayalam.csv")
    print("=" * 80)

    rows_processed = 0
    category_counts = Counter()
    focus_type_counts = Counter()
    classified_records = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for i, row in enumerate(reader, 2):
            if not row or len(row) < 2:
                continue

            eng_text = row[0].strip().strip('"')
            mal_text = row[1].strip().strip('"')

            rows_processed += 1

            # Run linguistic classification rule engine
            res = analyze_sentence_emphasis(
                malayalam_text=mal_text,
                english_text=eng_text,
                source_file="english_malayalam.csv",
                row_id=i,
            )

            if res is None:
                category = "NON_AANU_OTHER"
                focus_type = "NONE"
                focused_constituent = "N/A"
                deciding_rule = "No ആണ് copula or focus clitic present"
                explanation = "Standard non-copular / non-cleft sentence."
            else:
                category = res["category"]
                focus_type = res["focus_type"]
                focused_constituent = res["focused_constituent"]
                deciding_rule = res["deciding_rule"]
                explanation = res["emphasis_explanation"]

            category_counts[category] += 1
            focus_type_counts[focus_type] += 1

            classified_records.append({
                "row_id": i,
                "english": eng_text,
                "malayalam": mal_text,
                "category": category,
                "focus_type": focus_type,
                "focused_constituent": focused_constituent,
                "deciding_rule": deciding_rule,
                "explanation": explanation,
            })

    # Display Breakdown Summary Report
    print(f"\nTotal Sentences Processed: {rows_processed}")
    print("-" * 80)
    print(" CATEGORY BREAKDOWN:")
    print("-" * 80)
    for cat, count in category_counts.most_common():
        percentage = (count / rows_processed) * 100
        print(f"  * {cat:<20} : {count:>5} sentences ({percentage:>6.2f}%)")

    print("\n" + "-" * 80)
    print(" DETAILED FOCUS / COPULA SUB-TYPE BREAKDOWN:")
    print("-" * 80)
    for ftype, count in focus_type_counts.most_common():
        percentage = (count / rows_processed) * 100
        print(f"  - {ftype:<30} : {count:>5} sentences ({percentage:>6.2f}%)")

    # Save output CSV and JSON
    out_csv = os.path.join("Datasets", "english_malayalam_aanu_classified.csv")
    out_json = os.path.join("Datasets", "english_malayalam_aanu_summary.json")

    fieldnames = ["row_id", "english", "malayalam", "category", "focus_type", "focused_constituent", "deciding_rule", "explanation"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(classified_records)

    summary_data = {
        "dataset": "Datasets/english_malayalam.csv",
        "total_sentences": rows_processed,
        "category_counts": dict(category_counts),
        "focus_type_counts": dict(focus_type_counts),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print(f"\n Classified dataset saved to: '{out_csv}'")
    print(f" Summary JSON saved to      : '{out_json}'")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_classification()
