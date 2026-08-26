"""
filter_focus_only.py — Filter and Save Only True FOCUS / CLEFT Sentences from Datasets/english_malayalam.csv.
"""

import os
import sys
import csv
import json

# Ensure UTF-8 output encoding for Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def filter_focus_sentences():
    in_csv = os.path.join("Datasets", "english_malayalam_aanu_classified.csv")
    out_csv = os.path.join("Datasets", "english_malayalam_focus_only.csv")
    out_json = os.path.join("Datasets", "english_malayalam_focus_only.json")

    if not os.path.exists(in_csv):
        print(f"Error: Classification input file not found at '{in_csv}'")
        return

    print("=" * 80)
    print(" FILTERING TRUE FOCUS / CLEFT SENTENCES FROM CLASSIFICATION RESULTS")
    print("=" * 80)

    clean_focus_records = []

    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("category") == "FOCUS_EMPHASIS":
                clean_focus_records.append({
                    "english": row.get("english", "").strip(),
                    "malayalam": row.get("malayalam", "").strip(),
                })

    total_focus = len(clean_focus_records)
    print(f"\nExtracted {total_focus} True FOCUS / CLEFT sentences out of the classified dataset.\n")

    # Save to CSV (ONLY english and malayalam columns)
    fieldnames = ["english", "malayalam"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_focus_records)

    # Save to JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(clean_focus_records, f, ensure_ascii=False, indent=2)

    print(f" Focus-only sentences saved to CSV  : '{out_csv}'")
    print(f" Focus-only sentences saved to JSON : '{out_json}'")

    print("\n" + "=" * 80)
    print(" SAMPLE EXTRACTED STRUCTURAL CLEFT SENTENCES")
    print("=" * 80)
    for i, sample in enumerate(clean_focus_records[:5], 1):
        print(f"\n[Sample {i}]")
        print(f"  English   : \"{sample['english']}\"")
        print(f"  Malayalam : \"{sample['malayalam']}\"")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    filter_focus_sentences()
