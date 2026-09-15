import json
import os

PROJECT_ROOT = r"d:\Savio Shaju\Prosody Transfer"
json_path = os.path.join(PROJECT_ROOT, "morpho_simalign_eval_detailed.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

records = data["per_sentence"]

mismatched_correct = [r for r in records if r["cleft_happened"] and not r["exact_match"] and r["focus_correct"]]
failed_correct = [r for r in records if not r["cleft_happened"] and r["focus_correct"]]

out_path = os.path.join(PROJECT_ROOT, "evaluator", "analysis_26_cases.txt")

with open(out_path, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("GROUP 1: 13 SENTENCES MISMATCHED DESPITE CORRECT FOCUS ALIGNMENT\n")
    f.write("(Clefting succeeded, focus correctly identified, but differed from gold target)\n")
    f.write("=" * 90 + "\n\n")

    for i, r in enumerate(mismatched_correct, 1):
        f.write(f"--- [Case {i}/13] Row {r['idx']} ---\n")
        f.write(f"English:       {r['eng']}\n")
        f.write(f"English Focus: {r['focus']}\n")
        f.write(f"Projected Foc: {r['proj_focus']}\n")
        f.write(f"Gold Target:   {r['target']}\n")
        f.write(f"Generated Out: {r['cleft_out']}\n")
        f.write("\n")

    f.write("\n" + "=" * 90 + "\n")
    f.write("GROUP 2: 13 SENTENCES FAILED / BLOCKED DESPITE CORRECT FOCUS ALIGNMENT\n")
    f.write("(Focus correctly identified, but clefting failed or was blocked by syntactic gate)\n")
    f.write("=" * 90 + "\n\n")

    for i, r in enumerate(failed_correct, 1):
        f.write(f"--- [Case {i}/13] Row {r['idx']} ---\n")
        f.write(f"English:       {r['eng']}\n")
        f.write(f"English Focus: {r['focus']}\n")
        f.write(f"Projected Foc: {r['proj_focus']}\n")
        f.write(f"Malayalam In:  {r['mal']}\n")
        f.write(f"Gold Target:   {r['target']}\n")
        f.write(f"Pipeline Stat: {r['status']}\n")
        f.write(f"Reason/Error:  {r['error']}\n")
        f.write("\n")

print(f"Written analysis to {out_path}")
