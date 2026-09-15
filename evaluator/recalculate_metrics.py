import json
import re
import os

PROJECT_ROOT = r"d:\Savio Shaju\Prosody Transfer"
json_path = os.path.join(PROJECT_ROOT, "morpho_simalign_eval_detailed.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

records = data["per_sentence"]
total_n = len(records)

def clean_no_punct(text):
    if not text:
        return ""
    return re.sub(r"[\s.,!?\"\'\(\)\-_:;]+", "", text).strip()

def normalize_sandhi(text):
    t = text.strip().replace("ൽ", "ല").replace("ൾ", "ള").replace("ൻ", "ന").replace("ൺ", "ണ").replace("ർ", "ര")
    return re.sub(r"[യവ]$", "", t)

def get_gold_focus(target):
    words = target.replace(".", " ").replace(",", " ").split()
    for w in words:
        for sfx in ("യിലെയാണ്", "ിലെയാണ്", "ലെയാണ്", "ത്താണ്", "ാണ്", "യാണ്", "നാണ്", "ലാണ്", "ആണ്", "ആണു്"):
            if w.endswith(sfx):
                base = w[:-len(sfx)]
                return base.strip(), w.strip()
    return "", ""

def check_focus_match(proj, gold_base, gold_word):
    if not proj:
        return False
    c_proj = clean_no_punct(proj)
    c_base = clean_no_punct(gold_base)
    c_word = clean_no_punct(gold_word)
    if not c_proj:
        return False
    if c_proj == c_base or c_proj == c_word:
        return True
    if c_base and (c_base in c_proj or c_proj in c_base):
        return True
    if c_word and (c_word in c_proj or c_proj in c_word):
        return True
    p_words = set(proj.split())
    b_words = set(gold_base.split())
    if p_words & b_words:
        return True
    n_proj = normalize_sandhi(c_proj)
    n_base = normalize_sandhi(c_base)
    return bool(n_base and (n_base in n_proj or n_proj in n_base))

for r in records:
    base, full = get_gold_focus(r["target"])
    r["gold_focus"] = base
    r["gold_focus_word"] = full
    r["focus_correct"] = check_focus_match(r["proj_focus"], base, full)

# 1. Successfully clefted
successfully_clefted = [r for r in records if r["cleft_happened"]]
cleft_success_cnt = len(successfully_clefted)

# 2. Clefting failed
cleft_failed = [r for r in records if not r["cleft_happened"]]
cleft_failed_cnt = len(cleft_failed)

# 3. Among successfully clefted: how many MATCHED the gold target exactly?
cleft_matched = [r for r in successfully_clefted if r["exact_match"]]
cleft_matched_cnt = len(cleft_matched)

# 4. Among successfully clefted: how many were MISMATCHED (clefting happened but didn't match gold)?
cleft_mismatched = [r for r in successfully_clefted if not r["exact_match"]]
cleft_mismatched_cnt = len(cleft_mismatched)

# 5. Among mismatched successful clefts: how many had INCORRECT focus alignment?
mismatched_wrong_align = [r for r in cleft_mismatched if not r["focus_correct"]]
mismatched_wrong_align_cnt = len(mismatched_wrong_align)

# 6. Among mismatched successful clefts: how many had CORRECT alignment but still mismatched?
mismatched_correct_align = [r for r in cleft_mismatched if r["focus_correct"]]
mismatched_correct_align_cnt = len(mismatched_correct_align)

# 7. Among failed clefting: how many were due to INCORRECT alignment?
failed_wrong_align = [r for r in cleft_failed if not r["focus_correct"]]
failed_wrong_align_cnt = len(failed_wrong_align)

# 8. Among failed clefting: how many had CORRECT alignment but still failed?
failed_correct_align = [r for r in cleft_failed if r["focus_correct"]]
failed_correct_align_cnt = len(failed_correct_align)

# 9. Focus projection accuracy overall
focus_correct_cnt = sum(1 for r in records if r["focus_correct"])

print("=" * 80)
print(f"MORPHO-SIMALIGN EVALUATION METRICS (RECALCULATED)")
print("=" * 80)
print(f"Total Sentences: {total_n}")
print(f"Overall Focus Alignment Accuracy: {focus_correct_cnt}/{total_n} ({focus_correct_cnt/total_n*100:.1f}%)")
print()
print("A. CLEFTING OUTCOMES:")
print(f"  1. Successfully clefted: {cleft_success_cnt}/{total_n} ({cleft_success_cnt/total_n*100:.1f}%)")
print(f"  2. Clefting failed:     {cleft_failed_cnt}/{total_n} ({cleft_failed_cnt/total_n*100:.1f}%)")
print()
print("B. AMONG SUCCESSFULLY CLEFTED:")
print(f"  - Matched Gold Target:   {cleft_matched_cnt}/{cleft_success_cnt} ({cleft_matched_cnt/cleft_success_cnt*100:.1f}%)")
print(f"  3. Mismatched (Not Gold): {cleft_mismatched_cnt}/{cleft_success_cnt} ({cleft_mismatched_cnt/cleft_success_cnt*100:.1f}%)")
print(f"     4. Caused by INCORRECT alignment: {mismatched_wrong_align_cnt}/{cleft_mismatched_cnt} ({mismatched_wrong_align_cnt/cleft_mismatched_cnt*100:.1f}%)")
print(f"     - Mismatched despite CORRECT alignment: {mismatched_correct_align_cnt}/{cleft_mismatched_cnt} ({mismatched_correct_align_cnt/cleft_mismatched_cnt*100:.1f}%)")
print()
print("C. AMONG FAILED CLEFTING:")
print(f"  5. Caused by INCORRECT alignment: {failed_wrong_align_cnt}/{cleft_failed_cnt} ({failed_wrong_align_cnt/cleft_failed_cnt*100:.1f}%)")
print(f"  - Failed despite CORRECT alignment: {failed_correct_align_cnt}/{cleft_failed_cnt} ({failed_correct_align_cnt/cleft_failed_cnt*100:.1f}%)")
print("=" * 80)

# Save updated json
data["summary"] = {
    "total": total_n,
    "focus_correct": focus_correct_cnt,
    "successfully_clefted": cleft_success_cnt,
    "cleft_failed": cleft_failed_cnt,
    "cleft_matched_gold": cleft_matched_cnt,
    "cleft_mismatched": cleft_mismatched_cnt,
    "mismatched_wrong_align": mismatched_wrong_align_cnt,
    "mismatched_correct_align": mismatched_correct_align_cnt,
    "failed_wrong_align": failed_wrong_align_cnt,
    "failed_correct_align": failed_correct_align_cnt,
}
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
