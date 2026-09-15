"""
Morpho-SimAlign focused evaluation on Synthetic_data - Sheet1.csv.
Reports the exact metrics requested:
  1. Successfully clefted sentences
  2. Failed clefting attempts
  3. Mismatched among successful clefts
  4. Caused by incorrect focus alignment (among mismatched)
  5. Caused by incorrect alignment (among failed clefting)
"""

import os, sys, re, time, json
from typing import Tuple
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

eval_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(eval_dir) if os.path.basename(eval_dir) == "evaluator" else eval_dir
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from cleft.awesome_cleft_pipeline import AwesomeCleftPipeline
from aligner.alignment_utils import strip_punctuation
from evaluator.export_mismatches import clean_no_punct


def extract_target_gold_focus(target_sent: str, mal_sent: str = "") -> Tuple[str, str]:
    words = target_sent.replace(".", " ").replace(",", " ").split()
    for w in words:
        for sfx in ("യിലെയാണ്", "ിലെയാണ്", "ലെയാണ്", "ത്താണ്", "ാണ്", "യാണ്", "നാണ്", "ലാണ്", "ആണ്", "ആണു്"):
            if w.endswith(sfx):
                base = w[:-len(sfx)]
                return base.strip(), w.strip()
    return "", ""


def normalize_sandhi(text: str) -> str:
    t = text.strip().replace("ൽ", "ല").replace("ൾ", "ള").replace("ൻ", "ന").replace("ൺ", "ണ").replace("ർ", "ര")
    return re.sub(r"[യവ]$", "", t)


def check_focus_match(proj: str, gold_base: str, gold_word: str = "") -> bool:
    if not proj:
        return False
    c_proj = clean_no_punct(proj)
    c_base = clean_no_punct(gold_base)
    c_word = clean_no_punct(gold_word)
    if not c_proj:
        return False
    if c_proj == c_base or (c_word and c_proj == c_word):
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


def main():
    csv_path = os.path.join(PROJECT_ROOT, "Synthetic_data - Sheet1.csv")
    df = pd.read_csv(csv_path)
    total = len(df)

    print("=" * 80)
    print(f"MORPHO-SIMALIGN EVALUATION ON COMPLETE DATASET: {total} ROWS")
    print("=" * 80)

    pipeline = AwesomeCleftPipeline(aligner_type="morpho")

    results = []
    t0 = time.time()

    for idx, row in df.iterrows():
        eng = str(row["eng"]).strip() if pd.notna(row["eng"]) else ""
        mal = str(row["mal"]).strip() if pd.notna(row["mal"]) else ""
        foc = str(row["Source_Prosody_word"]).strip() if pd.notna(row["Source_Prosody_word"]) else ""
        tgt = str(row["Target"]).strip() if pd.notna(row["Target"]) else ""

        gold_focus, gold_focus_word = extract_target_gold_focus(tgt, mal)

        try:
            res = pipeline.process(
                english_sentence=eng,
                malayalam_sentence=mal,
                english_focus=foc,
            )
            proj_focus = res.get("focused_malayalam_constituent", "")
            cleft_out = res.get("emphasized_malayalam_sentence", "") or ""
            cleft_dict = res.get("cleft_pipeline_output", {}) or {}

            focus_correct = check_focus_match(proj_focus, gold_focus, gold_focus_word)

            status = cleft_dict.get("status", "")
            copula_form = cleft_dict.get("copula_form", "")
            norm_verb = cleft_dict.get("nominalized_verb", "") or cleft_dict.get("normalized_verb", "")
            main_verb = cleft_dict.get("main_verb", "")

            has_copula = any(cop in copula_form for cop in ("ാണ്", "ആണ്", "യാണ്", "മായാണ്", "ാൺ", "വാൺ", "ആയിട്ടാണ്")) or any(cop in cleft_out for cop in ("ാണ്", "ആണ്", "യാണ്", "മായാണ്", "ാൺ", "വാൺ", "ആയിട്ടാണ്"))
            has_norm_verb = bool(
                norm_verb and norm_verb != main_verb and any(norm_verb.endswith(sfx) for sfx in ("ത്", "തു്", "ച്ചത്", "ട്ടത്", "ന്നത്", "ത്തത്", "ുന്നത്"))
            ) or (not main_verb and has_copula) or bool(norm_verb and any(norm_verb.endswith(sfx) for sfx in ("ത്", "തു്", "ച്ചത്", "ട്ടത്", "ന്നത്", "ത്തത്", "ുന്നത്")))

            cleft_happened = bool(status == "VALID" and cleft_out and cleft_out != mal and "<PE>" not in cleft_out and has_copula and has_norm_verb)

            exact_match = (clean_no_punct(cleft_out) == clean_no_punct(tgt)) if cleft_out and tgt else False

            results.append({
                "idx": idx,
                "eng": eng,
                "mal": mal,
                "focus": foc,
                "gold_focus": gold_focus,
                "proj_focus": proj_focus,
                "focus_correct": focus_correct,
                "cleft_happened": cleft_happened,
                "cleft_out": cleft_out,
                "target": tgt,
                "exact_match": exact_match,
                "status": status,
                "error": cleft_dict.get("error", ""),
            })
        except Exception as e:
            results.append({
                "idx": idx,
                "eng": eng,
                "mal": mal,
                "focus": foc,
                "gold_focus": gold_focus,
                "proj_focus": "",
                "focus_correct": False,
                "cleft_happened": False,
                "cleft_out": "",
                "target": tgt,
                "exact_match": False,
                "status": "ERROR",
                "error": str(e),
            })

        if (idx + 1) % 25 == 0 or (idx + 1) == total:
            print(f"  [{idx + 1}/{total}] processed...", flush=True)

    elapsed = time.time() - t0

    # ==========================================
    # EXACT METRICS
    # ==========================================
    total_n = len(results)

    # 1. Successfully clefted
    successfully_clefted = [r for r in results if r["cleft_happened"]]
    cleft_success_cnt = len(successfully_clefted)

    # 2. Clefting failed
    cleft_failed = [r for r in results if not r["cleft_happened"]]
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
    focus_correct_cnt = sum(1 for r in results if r["focus_correct"])

    print("\n" + "=" * 80)
    print("MORPHO-SIMALIGN — DETAILED EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Sentences Evaluated: {total_n}")
    print(f"Elapsed: {elapsed:.1f}s  |  Avg Latency: {elapsed/total_n*1000:.0f} ms/sent  |  Throughput: {total_n/elapsed:.1f} sent/sec")
    print()

    print(f"Focus Projection Accuracy: {focus_correct_cnt}/{total_n} ({focus_correct_cnt/total_n*100:.1f}%)")
    print()

    print("-" * 80)
    print("A. CLEFTING OUTCOMES")
    print("-" * 80)
    print(f"  Successfully Clefted:        {cleft_success_cnt}/{total_n} ({cleft_success_cnt/total_n*100:.1f}%)")
    print(f"  Clefting Failed/Blocked:     {cleft_failed_cnt}/{total_n} ({cleft_failed_cnt/total_n*100:.1f}%)")
    print()

    print("-" * 80)
    print("B. AMONG SUCCESSFULLY CLEFTED SENTENCES")
    print("-" * 80)
    print(f"  Matched Gold Target (Exact): {cleft_matched_cnt}/{cleft_success_cnt} ({cleft_matched_cnt/cleft_success_cnt*100:.1f}%)" if cleft_success_cnt else "  N/A")
    print(f"  Mismatched (Not Gold):       {cleft_mismatched_cnt}/{cleft_success_cnt} ({cleft_mismatched_cnt/cleft_success_cnt*100:.1f}%)" if cleft_success_cnt else "  N/A")
    print()
    if cleft_mismatched_cnt:
        print(f"    Mismatched due to WRONG Alignment:   {mismatched_wrong_align_cnt}/{cleft_mismatched_cnt} ({mismatched_wrong_align_cnt/cleft_mismatched_cnt*100:.1f}%)")
        print(f"    Mismatched despite CORRECT Alignment: {mismatched_correct_align_cnt}/{cleft_mismatched_cnt} ({mismatched_correct_align_cnt/cleft_mismatched_cnt*100:.1f}%)")
    print()

    print("-" * 80)
    print("C. AMONG FAILED/BLOCKED CLEFTING ATTEMPTS")
    print("-" * 80)
    if cleft_failed_cnt:
        print(f"  Failed due to WRONG Alignment:   {failed_wrong_align_cnt}/{cleft_failed_cnt} ({failed_wrong_align_cnt/cleft_failed_cnt*100:.1f}%)")
        print(f"  Failed despite CORRECT Alignment: {failed_correct_align_cnt}/{cleft_failed_cnt} ({failed_correct_align_cnt/cleft_failed_cnt*100:.1f}%)")
    print()

    print("=" * 80)

    # Save per-sentence details
    out_file = os.path.join(PROJECT_ROOT, "morpho_simalign_eval_detailed.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
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
            },
            "per_sentence": results
        }, f, ensure_ascii=False, indent=2)
    print(f"Detailed per-sentence results saved to: {out_file}")


if __name__ == "__main__":
    main()
