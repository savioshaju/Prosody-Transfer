"""
benchmark_aligners_cleft_comprehensive.py — Comprehensive Benchmark of 3 Alignment Methods on Synthetic_data.

Evaluates on all 127 rows of Synthetic_data - Sheet1.csv:
  1. Standalone Syntactic Aligner (Symbolic phrase chunking + grammatical constraints)
  2. Morpho-SimAlign (mBERT + mlmorph compound splitting & case stripping)
  3. Raw SimAlign (mBERT token-level itermax)

Calculates for each:
  - Total Sentences Evaluated
  - Correct Focus Projection (Focus Accuracy)
  - Successful Clefting Completed (Clefting happened to given/projected alignment)
  - Correct Cleft Matches to Target (Strict exact match & Normalized match)
  - Cleft Failures:
      a) Failed due to wrong focus alignment (alignment was incorrect)
      b) Failed due to cleft engine / grammatical constraints (alignment was correct, but cleft failed/PE routed)
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
import time
import json
import re
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

eval_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(eval_dir) if os.path.basename(eval_dir) == "evaluator" else eval_dir
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cleft.awesome_cleft_pipeline import AwesomeCleftPipeline
from aligner.alignment_utils import strip_punctuation
from evaluator.export_mismatches import clean_no_punct


def extract_target_gold_focus(target_sent: str, mal_sent: str) -> str:
    """Extract gold focused constituent from Target sentence."""
    aanu_patterns = [
        r"(\b[\w\u0D00-\u0D7F]+(?:ാണ്|യാണ്|നാണ്|ത്താണ്|ലാണ്|ിലെയാണ്|ലെയാണ്|ആണ്|ആണു്))\b"
    ]
    for pat in aanu_patterns:
        m = re.search(pat, target_sent)
        if m:
            cleft_word = m.group(1)
            base = cleft_word
            for sfx in ("ിലെയാണ്", "ലെയാണ്", "ത്താണ്", "ാണ്", "യാണ്", "നാണ്", "ലാണ്", "ആണ്", "ആണു്"):
                if base.endswith(sfx):
                    base = base[:-len(sfx)]
                    break
            return base.strip()
    return ""


def normalize_sandhi(text: str) -> str:
    t = text.strip().replace("ൽ", "ല").replace("ൾ", "ള").replace("ൻ", "ന").replace("ൺ", "ണ").replace("ർ", "ര")
    return re.sub(r"[യവ]$", "", t)


def check_focus_match(proj: str, gold: str) -> bool:
    if not proj or not gold:
        return False
    c_proj = clean_no_punct(proj)
    c_gold = clean_no_punct(gold)
    if not c_proj or not c_gold:
        return False
    if c_proj == c_gold:
        return True
    if c_gold in c_proj or c_proj in c_gold:
        return True
    # Word-level overlap
    p_words = set(c_proj.split())
    g_words = set(c_gold.split())
    if p_words & g_words:
        return True
    # Sandhi normalized check
    n_proj = normalize_sandhi(proj)
    n_gold = normalize_sandhi(gold)
    return (n_gold in n_proj or n_proj in n_gold)


def evaluate_pipeline(pipeline_name: str, pipeline: AwesomeCleftPipeline, df: pd.DataFrame):
    print(f"\nEvaluating: {pipeline_name} on {len(df)} sentences...", flush=True)
    t0 = time.time()
    results = []

    for idx, row in df.iterrows():
        eng = str(row["eng"]).strip() if pd.notna(row["eng"]) else ""
        mal = str(row["mal"]).strip() if pd.notna(row["mal"]) else ""
        foc = str(row["Source_Prosody_word"]).strip() if pd.notna(row["Source_Prosody_word"]) else ""
        tgt = str(row["Target"]).strip() if pd.notna(row["Target"]) else ""

        gold_focus = extract_target_gold_focus(tgt, mal)

        try:
            res = pipeline.process(
                english_sentence=eng,
                malayalam_sentence=mal,
                english_focus=foc,
            )
            proj_focus = res.get("focused_malayalam_constituent", "")
            cleft_out = res.get("emphasized_malayalam_sentence", "") or ""
            cleft_dict = res.get("cleft_pipeline_output", {}) or {}

            # Did focus projection match target focus?
            focus_correct = check_focus_match(proj_focus, gold_focus)

            # Clefting happened check (to the given/projected alignment)
            status = cleft_dict.get("status", "")
            copula_form = cleft_dict.get("copula_form", "")
            norm_verb = cleft_dict.get("nominalized_verb", "") or cleft_dict.get("normalized_verb", "")
            main_verb = cleft_dict.get("main_verb", "")

            has_copula = any(cop in copula_form for cop in ("ാണ്", "ആണ്", "യാണ്", "മായാണ്", "ാൺ", "വാൺ", "ആയിട്ടാണ്")) or any(cop in cleft_out for cop in ("ാണ്", "ആണ്", "യാണ്", "മായാണ്", "ാൺ", "വാൺ", "ആയിട്ടാണ്"))
            has_norm_verb = bool(
                norm_verb and norm_verb != main_verb and any(norm_verb.endswith(sfx) for sfx in ("ത്", "തു്", "ച്ചത്", "ട്ടത്", "ന്നത്", "ത്തത്", "ുന്നത്"))
            ) or (not main_verb and has_copula) or bool(norm_verb and any(norm_verb.endswith(sfx) for sfx in ("ത്", "തു്", "ച്ചത്", "ട്ടത്", "ന്നത്", "ത്തത്", "ുന്നത്")))

            cleft_happened = bool(status == "VALID" and cleft_out and cleft_out != mal and "<PE>" not in cleft_out and has_copula and has_norm_verb)

            # Target match checks
            exact_match = (clean_no_punct(cleft_out) == clean_no_punct(tgt)) if cleft_out and tgt else False

            results.append({
                "idx": idx,
                "eng": eng,
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

        if (idx + 1) % 25 == 0 or (idx + 1) == len(df):
            print(f"  [{idx + 1}/{len(df)}] processed...", flush=True)

    elapsed = time.time() - t0
    total = len(results)

    # Metrics
    correct_focus_cnt = sum(1 for r in results if r["focus_correct"])
    cleft_happened_cnt = sum(1 for r in results if r["cleft_happened"])
    correct_cleft_cnt = sum(1 for r in results if r["exact_match"])

    # Failures breakdown:
    # 1. Failed because cleft didn't match target
    # 1a. Failed due to wrong alignment:
    failed_due_to_wrong_align = sum(1 for r in results if not r["focus_correct"])
    # 1b. Failed despite correct alignment (cleft engine failed, PE routed, or grammatical divergence):
    failed_despite_correct_align = sum(1 for r in results if r["focus_correct"] and not r["exact_match"])
    # 1c. Cleft generation completely failed / blocked:
    cleft_completely_failed = sum(1 for r in results if not r["cleft_happened"])

    summary = {
        "pipeline_name": pipeline_name,
        "total_evaluated": total,
        "time_seconds": elapsed,
        "avg_latency_ms": elapsed / total * 1000,
        "throughput_sent_sec": total / elapsed,
        "correct_focus_alignments": correct_focus_cnt,
        "focus_accuracy_pct": correct_focus_cnt / total * 100,
        "cleft_happened_count": cleft_happened_cnt,
        "cleft_happened_pct": cleft_happened_cnt / total * 100,
        "correct_cleft_matches": correct_cleft_cnt,
        "correct_cleft_pct": correct_cleft_cnt / total * 100,
        "failed_due_to_wrong_alignment": failed_due_to_wrong_align,
        "failed_despite_correct_alignment": failed_despite_correct_align,
        "cleft_generation_failed": cleft_completely_failed,
    }

    return summary, results


def main():
    csv_path = os.path.join(PROJECT_ROOT, "Synthetic_data - Sheet1.csv")
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print(f"BENCHMARKING 3 ALIGNERS ON COMPLETE DATASET: {len(df)} ROWS")
    print("=" * 80)

    # Pipeline 1: Standalone Syntactic Aligner
    p_syn = AwesomeCleftPipeline(aligner_type="syntactic")
    sum_syn, res_syn = evaluate_pipeline("Standalone Syntactic Aligner", p_syn, df)

    # Pipeline 2: Morpho-SimAlign
    p_morph = AwesomeCleftPipeline(aligner_type="morpho")
    sum_morph, res_morph = evaluate_pipeline("Morpho-SimAlign", p_morph, df)

    # Pipeline 3: Raw SimAlign
    p_raw = AwesomeCleftPipeline(aligner_type="simalign")
    sum_raw, res_raw = evaluate_pipeline("Raw SimAlign", p_raw, df)

    all_summaries = [sum_syn, sum_morph, sum_raw]

    print("\n" + "=" * 95)
    print("EXACT BENCHMARK RESULTS ACROSS ALL 127 SENTENCES")
    print("=" * 95)
    header = (
        f"{'Aligner Pipeline':<28} | {'Focus Acc':<14} | {'Cleft Happened':<15} | "
        f"{'Target Cleft Match':<18} | {'Fail (Wrong Align)':<18} | {'Fail (Grammar/PE)'}"
    )
    print(header)
    print("-" * 115)

    for s in all_summaries:
        row_str = (
            f"{s['pipeline_name']:<28} | "
            f"{s['correct_focus_alignments']:>3}/{s['total_evaluated']} ({s['focus_accuracy_pct']:.1f}%) | "
            f"{s['cleft_happened_count']:>3}/{s['total_evaluated']} ({s['cleft_happened_pct']:.1f}%) | "
            f"{s['correct_cleft_matches']:>3}/{s['total_evaluated']} ({s['correct_cleft_pct']:.1f}%) | "
            f"{s['failed_due_to_wrong_alignment']:>3}/{s['total_evaluated']} ({s['failed_due_to_wrong_alignment']/s['total_evaluated']*100:.1f}%) | "
            f"{s['failed_despite_correct_alignment']:>3}/{s['total_evaluated']}"
        )
        print(row_str)
    print("=" * 115)

    out_file = os.path.join(PROJECT_ROOT, "benchmark_127_full_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summaries": all_summaries, "results_syn": res_syn, "results_morph": res_morph, "results_raw": res_raw}, f, ensure_ascii=False, indent=2)
    print(f"\nFull report written to {out_file}")


if __name__ == "__main__":
    main()
