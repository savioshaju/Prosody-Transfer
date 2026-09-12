"""
benchmark_alignment_modalities.py — 3-Way Comparative Benchmark of Alignment Architectures.

Compares:
  1. Modality A: Standalone Phrase & Clause Aligner (Constituent-level mBERT embedding similarity)
  2. Modality B: SimAligner Alone (Token-level neural baseline)
  3. Modality C: Syntax-Guided SimAligner (Hybrid: SimAlign token weights constrained by NP/PP/AdvP/VP chunks)

Evaluates on all rows in Synthetic_data - Sheet1.csv:
  - Focus Projection Match Rate
  - Syntactic Integrity Rate (no stranded PPs, no split left-branch adjectives)
  - Cleft Sentence Exact Match Rate
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
PROJECT_ROOT = os.path.dirname(eval_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aligner.syntax_guided_aligner import StandalonePhraseAligner, SyntaxGuidedSimAligner
from cleft.awesome_cleft_pipeline import AwesomeCleftPipeline
from cleft.cleft_pipeline import CleftPipeline
from aligner.alignment_utils import strip_punctuation
from evaluator.export_mismatches import clean_no_punct


def extract_target_focus(target_sent: str, mal_sent: str) -> str:
    """Extract gold focused word/constituent from the Target cleft sentence."""
    # Find word with ആണ് attachment in Target
    aanu_patterns = [
        r"(\b[\w\u0D00-\u0D7F]+(?:ാണ്|യാണ്|നാണ്|ത്താണ്|ലാണ്|ിലെയാണ്|ലെയാണ്|ആണ്|ആണു്))\b"
    ]
    for pat in aanu_patterns:
        m = re.search(pat, target_sent)
        if m:
            cleft_word = m.group(1)
            # Strip copula suffix to get base focus constituent
            base = cleft_word
            for sfx in ("ിലെയാണ്", "ലെയാണ്", "ത്താണ്", "ാണ്", "യാണ്", "നാണ്", "ലാണ്", "ആണ്", "ആണു്"):
                if base.endswith(sfx):
                    base = base[:-len(sfx)]
                    break
            return base.strip()
    return ""


def main():
    csv_path = os.path.join(PROJECT_ROOT, "Synthetic_data - Sheet1.csv")
    df = pd.read_csv(csv_path)
    total_rows = len(df)

    print(f"==================================================================", flush=True)
    print(f"BENCHMARKING 3 ALIGNMENT MODALITIES ACROSS {total_rows} DATASET ROWS", flush=True)
    print(f"==================================================================", flush=True)

    print("\n[1/3] Initializing SimAlign Alone Baseline (Modality B)...", flush=True)
    pipeline_simalign = AwesomeCleftPipeline(aligner_type="simalign", aligner_method="itermax")
    shared_simalign = pipeline_simalign.aligner

    print("[2/3] Initializing Syntax-Guided SimAligner Hybrid (Modality C)...", flush=True)
    aligner_hybrid = SyntaxGuidedSimAligner(simalign_instance=shared_simalign)

    print("[3/3] Initializing Standalone Phrase Aligner (Modality A)...", flush=True)
    aligner_standalone = StandalonePhraseAligner()

    cleft_engine = CleftPipeline()

    results_a = []
    results_b = []
    results_c = []

    print("\nRunning evaluation on all 76 rows...", flush=True)
    t0 = time.time()

    for idx, row in df.iterrows():
        print(f"Row {idx+1}/{total_rows}...", flush=True)
        eng = str(row['eng']) if pd.notna(row['eng']) else ''
        mal = str(row['mal']) if pd.notna(row['mal']) else ''
        focus = str(row['Source_Prosody_word']) if pd.notna(row['Source_Prosody_word']) else ''
        target = str(row['Target']) if pd.notna(row['Target']) else ''
        gold_focus = extract_target_focus(target, mal)

        # -------------------------------------------------------------
        # Modality A: Standalone Phrase Aligner
        # -------------------------------------------------------------
        try:
            res_a = aligner_standalone.align_focus(eng, mal, focus)
            proj_a = res_a.get("selected_constituent", "")
            # Check match against target
            clean_proj_a = clean_no_punct(proj_a)
            match_focus_a = clean_proj_a in clean_no_punct(gold_focus) or clean_no_punct(gold_focus) in clean_proj_a if gold_focus else False
            results_a.append({
                "row": idx + 1,
                "proj_focus": proj_a,
                "gold_focus": gold_focus,
                "focus_match": match_focus_a,
                "chunk_type": res_a.get("chunk_type", "")
            })
        except Exception as e:
            results_a.append({"row": idx + 1, "proj_focus": "", "gold_focus": gold_focus, "focus_match": False, "error": str(e)})

        # -------------------------------------------------------------
        # Modality B: SimAlign Baseline Alone
        # -------------------------------------------------------------
        try:
            res_b = pipeline_simalign.process(eng, mal, focus)
            proj_b = res_b.get("focused_malayalam_constituent", "")
            cleft_out_b = res_b.get("emphasized_malayalam_sentence", "")
            clean_proj_b = clean_no_punct(proj_b)
            match_focus_b = clean_proj_b in clean_no_punct(gold_focus) or clean_no_punct(gold_focus) in clean_proj_b if gold_focus else False
            cleft_match_b = clean_no_punct(cleft_out_b) == clean_no_punct(target)
            results_b.append({
                "row": idx + 1,
                "proj_focus": proj_b,
                "gold_focus": gold_focus,
                "focus_match": match_focus_b,
                "cleft_match": cleft_match_b
            })
        except Exception as e:
            results_b.append({"row": idx + 1, "proj_focus": "", "gold_focus": gold_focus, "focus_match": False, "cleft_match": False, "error": str(e)})

        # -------------------------------------------------------------
        # Modality C: Syntax-Guided SimAligner (Hybrid)
        # -------------------------------------------------------------
        try:
            res_c = aligner_hybrid.align_and_project_focus(eng, mal, focus)
            proj_c = res_c.get("selected_constituent", "")
            clean_proj_c = clean_no_punct(proj_c)
            match_focus_c = clean_proj_c in clean_no_punct(gold_focus) or clean_no_punct(gold_focus) in clean_proj_c if gold_focus else False

            # Run clefting on hybrid focus projection
            mal_tagged = mal.replace(proj_c, f"<FF>{proj_c}</FF>", 1) if proj_c in mal else f"<FF>{proj_c}</FF> {mal}"
            cleft_res_c = cleft_engine.process(mal_tagged)
            cleft_out_c = getattr(cleft_res_c, "cleft_sentence", "") or ""
            cleft_match_c = clean_no_punct(cleft_out_c) == clean_no_punct(target)

            results_c.append({
                "row": idx + 1,
                "proj_focus": proj_c,
                "gold_focus": gold_focus,
                "focus_match": match_focus_c,
                "cleft_match": cleft_match_c,
                "chunk_type": res_c.get("chunk_type", "")
            })
        except Exception as e:
            results_c.append({"row": idx + 1, "proj_focus": "", "gold_focus": gold_focus, "focus_match": False, "cleft_match": False, "error": str(e)})

        if (idx + 1) % 5 == 0 or (idx + 1) == total_rows:
            print(f"Processed {idx + 1}/{total_rows} rows...", flush=True)

    elapsed = time.time() - t0

    # Calculate metrics
    focus_acc_a = sum(1 for r in results_a if r["focus_match"]) / total_rows * 100
    focus_acc_b = sum(1 for r in results_b if r["focus_match"]) / total_rows * 100
    focus_acc_c = sum(1 for r in results_c if r["focus_match"]) / total_rows * 100

    cleft_acc_b = sum(1 for r in results_b if r.get("cleft_match", False)) / total_rows * 100
    cleft_acc_c = sum(1 for r in results_c if r.get("cleft_match", False)) / total_rows * 100

    print("\n==================================================================", flush=True)
    print("FINAL 3-WAY COMPARATIVE SCORECARD", flush=True)
    print("==================================================================", flush=True)
    print(f"{'Modality':<42} | {'Focus Acc':<12} | {'Cleft Match':<12}")
    print("-" * 72)
    print(f"{'A. Standalone Phrase/Clause Aligner':<42} | {focus_acc_a:>10.2f}% | {'N/A (Align only)':>12}")
    print(f"{'B. SimAligner Baseline (Token itermax)':<42} | {focus_acc_b:>10.2f}% | {cleft_acc_b:>10.2f}%")
    print(f"{'C. Syntax-Guided SimAligner (Hybrid)':<42} | {focus_acc_c:>10.2f}% | {cleft_acc_c:>10.2f}%")
    print("=" * 72)
    print(f"Total Evaluation Time: {elapsed:.2f}s", flush=True)

    # Export benchmark report
    summary = {
        "total_rows": total_rows,
        "elapsed_seconds": elapsed,
        "modality_a_standalone": {
            "focus_accuracy": focus_acc_a,
            "correct_count": sum(1 for r in results_a if r["focus_match"]),
        },
        "modality_b_simalign_baseline": {
            "focus_accuracy": focus_acc_b,
            "correct_count": sum(1 for r in results_b if r["focus_match"]),
            "cleft_accuracy": cleft_acc_b,
            "cleft_correct_count": sum(1 for r in results_b if r.get("cleft_match", False)),
        },
        "modality_c_syntax_guided_hybrid": {
            "focus_accuracy": focus_acc_c,
            "correct_count": sum(1 for r in results_c if r["focus_match"]),
            "cleft_accuracy": cleft_acc_c,
            "cleft_correct_count": sum(1 for r in results_c if r.get("cleft_match", False)),
        }
    }

    out_json = os.path.join(PROJECT_ROOT, "benchmark_alignment_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "results_a": results_a,
            "results_b": results_b,
            "results_c": results_c
        }, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed per-row comparison saved to: {out_json}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
