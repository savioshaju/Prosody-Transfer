"""
benchmark_syntactic_aligner.py — Comprehensive Dataset Benchmark for Standalone Syntactic Aligner.
"""

import os
import re
import sys
import time
import pandas as pd

# Project root setup
_curr_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_curr_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aligner.standalone_syntactic_aligner import StandaloneSyntacticAligner


def normalize_sandhi(text: str) -> str:
    """Normalize Malayalam chillus and sandhi glides for constituent matching."""
    t = text.strip()
    t = t.replace("ൽ", "ല").replace("ൾ", "ള").replace("ൻ", "ന").replace("ൺ", "ണ").replace("ർ", "ര")
    t = re.sub(r"[യവ]$", "", t)
    return t


def run_benchmark():
    csv_path = os.path.join(PROJECT_ROOT, "Synthetic_data - Sheet1.csv")
    df = pd.read_csv(csv_path)

    print(f"Starting Standalone Syntactic Aligner Benchmark on {len(df)} rows from {os.path.basename(csv_path)}...\n")
    aligner = StandaloneSyntacticAligner()

    focus_matches = 0
    focus_evaluated = 0
    total_pairs = 0
    total_coverage = 0.0
    phrase_counts = {"NP": 0, "PP": 0, "VP": 0, "AdvP": 0, "XP": 0}
    sample_results = []
    t0 = time.time()

    for idx, row in df.iterrows():
        eng = str(row["eng"]).strip()
        mal = str(row["mal"]).strip()
        foc = str(row["Source_Prosody_word"]).strip()
        tgt = str(row["Target"]).strip()

        res = aligner.align(eng, mal)
        pairs = res["aligned_pairs"]
        total_pairs += len(pairs)
        total_coverage += res.get("coverage", 0.0)

        for pa in res.get("phrase_alignments", []):
            ptype = pa.get("en_type", "XP")
            phrase_counts[ptype] = phrase_counts.get(ptype, 0) + 1

        # Check focus projection
        if foc:
            focus_evaluated += 1
            fres = aligner.align_focus(eng, mal, foc)
            sel = fres.get("selected_constituent", "")

            # Ground truth constituent from Target sentence:
            # Target has aanu attached to the focused constituent (e.g. ഡൽഹിയിലാണ്, ഭക്ഷണമാണ്)
            gt_match = re.search(r"([\w\u0D00-\u0D7F]+)(?:ാണ്|ആണ്|യാണ്|മായാണ്|ാൺ|വാൺ|ആയിട്ടാണ്)", tgt)
            is_hit = False
            gt_stem = ""
            if gt_match:
                gt_stem = gt_match.group(1)
                norm_sel = normalize_sandhi(sel)
                norm_gt = normalize_sandhi(gt_stem)
                is_hit = bool(norm_gt in norm_sel or norm_sel in norm_gt or any(normalize_sandhi(w) in norm_sel for w in gt_stem.split()))
                if is_hit:
                    focus_matches += 1

            if idx < 15:
                sample_results.append({
                    "idx": idx,
                    "eng_focus": foc,
                    "projected": sel,
                    "gt_target_word": gt_stem,
                    "match": is_hit
                })

    elapsed = time.time() - t0
    print("=" * 80)
    print("STANDALONE SYNTACTIC ALIGNER BENCHMARK REPORT (127 SENTENCES)")
    print("=" * 80)
    print(f"Dataset:                          Synthetic_data - Sheet1.csv")
    print(f"Total Sentences Evaluated:        {len(df)}")
    print(f"Total Benchmark Execution Time:   {elapsed:.2f} seconds")
    print(f"Average Latency per Sentence:     {elapsed/len(df)*1000:.1f} ms")
    print(f"Processing Throughput:            {len(df)/elapsed:.1f} sentences/second")
    print(f"Total Aligned Word/Token Pairs:   {total_pairs}")
    print(f"Average Aligned Pairs / Sentence: {total_pairs/len(df):.1f}")
    print(f"Mean Lexical Token Coverage:      {total_coverage/len(df)*100:.1f}%")
    print(f"Focus Projection Match Count:     {focus_matches} / {focus_evaluated} ({focus_matches/focus_evaluated*100:.1f}%)")
    print("-" * 80)
    print("Syntactic Phrase Alignments by Type:")
    for k, v in phrase_counts.items():
        print(f"  {k:<5}: {v:4d} phrases (avg {v/len(df):.2f}/sentence)")
    print("-" * 80)
    print("Sample Qualitative Focus Projections (First 15 Sentences):")
    print(f"{'Status':<8} | {'Row':<4} | {'English Focus':<25} | {'Projected Malayalam':<25} | {'Ground Truth'}")
    print("-" * 80)
    for s in sample_results:
        m_sym = "[MATCH]" if s["match"] else "[MISS]"
        print(f"{m_sym:<8} | {s['idx']:<4} | {s['eng_focus']:<25} | {s['projected']:<25} | {s['gt_target_word']}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
