"""
benchmark_comparison.py — Comparative benchmark between:
  1. StandaloneSyntacticAligner (Symbolic / Rule-based / Morpho-syntactic)
  2. MorphoSimAligner (SimAlign + mlmorph compound splitting & case stripping)
  3. Raw SimAligner (bert-base-multilingual-cased baseline)
"""

import os
import re
import sys
import time
import pandas as pd

_curr_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_curr_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aligner.standalone_syntactic_aligner import StandaloneSyntacticAligner
from aligner.morpho_simalign import MorphoSimAligner
from aligner.simalign_wrapper import SimAlignerWrapper


def normalize_sandhi(text: str) -> str:
    t = text.strip().replace("ൽ", "ല").replace("ൾ", "ള").replace("ൻ", "ന").replace("ൺ", "ണ").replace("ർ", "ര")
    return re.sub(r"[യവ]$", "", t)


def run_comparative_benchmark(num_sentences: int = 30):
    csv_path = os.path.join(PROJECT_ROOT, "Synthetic_data - Sheet1.csv")
    df = pd.read_csv(csv_path).head(num_sentences)

    print(f"Running 3-way comparative benchmark on {len(df)} sentences from {os.path.basename(csv_path)}...\n")

    aligner_syn = StandaloneSyntacticAligner()
    aligner_morph = MorphoSimAligner()
    aligner_raw = SimAlignerWrapper()

    results = {
        "Standalone Syntactic": {"matches": 0, "pairs": 0, "coverage_sum": 0.0, "time": 0.0},
        "Morpho-SimAlign":      {"matches": 0, "pairs": 0, "coverage_sum": 0.0, "time": 0.0},
        "Raw SimAlign":         {"matches": 0, "pairs": 0, "coverage_sum": 0.0, "time": 0.0},
    }

    for idx, row in df.iterrows():
        eng = str(row["eng"]).strip()
        mal = str(row["mal"]).strip()
        foc = str(row["Source_Prosody_word"]).strip()
        tgt = str(row["Target"]).strip()

        gt_match = re.search(r"([\w\u0D00-\u0D7F]+)(?:ാണ്|ആണ്|യാണ്|മായാണ്|ാൺ|വാൺ|ആയിട്ടാണ്)", tgt)
        gt_stem = normalize_sandhi(gt_match.group(1)) if gt_match else ""
        foc_words = [w.lower() for w in foc.split() if w]

        # 1. Standalone Syntactic Aligner
        t0 = time.time()
        r_syn = aligner_syn.align(eng, mal)
        f_syn = aligner_syn.align_focus(eng, mal, foc)["selected_constituent"]
        results["Standalone Syntactic"]["time"] += time.time() - t0
        results["Standalone Syntactic"]["pairs"] += len(r_syn["aligned_pairs"])
        results["Standalone Syntactic"]["coverage_sum"] += r_syn.get("coverage", 0.0)
        norm_syn = normalize_sandhi(f_syn)
        if gt_stem and (gt_stem in norm_syn or norm_syn in gt_stem or any(normalize_sandhi(w) in norm_syn for w in gt_stem.split())):
            results["Standalone Syntactic"]["matches"] += 1

        # 2. Morpho-SimAlign
        t0 = time.time()
        r_morph = aligner_morph.align(eng, mal)
        results["Morpho-SimAlign"]["time"] += time.time() - t0
        results["Morpho-SimAlign"]["pairs"] += len(r_morph["reconciled_pairs"])
        results["Morpho-SimAlign"]["coverage_sum"] += r_morph.get("coverage", 0.0)
        aligned_tgts = [p["tgt_word"] for p in r_morph["reconciled_pairs"] if p["src_word"].lower() in foc_words and p["is_lexical"]]
        f_morph = " ".join(aligned_tgts)
        norm_morph = normalize_sandhi(f_morph)
        if gt_stem and (gt_stem in norm_morph or any(gt_stem in normalize_sandhi(w) for w in aligned_tgts)):
            results["Morpho-SimAlign"]["matches"] += 1

        # 3. Raw SimAlign
        t0 = time.time()
        r_raw = aligner_raw.align(eng, mal)
        results["Raw SimAlign"]["time"] += time.time() - t0
        results["Raw SimAlign"]["pairs"] += len(r_raw["reconciled_pairs"])
        results["Raw SimAlign"]["coverage_sum"] += r_raw.get("coverage", 0.0)
        aligned_tgts_raw = [p["tgt_word"] for p in r_raw["reconciled_pairs"] if p["src_word"].lower() in foc_words and p["is_lexical"]]
        f_raw = " ".join(aligned_tgts_raw)
        norm_raw = normalize_sandhi(f_raw)
        if gt_stem and (gt_stem in norm_raw or any(gt_stem in normalize_sandhi(w) for w in aligned_tgts_raw)):
            results["Raw SimAlign"]["matches"] += 1

    N = len(df)
    print("=" * 88)
    print("THREE-WAY ALIGNER BENCHMARK COMPARISON TABLE")
    print("=" * 88)
    print(f"{'Aligner Pipeline':<24} | {'Focus Acc':<11} | {'Avg Pairs':<11} | {'Mean Cov':<10} | {'Latency':<12} | {'Throughput'}")
    print("-" * 88)
    for name, d in results.items():
        acc = d["matches"] / N * 100
        avg_pairs = d["pairs"] / N
        mean_cov = d["coverage_sum"] / N * 100
        latency_ms = d["time"] / N * 1000
        throughput = N / d["time"] if d["time"] > 0 else 0
        print(f"{name:<24} | {acc:>5.1f}% ({d['matches']}/{N}) | {avg_pairs:>6.1f} pairs | {mean_cov:>6.1f}%   | {latency_ms:>6.1f} ms/s   | {throughput:>5.1f} sent/s")
    print("=" * 88)


if __name__ == "__main__":
    run_comparative_benchmark(num_sentences=30)
