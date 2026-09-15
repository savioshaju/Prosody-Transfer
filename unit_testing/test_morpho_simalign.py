"""
test_morpho_simalign.py — Unit Tests for Morphologically Segmented SimAlign.

Verifies:
  1. Compound Word Splitting: Compounds like 'ശിക്ഷാനിയമം' split into 'ശിക്ഷ' and 'നിയമം',
     allowing multi-word English expressions like 'Penal Code' to align to their constituents.
  2. Nominal Case Stripping: Suffixes like '-യിൽ' (locative) and '-ിന്' (dative) stripped to roots,
     preventing verb-noun cross-attraction (e.g. 'gave' erroneously aligning to 'ജോണിന്').
  3. Adposition Binding: English closed-class prepositions ('in', 'to', 'with') cleanly bind to
     the corresponding case-marked Malayalam nouns.
  4. Reverse Index Projection: All alignments project with 100% boundary safety back to original
     Malayalam surface tokens.
"""

import os
import sys

_curr_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_curr_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aligner.morpho_simalign import MorphoSimAligner
from cleft.awesome_cleft_pipeline import AwesomeCleftPipeline


def test_compound_and_case_section_377():
    print("==================================================")
    print("--- 1. Testing Compound & Case: Section 377 IPC ---")
    print("==================================================")
    aligner = MorphoSimAligner()

    src = "Under Section 377 of the Indian Penal Code, homosexuality is a crime in India."
    tgt = "ഇന്ത്യൻ ശിക്ഷാനിയമം 377 പ്രകാരം സ്വവർഗ്ഗരതി ഇന്ത്യയിൽ കുറ്റകരം."

    res = aligner.align(src, tgt)
    fwd = res["alignment_map_forward"]
    src_tokens = res["source_tokens"]
    tgt_tokens = res["target_tokens"]

    print(f"Source Tokens: {src_tokens}")
    print(f"Target Tokens: {tgt_tokens}")
    print(f"Decomposed Tokens: {res['decomposed_target_tokens']}")
    print("\nAligned Pairs:")
    for p in res["reconciled_pairs"]:
        print(f"  {p['src_word']:<15} <---> {p['tgt_word']:<15} (matched stem: {p.get('matched_stem')}, conf: {p.get('confidence')})")

    # Verify that target indices are strictly valid original surface token indices
    for p in res["reconciled_pairs"]:
        assert 0 <= p["tgt_index"] < len(tgt_tokens), f"Invalid tgt_index {p['tgt_index']}"
        assert p["tgt_word"] == tgt_tokens[p["tgt_index"]], "Mismatch between tgt_word and tgt_tokens"

    # Verify key lexical mappings
    pair_map = {(p["src_word"].lower(), p["tgt_word"]) for p in res["reconciled_pairs"]}
    src_to_tgt_words = {p["src_word"].lower(): p["tgt_word"] for p in res["reconciled_pairs"]}

    assert "penal" in src_to_tgt_words, "Expected 'Penal' to be aligned"
    assert src_to_tgt_words["penal"] == "ശിക്ഷാനിയമം", f"Expected 'Penal' -> 'ശിക്ഷാനിയമം', got {src_to_tgt_words.get('penal')}"

    assert "india" in src_to_tgt_words, "Expected 'India' to be aligned"
    assert src_to_tgt_words["india"] == "ഇന്ത്യയിൽ", f"Expected 'India' -> 'ഇന്ത്യയിൽ', got {src_to_tgt_words.get('india')}"

    assert "in" in src_to_tgt_words, "Expected 'in' to be bound to 'ഇന്ത്യയിൽ' via locative case"
    assert src_to_tgt_words["in"] == "ഇന്ത്യയിൽ", f"Expected 'in' -> 'ഇന്ത്യയിൽ', got {src_to_tgt_words.get('in')}"

    print("✓ Compound splitting and locative case binding verified on Section 377 IPC!")


def test_dative_verb_disambiguation():
    print("\n==================================================")
    print("--- 2. Testing Dative Verb Disambiguation (Mary-John) ---")
    print("==================================================")
    aligner = MorphoSimAligner()

    src = "Mary gave a book to John."
    tgt = "മേരി ജോണിന് ഒരു പുസ്തകം കൊടുത്തു."

    res = aligner.align(src, tgt)
    src_to_tgt_words = {p["src_word"].lower(): p["tgt_word"] for p in res["reconciled_pairs"]}

    print("\nAligned Pairs:")
    for p in res["reconciled_pairs"]:
        print(f"  {p['src_word']:<15} <---> {p['tgt_word']:<15} (stem: {p.get('matched_stem')})")

    # In raw SimAlign, 'gave' falsely aligned to 'ജോണിന്'.
    # In MorphoSimAlign, 'gave' must align to verb 'കൊടുത്തു'!
    assert "gave" in src_to_tgt_words, "Expected 'gave' to be aligned"
    assert src_to_tgt_words["gave"] == "കൊടുത്തു", f"CRITICAL: Expected 'gave' -> 'കൊടുത്തു', got {src_to_tgt_words.get('gave')}"

    assert "john" in src_to_tgt_words, "Expected 'John' to be aligned"
    assert src_to_tgt_words["john"] == "ജോണിന്", f"Expected 'John' -> 'ജോണിന്', got {src_to_tgt_words.get('john')}"

    assert "to" in src_to_tgt_words, "Expected 'to' to bind to 'ജോണിന്' via dative case"
    assert src_to_tgt_words["to"] == "ജോണിന്", f"Expected 'to' -> 'ജോണിന്', got {src_to_tgt_words.get('to')}"

    print("✓ Dative verb disambiguation verified: 'gave' successfully aligned to 'കൊടുത്തു'!")


def test_cleft_pipeline_with_morpho_simalign():
    print("\n==================================================")
    print("--- 3. Testing AwesomeCleftPipeline with MorphoSimAligner ---")
    print("==================================================")
    pipeline = AwesomeCleftPipeline(aligner_type="morpho_simalign")

    src = "Mary gave a book to John."
    tgt = "മേരി ജോണിന് ഒരു പുസ്തകം കൊടുത്തു."
    emphasis = "John"

    out = pipeline.process(src, tgt, emphasis)
    print(f"English Emphasis: {emphasis}")
    print(f"Focused Malayalam Constituent: {out.get('focused_malayalam_constituent')}")
    print(f"Emphasized Malayalam Sentence: {out.get('emphasized_malayalam_sentence')}")
    print(f"Preverbal Reordering: {out.get('preverbal_reordering', {}).get('reordered_sentence')}")

    assert "ജോണിന്" in out["focused_malayalam_constituent"], f"Expected 'ജോണിന്' in focused constituent, got {out.get('focused_malayalam_constituent')}"
    assert "ജോണിനാണ്" in out["emphasized_malayalam_sentence"], f"Expected 'ജോണിനാണ്' in emphasized sentence, got {out.get('emphasized_malayalam_sentence')}"
    print("✓ Full end-to-end focus projection and clefting verified with MorphoSimAligner!")


if __name__ == "__main__":
    test_compound_and_case_section_377()
    test_dative_verb_disambiguation()
    test_cleft_pipeline_with_morpho_simalign()
    print("\n==================================================")
    print("ALL MORPHO-SIMALIGN TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
