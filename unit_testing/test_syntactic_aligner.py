import os
import sys

_curr_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_curr_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aligner.standalone_syntactic_aligner import StandaloneSyntacticAligner
from cleft.awesome_cleft_pipeline import AwesomeCleftPipeline


def test_standalone_phrase_alignment():
    print("==================================================")
    print("--- 1. Testing Standalone Syntactic Phrase Alignment ---")
    print("==================================================")
    aligner = StandaloneSyntacticAligner()

    en_sent = "Father bought a book in the garden yesterday."
    ml_sent = "അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി."

    res = aligner.align(en_sent, ml_sent)
    print(f"Aligner Name: {res.get('aligner_name')}")
    print(f"Source Tokens: {res['source_tokens']}")
    print(f"Target Tokens: {res['target_tokens']}")
    print("\nAligned Pairs:")
    for p in res["aligned_pairs"]:
        print(f"  {p[0]:<20} <---> {p[1]:<20}")

    print("\nPhrase Alignments:")
    for pa in res["phrase_alignments"]:
        print(f"  [{pa['en_type']}] '{pa['en_chunk']}' <---> [{pa['ml_type']}] '{pa['ml_chunk']}' (score: {pa['score']})")

    # Verify key alignments
    aligned_dict = dict(res["aligned_pairs"])
    assert "Father" in aligned_dict, "Expected 'Father' to be aligned."
    assert aligned_dict["Father"] == "അച്ഛൻ", f"Expected 'Father' -> 'അച്ഛൻ', got {aligned_dict['Father']}"
    assert "bought" in aligned_dict, "Expected 'bought' to be aligned."
    assert aligned_dict["bought"] == "വാങ്ങി", f"Expected 'bought' -> 'വാങ്ങി', got {aligned_dict['bought']}"
    assert "yesterday" in aligned_dict, "Expected 'yesterday' to be aligned."
    assert aligned_dict["yesterday"] == "ഇന്നലെ", f"Expected 'yesterday' -> 'ഇന്നലെ', got {aligned_dict['yesterday']}"
    print("\n✓ Core syntactic and lexical alignments verified!")


def test_focus_projection_pp():
    print("\n==================================================")
    print("--- 2. Testing Focus Projection on PP (in the garden) ---")
    print("==================================================")
    aligner = StandaloneSyntacticAligner()

    en_sent = "Father bought a book in the garden yesterday."
    ml_sent = "അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി."

    # Focus on PP: "in the garden"
    focus_res = aligner.align_focus(en_sent, ml_sent, "in the garden")
    print(f"Projected Focus: '{focus_res['selected_constituent']}'")
    print(f"Span Range     : {focus_res['span_range']}")
    assert "തോട്ടത്തിൽ" in focus_res["selected_constituent"], f"Expected 'തോട്ടത്തിൽ' in projected focus, got {focus_res['selected_constituent']}"
    assert "വെച്ച്" in focus_res["selected_constituent"], f"Expected postposition 'വെച്ച്' to be absorbed, got {focus_res['selected_constituent']}"
    print("✓ Postpositional phrase successfully projected with Dravidian island constraint (no P-stranding)!")


def test_dative_io_alignment():
    print("\n==================================================")
    print("--- 3. Testing Dative Indirect Object (to John) ---")
    print("==================================================")
    aligner = StandaloneSyntacticAligner()

    en_sent = "Mary gave a letter to John."
    ml_sent = "മേരി ജോണിന് ഒരു കത്ത് കൊടുത്തു."

    res = aligner.align(en_sent, ml_sent)
    print("Aligned Pairs:")
    for p in res["aligned_pairs"]:
        print(f"  {p[0]:<20} <---> {p[1]:<20}")

    aligned_dict = dict(res["aligned_pairs"])
    assert "Mary" in aligned_dict and aligned_dict["Mary"] == "മേരി"
    assert "gave" in aligned_dict and aligned_dict["gave"] == "കൊടുത്തു"
    assert "letter" in aligned_dict and aligned_dict["letter"] == "കത്ത്"
    assert "John" in aligned_dict and aligned_dict["John"] == "ജോണിന്"
    print("✓ Dative IO and Named Entity alignments verified!")


def test_e2e_pipeline_with_syntactic_aligner():
    print("\n==================================================")
    print("--- 4. Testing End-to-End Pipeline with Standalone Syntactic Aligner ---")
    print("==================================================")
    pipeline = AwesomeCleftPipeline(aligner_type="syntactic")

    res = pipeline.process(
        english_sentence="Father bought a book in the garden yesterday.",
        malayalam_sentence="അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി.",
        english_focus="Father",
    )

    print("Pipeline Output Status  :", res.get("pipeline_status", "SUCCESS"))
    print("Focused Malayalam Focus :", res.get("focused_malayalam_constituent"))
    print("Emphasized Malayalam    :", res.get("emphasized_malayalam_sentence"))
    pfr = res.get("preverbal_focus_reordering", {})
    print("Preverbal Reordered     :", pfr.get("reordered_sentence"))
    print("Movement Vector         :", pfr.get("movement_vector"))

    assert res.get("focused_malayalam_constituent") == "അച്ഛൻ"
    assert "അച്ഛനാണ്" in res.get("emphasized_malayalam_sentence")
    assert pfr.get("applicable") is True
    assert pfr.get("reordered_sentence") == "ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം അച്ഛൻ വാങ്ങി."
    print("✓ Full end-to-end focus transfer with Standalone Syntactic Aligner succeeded flawlessly!")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    test_standalone_phrase_alignment()
    test_focus_projection_pp()
    test_dative_io_alignment()
    test_e2e_pipeline_with_syntactic_aligner()
    print("\n==============================================")
    print("ALL STANDALONE SYNTACTIC ALIGNER TESTS PASSED!")
    print("==============================================")
