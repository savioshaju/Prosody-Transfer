import os
import sys

_curr_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_curr_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cleft.multi_target_emphasis_generator import (
    MultiTargetEmphasisGenerator,
    explain_sandhi_rule,
    explain_verb_norm_rule,
)


def test_sandhi_and_verb_rule_explanations():
    print("--- 1. Testing Sandhi & Verb Normalization Rule Explanations ---")

    # Sandhi explanations matching Synthetic_data patterns
    s1 = explain_sandhi_rule("2015 ലെ", "2015 ലാണ്")
    print(f"2015 ലെ rule: {s1}")
    assert "Year" in s1 and "ലാണ്" in s1

    s2 = explain_sandhi_rule("ഭക്ഷണം", "ഭക്ഷണമാണ്")
    print(f"ഭക്ഷണം rule: {s2}")
    assert "മാണ്" in s2

    s3 = explain_sandhi_rule("ഡൽഹിയിൽ", "ഡൽഹിയിലാണ്")
    print(f"ഡൽഹിയിൽ rule: {s3}")
    assert "ൽ" in s3 and "ലാണ്" in s3

    s4 = explain_sandhi_rule("കരാട്ടെ ഗെയിംസ്", "കരാട്ടെ ഗെയിംസാണ്")
    print(f"കരാട്ടെ ഗെയിംസ് rule: {s4}")
    assert "C് + ആണ്" in s4

    s5 = explain_sandhi_rule("കരാട്ടെ", "കരാട്ടെയാണ്")
    print(f"കരാട്ടെ rule: {s5}")
    assert "insert യ്" in s5

    # Verb normalization explanations
    v1 = explain_verb_norm_rule("നടത്തി", "നടത്തിയത്")
    print(f"നടത്തി rule: {v1}")
    assert "നടത്തിയത്" in v1

    v2 = explain_verb_norm_rule("കിട്ടുന്നു", "കിട്ടുന്നത്")
    print(f"കിട്ടുന്നു rule: {v2}")
    assert "കിട്ടുന്നത്" in v2

    print("✓ Rule explanations match Synthetic_data specification!\n")


def test_canonical_sentence_multi_target():
    print("--- 2. Testing Canonical Sentence Multi-Target Generation ---")
    gen = MultiTargetEmphasisGenerator(aligner_type="morpho")

    en = "The 2015 Commonwealth Karate Games were held in Delhi, India."
    ml = "2015 ലെ കോമൺവെൽത്ത് കരാട്ടെ ഗെയിംസ് ഇന്ത്യയിലെ ഡൽഹിയിൽ നടത്തി."

    results = gen.process_pair(en, ml)
    print(f"Generated {len(results)} valid emphasis target rows:")

    expected_columns = [
        "Source_Sentence",
        "Malayalam Input",
        "Target_Sentence",
        "Prosody_strategy (Emphasis/rising contour)",
        "Strategy_in_Target",
        "Target_Prosody_word",
        "Source_Prosody_word",
        "Source_Prosody_labels",
    ]

    en_words = en.split()

    found_targets = set()
    for idx, r in enumerate(results, 1):
        # 1. Verify all expected columns exist
        for col in expected_columns:
            assert col in r, f"Missing required column: {col}"

        ml_emph = r["Target_Prosody_word"]
        en_emph = r["Source_Prosody_word"]
        label_raw = r["Source_Prosody_labels"]
        if label_raw.startswith("[") and label_raw.endswith("]"):
            import ast
            labels = [str(x) for x in ast.literal_eval(label_raw)]
        else:
            labels = label_raw.split()

        found_targets.add(ml_emph)

        # 2. Verify word-level prosody labels count matches English input words count
        assert len(labels) == len(en_words), (
            f"Label count mismatch: got {len(labels)} labels for {len(en_words)} English words."
        )

        # 3. Verify labels are binary 0 or 1
        assert all(l in ("0", "1") for l in labels), f"Non-binary label found: {labels}"

        # 4. Verify at least one '1' exists in prosody labels
        assert "1" in labels, "Expected at least one emphasized token in prosody labels"

        # 5. Verify Strategy_in_Target format
        strat = r["Strategy_in_Target"]
        assert strat.startswith("Focus :"), f"Strategy must start with 'Focus :', got: {strat}"
        assert "Verb:" in strat, f"Strategy must include 'Verb:', got: {strat}"

        print(f"  [{idx}] ML: {ml_emph:<25} | EN: {en_emph:<20} | Labels: {' '.join(labels)}")

    # 6. Verify that key dataset targets from Synthetic_data - Sheet1.csv are realized:
    # Row 2: ഡൽഹിയിലാണ് ↔ Delhi -> [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    # Row 4: കരാട്ടെ ഗെയിംസാണ് ↔ Karate Games -> [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    # Row 6: 2015 ലാണ് ↔ The 2015 -> [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    target_dict = {r["Target_Prosody_word"]: r for r in results}
    assert "ഡൽഹിയിലാണ്" in target_dict, "Expected 'ഡൽഹിയിലാണ്' target"
    assert target_dict["ഡൽഹിയിലാണ്"]["Source_Prosody_word"] == "Delhi"
    assert target_dict["ഡൽഹിയിലാണ്"]["Source_Prosody_labels"] == "[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]"

    assert "കരാട്ടെ ഗെയിംസാണ്" in target_dict, "Expected 'കരാട്ടെ ഗെയിംസാണ്' target"
    assert target_dict["കരാട്ടെ ഗെയിംസാണ്"]["Source_Prosody_word"] == "Karate Games"
    assert target_dict["കരാട്ടെ ഗെയിംസാണ്"]["Source_Prosody_labels"] == "[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]"

    assert "2015 ലാണ്" in target_dict, "Expected '2015 ലാണ്' target"
    assert target_dict["2015 ലാണ്"]["Source_Prosody_word"] == "The 2015"
    assert target_dict["2015 ലാണ്"]["Source_Prosody_labels"] == "[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]"

    # Also verify individual word targets
    assert "കോമൺവെൽത്താണ്" in found_targets, "Expected 'കോമൺവെൽത്താണ്' target"
    assert "കരാട്ടെയാണ്" in found_targets, "Expected 'കരാട്ടെയാണ്' target"
    assert "ഗെയിംസാണ്" in found_targets, "Expected 'ഗെയിംസാണ്' target"

    print("✓ Canonical multi-target generation passed all assertions!\n")


def test_second_sentence_multi_target():
    print("--- 3. Testing Second Sentence Multi-Target Generation ---")
    gen = MultiTargetEmphasisGenerator(aligner_type="morpho")

    en = "Father bought a book in the garden yesterday."
    ml = "അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ പുസ്തകം വാങ്ങി."

    results = gen.process_pair(en, ml)
    print(f"Generated {len(results)} valid emphasis target rows:")

    found_targets = set()
    en_words = en.split()
    for idx, r in enumerate(results, 1):
        ml_emph = r["Target_Prosody_word"]
        en_emph = r["Source_Prosody_word"]
        label_raw = r["Source_Prosody_labels"]
        if label_raw.startswith("[") and label_raw.endswith("]"):
            import ast
            labels = [str(x) for x in ast.literal_eval(label_raw)]
        else:
            labels = label_raw.split()
        found_targets.add(ml_emph)

        assert len(labels) == len(en_words)
        print(f"  [{idx}] ML: {ml_emph:<20} | EN: {en_emph:<15} | Labels: {' '.join(labels)}")

    assert len(results) >= 2, f"Expected at least 2 candidates, got {len(results)}"
    assert "അച്ഛനാണ്" in found_targets or "പുസ്തകമാണ്" in found_targets or "തോട്ടത്തിലാണ്" in found_targets
    print("✓ Second sentence multi-target generation passed all assertions!\n")


if __name__ == "__main__":
    test_sandhi_and_verb_rule_explanations()
    test_canonical_sentence_multi_target()
    test_second_sentence_multi_target()
    print("==================================================")
    print(" ALL MULTI-TARGET EMPHASIS GENERATOR TESTS PASSED!")
    print("==================================================")
