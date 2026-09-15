import os
import sys

_curr_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_curr_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cleft.preverbal_focus_reorderer import PreverbalFocusReorderer, is_wh_question
from cleft.awesome_cleft_pipeline import AwesomeCleftPipeline


def test_wh_question_blocking():
    print("--- Testing WH Question Blocking ---")
    
    # English WH
    is_wh, reason = is_wh_question("Who beat you?")
    assert is_wh, f"Expected WH question, got {is_wh}"
    print(f"✓ English WH blocked: {reason}")

    is_wh, reason = is_wh_question("What did Father buy in the garden?")
    assert is_wh, f"Expected WH question, got {is_wh}"
    print(f"✓ English WH blocked: {reason}")

    # Malayalam WH
    is_wh, reason = is_wh_question("നിന്നെ ആരാണ് തല്ലിയത്?")
    assert is_wh, f"Expected WH question, got {is_wh}"
    print(f"✓ Malayalam WH blocked: {reason}")

    is_wh, reason = is_wh_question("അച്ഛൻ എവിടെയാണ് പോയത്?")
    assert is_wh, f"Expected WH question, got {is_wh}"
    print(f"✓ Malayalam WH blocked: {reason}")

    # Declarative sentence (NOT WH)
    is_wh, reason = is_wh_question("അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി.")
    assert not is_wh, f"Expected non-WH, got {is_wh}: {reason}"
    print("✓ Declarative Malayalam sentence allowed.")

    is_wh, reason = is_wh_question("Father bought a book in the garden yesterday.")
    assert not is_wh, f"Expected non-WH, got {is_wh}: {reason}"
    print("✓ Declarative English sentence allowed.")


def test_preverbal_subject_focus():
    print("\n--- Testing Subject Focus (Jayaseelan 2023 Spec-FocP) ---")
    reorderer = PreverbalFocusReorderer()

    # SOV: S(കുട്ടി) O(ആനയെ) V(കണ്ടു)
    # With focus on Subject 'കുട്ടി' -> moves preverbal: O(ആനയെ) FOC:S(കുട്ടി) V(കണ്ടു)
    res = reorderer.reorder(
        original_sentence="കുട്ടി ആനയെ കണ്ടു",
        focused_constituent="കുട്ടി",
        main_verb="കണ്ടു"
    )
    print("Result:", res)
    assert res["applicable"] is True
    assert res["reordered_sentence"] == "ആനയെ കുട്ടി കണ്ടു"
    assert res["position_before"] == 1
    assert res["position_after"] == 2
    assert res["movement_vector"] == "1 -> 2"
    print(f"✓ Subject Focus successfully placed preverbal: '{res['reordered_sentence']}'")


def test_preverbal_adverb_focus():
    print("\n--- Testing Adverb Focus ---")
    reorderer = PreverbalFocusReorderer()

    # S(അച്ഛൻ) Adv(ഇന്നലെ) O(പുസ്തകം) V(വാങ്ങി)
    # With focus on 'ഇന്നലെ' -> S(അച്ഛൻ) O(പുസ്തകം) FOC:Adv(ഇന്നലെ) V(വാങ്ങി)
    res = reorderer.reorder(
        original_sentence="അച്ഛൻ ഇന്നലെ പുസ്തകം വാങ്ങി",
        focused_constituent="ഇന്നലെ",
        main_verb="വാങ്ങി"
    )
    print("Result:", res)
    assert res["applicable"] is True
    assert res["reordered_sentence"] == "അച്ഛൻ പുസ്തകം ഇന്നലെ വാങ്ങി"
    assert res["position_before"] == 2
    assert res["position_after"] == 3
    print(f"✓ Adverb Focus successfully placed preverbal: '{res['reordered_sentence']}'")


def test_already_preverbal_object():
    print("\n--- Testing Already Preverbal Object (In-Situ Focus) ---")
    reorderer = PreverbalFocusReorderer()

    # S(കുട്ടി) O(ആനയെ) V(കണ്ടു)
    # Direct object 'ആനയെ' is already immediately before V(കണ്ടു)
    res = reorderer.reorder(
        original_sentence="കുട്ടി ആനയെ കണ്ടു",
        focused_constituent="ആനയെ",
        main_verb="കണ്ടു"
    )
    print("Result:", res)
    assert res["applicable"] is True
    assert res["already_preverbal"] is True
    assert res["reordered_sentence"] == "കുട്ടി ആനയെ കണ്ടു"
    print(f"✓ Direct Object correctly recognized as already preverbal: '{res['reordered_sentence']}'")


def test_multi_sentence_preverbal():
    print("\n--- Testing Multi-Sentence Preservation ---")
    reorderer = PreverbalFocusReorderer()

    # S1: അവൻ വന്നു. S2: കുട്ടി ആനയെ കണ്ടു.
    # Focus on 'കുട്ടി' in S2
    res = reorderer.reorder(
        original_sentence="അവൻ വന്നു. കുട്ടി ആനയെ കണ്ടു.",
        focused_constituent="കുട്ടി",
        main_verb="കണ്ടു"
    )
    print("Result:", res)
    assert res["applicable"] is True
    assert res["reordered_sentence"] == "അവൻ വന്നു. ആനയെ കുട്ടി കണ്ടു."
    print(f"✓ Multi-sentence preserved and clause isolated: '{res['reordered_sentence']}'")


def test_end_to_end_pipeline():
    print("\n--- Testing End-to-End AwesomeCleftPipeline ---")
    pipeline = AwesomeCleftPipeline()

    # 1. Test WH-Question blocking through pipeline
    res_wh = pipeline.process(
        english_sentence="Who bought a book in the garden?",
        malayalam_sentence="ആരാണ് തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങിയത്?",
        english_focus="Who",
    )
    assert res_wh["status"] == "BLOCKED"
    assert res_wh["pipeline_status"] == "BLOCKED_WH_QUESTION"
    print("✓ WH-question correctly blocked through AwesomeCleftPipeline.")

    # 2. Test Declarative Focus Transfer & Parallel Preverbal Reordering
    res = pipeline.process(
        english_sentence="Father bought a book in the garden yesterday.",
        malayalam_sentence="അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി.",
        english_focus="Father",
    )
    print("Pipeline Output Status:", res.get("pipeline_status", "SUCCESS"))
    print("Cleft Emphasized:", res.get("emphasized_malayalam_sentence"))
    pfr = res.get("preverbal_focus_reordering", {})
    print("Preverbal Focus Reordering:", pfr)
    assert pfr.get("applicable") is True
    print(f"✓ Preverbal Reordered Output: '{pfr.get('reordered_sentence')}'")
    print(f"✓ Movement: {pfr.get('movement_vector')}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    test_wh_question_blocking()
    test_preverbal_subject_focus()
    test_preverbal_adverb_focus()
    test_already_preverbal_object()
    test_multi_sentence_preverbal()
    test_end_to_end_pipeline()
    print("\n==============================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==============================================")
