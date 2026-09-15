import os
import sys

_curr_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_curr_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cleft.cleft_pipeline import CleftPipeline
from cleft.awesome_cleft_pipeline import AwesomeCleftPipeline


def test_verbless_cleft_emphasis():
    """
    Test emphasis in verbless / equational sentences (Moag §2.5, §5.4, §11.2).
    In the absence of a finite verb:
      - Do not attempt to find or normalize a verb.
      - Attach copula 'ആണ്' to the focused constituent directly.
      - Retain non-verbal predicate (e.g. കുറ്റകരം) intact.
    """
    pipeline = CleftPipeline()

    test_cases = [
        {
            "tagged": "ഇന്ത്യൻ ശിക്ഷാ നിയമം 377 പ്രകാരം <FF>സ്വവർഗ്ഗരതി</FF> ഇന്ത്യയിൽ കുറ്റകരം.",
            "target": "സ്വവർഗ്ഗരതി",
            "expected_cleft": "ഇന്ത്യൻ ശിക്ഷാ നിയമം 377 പ്രകാരം സ്വവർഗ്ഗരതിയാണ് ഇന്ത്യയിൽ കുറ്റകരം.",
            "expected_copula": "സ്വവർഗ്ഗരതിയാണ്",
        },
        {
            "tagged": "ഇന്ത്യൻ ശിക്ഷാ നിയമം 377 പ്രകാരം സ്വവർഗ്ഗരതി <FF>ഇന്ത്യയിൽ</FF> കുറ്റകരം.",
            "target": "ഇന്ത്യയിൽ",
            "expected_cleft": "ഇന്ത്യൻ ശിക്ഷാ നിയമം 377 പ്രകാരം സ്വവർഗ്ഗരതി ഇന്ത്യയിലാണ് കുറ്റകരം.",
            "expected_copula": "ഇന്ത്യയിലാണ്",
        },
        {
            "tagged": "<FF>ഇന്ത്യൻ ശിക്ഷാ നിയമം 377 പ്രകാരം</FF> സ്വവർഗ്ഗരതി ഇന്ത്യയിൽ കുറ്റകരം.",
            "target": "ഇന്ത്യൻ ശിക്ഷാ നിയമം 377 പ്രകാരം",
            "expected_cleft": "ഇന്ത്യൻ ശിക്ഷാ നിയമം 377 പ്രകാരമാണ് സ്വവർഗ്ഗരതി ഇന്ത്യയിൽ കുറ്റകരം.",
            "expected_copula": "ഇന്ത്യൻ ശിക്ഷാ നിയമം 377 പ്രകാരമാണ്",
        },
    ]

    for tc in test_cases:
        res = pipeline.process(tc["tagged"])
        print(f"\n--- Testing Target: {tc['target']} ---")
        print(f"Status      : {res.status}")
        print(f"Copula Form : {res.copula_form}")
        print(f"Main Verb   : {res.main_verb!r}")
        print(f"Norm Verb   : {res.normalized_verb!r}")
        print(f"Cleft Output: {res.cleft_sentence}")

        assert res.status == "VALID", f"Expected VALID status, got {res.status} ({res.error})"
        assert res.copula_form == tc["expected_copula"], f"Expected copula {tc['expected_copula']}, got {res.copula_form}"
        assert res.cleft_sentence == tc["expected_cleft"], f"Expected {tc['expected_cleft']}, got {res.cleft_sentence}"
        assert res.main_verb in (None, ""), f"Expected no finite main verb, got {res.main_verb}"

    print("\n✓ All verbless equational clefting tests passed!")


def test_awesome_pipeline_verbless_e2e():
    """
    Test end-to-end AwesomeCleftPipeline cross-lingual focus transfer for verbless sentences.
    """
    pipeline = AwesomeCleftPipeline(aligner_type="syntactic")

    en_sent = "As per Section 377 of the IPC, homosexuality is a crime in India."
    ml_sent = "ഇന്ത്യൻ ശിക്ഷാ നിയമം 377 പ്രകാരം സ്വവർഗ്ഗരതി ഇന്ത്യയിൽ കുറ്റകരം."

    e2e_cases = [
        ("homosexuality", "ഇന്ത്യൻ ശിക്ഷാ നിയമം 377 പ്രകാരം സ്വവർഗ്ഗരതിയാണ് ഇന്ത്യയിൽ കുറ്റകരം."),
        ("in India", "ഇന്ത്യൻ ശിക്ഷാ നിയമം 377 പ്രകാരം സ്വവർഗ്ഗരതി ഇന്ത്യയിലാണ് കുറ്റകരം."),
        ("As per Section 377 of the IPC", "ഇന്ത്യൻ ശിക്ഷാ നിയമം 377 പ്രകാരമാണ് സ്വവർഗ്ഗരതി ഇന്ത്യയിൽ കുറ്റകരം."),
    ]

    for en_focus, expected_ml in e2e_cases:
        res = pipeline.process(
            english_sentence=en_sent,
            malayalam_sentence=ml_sent,
            english_focus=en_focus,
        )
        print(f"\n--- E2E Focus: '{en_focus}' ---")
        print(f"ML Focus    : {res.get('focused_malayalam_constituent')}")
        print(f"Emphasized  : {res.get('emphasized_malayalam_sentence')}")
        assert res.get("emphasized_malayalam_sentence") == expected_ml, (
            f"Expected {expected_ml}, got {res.get('emphasized_malayalam_sentence')}"
        )

    print("\n✓ All E2E verbless focus transfer tests passed!")


if __name__ == "__main__":
    test_verbless_cleft_emphasis()
    test_awesome_pipeline_verbless_e2e()
