"""
verb_normalizer.py — Minimal Malayalam verb normalizer for cleft construction.

Responsibility
--------------
Given a finite Malayalam verb that the Cleft Controller has already decided
should be nominalized, produce its correct nominalized (ത്-ending) form.

This module does NOT decide whether a sentence can be clefted.

Supported verb classes
---------------------
1. ALREADY_NORM — input already ends in <n><deriv> → passthrough
2. TENSE_POS    — past/present affirmative → <adv-clause-rp-{past|present}><n><deriv>
3. TENSE_NEG    — negative finite verb     → <adv-clause-rp-{past|present}-neg><n><deriv>
4. OBLIGATIVE   — obligative mood          → compound cvb path <n><deriv>
5. TENSE_POS (future) → NEEDS_VERIFICATION
6. HABITUAL_NEG       → NEEDS_VERIFICATION (pending cleft examples)
7. NON_VERB / unknown → UNRESOLVED

Everything else (existential, copular, permissive, conditional, sentence-level
decisions) belongs in the Cleft Controller.

Workflow: analyse → classify → build FST target → generate → validate.
"""

import re
import sys
import logging
from mlmorph import Analyser, Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tag constants (mlmorph tag names).
# ---------------------------------------------------------------------------
TAG_PAST    = "past"
TAG_PRESENT = "present"
TAG_FUTURE  = "future"

RP_PAST    = "adv-clause-rp-past"
RP_PRESENT = "adv-clause-rp-present"

RP_PAST_NEG    = "adv-clause-rp-past-neg"
RP_PRESENT_NEG = "adv-clause-rp-present-neg"

TAG_NOMINAL          = "<n><deriv>"
TAG_NEG              = "neg"
TAG_IMPERATIVE       = "imperative-mood"       # obligative surface tag in mlmorph
TAG_OPTATIVE         = "optative-mood"
TAG_PERMISSIVE       = "permissive-mood"
TAG_CONDITIONAL      = "conditional-mood"
TAG_HABITUAL_ASPECT  = "habitual-aspect"

# The obligative compound: verb<cvb-adv-part-simul> + അണ്ടുക<v><cvb-adv-part-absolute>
OBLIGATIVE_AUX_BLOCK = "അണ്ടുക<v><cvb-adv-part-absolute>"
TAG_CVB_SIMUL        = "cvb-adv-part-simul"

TENSE_TAGS  = {TAG_PAST, TAG_PRESENT, TAG_FUTURE}
VOICE_TAGS  = {"passive-voice", "causative-voice"}
NON_FINITE_TAGS = {
    RP_PAST, RP_PRESENT,
    RP_PAST_NEG, RP_PRESENT_NEG,
    "cvb-adv-part-past", "cvb-adv-part-absolute", TAG_CVB_SIMUL,
}

# Verb class labels
CLASS_TENSE_POS    = "tense-positive"
CLASS_TENSE_NEG    = "tense-negative"
CLASS_HABITUAL_NEG = "habitual-negative"
CLASS_OBLIGATIVE   = "obligative"
CLASS_OPTATIVE     = "optative"
CLASS_PERMISSIVE   = "permissive"
CLASS_CONDITIONAL  = "conditional"
CLASS_ALREADY_NORM = "already-normalized"
CLASS_NON_VERB     = "non-verb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_tags(raw: str) -> list:
    return re.findall(r"<([^>]+)>", raw)


def _is_already_normalized(raw: str) -> bool:
    return "<n><deriv>" in raw



# ---------------------------------------------------------------------------
# VerbAnalysis: full classification of an mlmorph raw analysis string.
# ---------------------------------------------------------------------------

class VerbAnalysis:
    def __init__(self, raw: str):
        self.raw  = raw
        self.tags = _all_tags(raw)
        self.already_normalized = _is_already_normalized(raw)

        # Locate the last <v> and extract suffix tags (tags after last <v>)
        last_v_idx = raw.rfind("<v>")
        self.has_verb = last_v_idx != -1
        suffix_str = raw[last_v_idx:] if last_v_idx != -1 else raw
        self.suffix_tags = _all_tags(suffix_str)

        # Primary tense (after the last <v>)
        self.tense = next((t for t in self.suffix_tags if t in TENSE_TAGS), None)

        # Voice (after the last <v>)
        self.voice = next((t for t in self.suffix_tags if t in VOICE_TAGS), None)

        # Negation
        self.has_neg = TAG_NEG in self.suffix_tags

        # Lemma: text before first <
        self.lemma = raw.split("<")[0] if "<" in raw else raw

        # Non-finite tag (participial) already present
        self.non_finite_tag = next(
            (t for t in self.suffix_tags if t in NON_FINITE_TAGS), None
        )

        # Mood classification (after the last <v>)
        self.is_obligative   = TAG_IMPERATIVE in self.suffix_tags
        self.is_optative     = TAG_OPTATIVE in self.suffix_tags
        self.is_permissive   = TAG_PERMISSIVE in self.suffix_tags
        self.is_conditional  = TAG_CONDITIONAL in self.suffix_tags

        # Habitual-aspect + neg
        self.is_habitual_neg = TAG_HABITUAL_ASPECT in self.suffix_tags and self.has_neg

    def verb_class(self) -> str:
        if self.already_normalized:
            return CLASS_ALREADY_NORM
        if not self.has_verb:
            return CLASS_NON_VERB
        if self.is_habitual_neg:
            return CLASS_HABITUAL_NEG
        if self.is_obligative:
            return CLASS_OBLIGATIVE
        if self.is_optative:
            return CLASS_OPTATIVE
        if self.is_permissive:
            return CLASS_PERMISSIVE
        if self.is_conditional:
            return CLASS_CONDITIONAL
        if self.has_neg:
            return CLASS_TENSE_NEG
        return CLASS_TENSE_POS

    @property
    def rp_tag_positive(self):
        """Tense → RP tag for positive (non-negated) verbs."""
        if self.tense == TAG_PAST:
            return RP_PAST
        if self.tense == TAG_PRESENT:
            return RP_PRESENT
        # Future tense: no automatic mapping to present RP.
        # Returns None; dispatch marks this as NEEDS_VERIFICATION.
        return None

    @property
    def rp_tag_negative(self):
        """Tense → RP-neg tag for negated verbs."""
        if self.tense == TAG_PAST:
            return RP_PAST_NEG
        # present/future/tenseless negative → present-neg
        return RP_PRESENT_NEG


# ---------------------------------------------------------------------------
# Generation target builders (one per verb class)
# ---------------------------------------------------------------------------

def _targets_tense_positive(raw: str, va: VerbAnalysis) -> list:
    rp = va.rp_tag_positive
    if rp is None:
        return []
    last_v_pos     = raw.rfind("<v>")
    prefix         = raw[:last_v_pos + len("<v>")]
    voice_clause   = f"<{va.voice}>" if va.voice else ""
    tense_clause   = f"<{va.tense}>" if va.tense else ""
    primary        = prefix + voice_clause + f"<{rp}>" + TAG_NOMINAL
    with_tense     = prefix + tense_clause + voice_clause + f"<{rp}>" + TAG_NOMINAL
    return [primary, with_tense]


def _targets_tense_negative(raw: str, va: VerbAnalysis) -> list:
    """
    Negative finite verbs: strip <tense><neg> from the suffix and append
    <adv-clause-rp-{past|present}-neg><n><deriv>.

    Examples:
      കണ്ടില്ല  → കാണുക<v><past>ഇല്ല<neg>  → കാണുക<v><adv-clause-rp-past-neg><n><deriv>
      കാണുന്നില്ല → കാണുക<v><present>ഇല്ല<neg> → കാണുക<v><adv-clause-rp-present-neg><n><deriv>
    """
    rp_neg = va.rp_tag_negative
    last_v_pos = raw.rfind("<v>")
    prefix     = raw[:last_v_pos + len("<v>")]
    primary    = prefix + f"<{rp_neg}>" + TAG_NOMINAL
    return [primary]


def _targets_obligative(raw: str, va: VerbAnalysis) -> list:
    """
    Obligative verbs: mlmorph tags these as <imperative-mood>.
    The study's target form (e.g. എഴുതേണ്ടത്) is generated by the compound
    analysis: lemma<v><cvb-adv-part-simul>അണ്ടുക<v><cvb-adv-part-absolute><n><deriv>
    """
    last_v_pos = raw.rfind("<v>")
    prefix     = raw[:last_v_pos + len("<v>")]
    target     = prefix + f"<{TAG_CVB_SIMUL}>{OBLIGATIVE_AUX_BLOCK}" + TAG_NOMINAL
    return [target]

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# All RP tag names that count as valid in re-analysis
_ALL_RP_TAGS = [
    RP_PAST, RP_PRESENT,
    RP_PAST_NEG, RP_PRESENT_NEG,
]

def _is_valid_nominalized_verb(form: str, analyser: Analyser) -> bool:
    results = analyser.analyse(form)
    if not results:
        return False
    for raw, _ in results:
        if "<n><deriv>" not in raw:
            continue
        # Standard relative-participle paths (past / present / neg / habitual)
        if any(f"<{rp}>" in raw for rp in _ALL_RP_TAGS):
            return True
        # Obligative compound path:
        #   lemma<v><cvb-adv-part-simul>അണ്ടുക<v><cvb-adv-part-absolute><n><deriv>
        # This is how mlmorph re-analyses forms like എഴുതേണ്ടത്, വരേണ്ടത്, etc.
        # Detection is purely tag-driven: both participial tags must co-occur with <n><deriv>.
        if "<cvb-adv-part-simul>" in raw and "<cvb-adv-part-absolute><n><deriv>" in raw:
            return True
    return False


# ---------------------------------------------------------------------------
# Main normalizer
# ---------------------------------------------------------------------------

class VerbNormalizer:
    """Standalone Malayalam verb normalizer."""

    def __init__(self):
        self._analyser  = Analyser()
        self._generator = Generator()

    def normalize(self, verb: str) -> dict:
        result = {
            "original"            : verb,
            "lemma"               : "",
            "analysis"            : "",
            "verb_class"          : "",
            "rp_tag"              : "",
            "gen_target"          : "",
            "derived_stem"        : "",
            "normalized"          : None,
            "normalized_analysis" : "",
            "status"              : "UNRESOLVED",
            "failure_reason"      : "",
        }

        # Step 1: Analyse
        raw_analyses = self._analyser.analyse(verb)
        if not raw_analyses:
            result["failure_reason"] = "mlmorph returned no analysis for this form."
            return result

        # Pick best: prefer analyses containing <v>; lower weight = better.
        def _score(entry):
            raw, weight = entry
            return weight + (-200 if "<v>" in raw else 0)

        raw_analyses_sorted = sorted(raw_analyses, key=_score)
        best_raw, _ = raw_analyses_sorted[0]

        result["analysis"] = best_raw
        va = VerbAnalysis(best_raw)
        result["lemma"]      = va.lemma
        result["verb_class"] = va.verb_class()

        # ----------------------------------------------------------------
        # Dispatch by verb class
        # ----------------------------------------------------------------

        vc = va.verb_class()

        # -- Already normalized --
        if vc == CLASS_ALREADY_NORM:
            result["normalized"]          = verb
            result["rp_tag"]              = va.non_finite_tag or ""
            result["normalized_analysis"] = best_raw
            result["derived_stem"]        = verb[:-2] if verb.endswith("\u0d24\u0d4d") else verb
            result["status"]              = "VALID"
            result["failure_reason"]      = "Input is already in nominalized form."
            return result

        # -- Non-verb --
        if vc == CLASS_NON_VERB:
            result["failure_reason"] = (
                f"Analysis '{best_raw}' contains no <v> tag; input may not be a verb."
            )
            return result

        # -- Future tense: needs linguistic verification --
        if vc == CLASS_TENSE_POS and va.tense == TAG_FUTURE:
            result["rp_tag"]         = ""
            result["status"]         = "NEEDS_VERIFICATION"
            result["failure_reason"] = (
                f"Future tense verb '{verb}'. "
                "Automatic future → present RP mapping removed; needs verification."
            )
            return result

        # -- Obligative mood: needs verification --
        if vc == CLASS_OBLIGATIVE:
            result["rp_tag"]         = ""
            result["status"]         = "NEEDS_VERIFICATION"
            result["failure_reason"] = (
                f"Obligative mood in '{best_raw}'. "
                "Compound obligative nominalization (e.g. ചെയ്യണം → ചെയ്യേണ്ടത്) is marked NEEDS_VERIFICATION."
            )
            return result

        # -- Permissive / Optative mood: needs verification --
        if vc in (CLASS_OPTATIVE, CLASS_PERMISSIVE):
            result["rp_tag"]         = ""
            result["status"]         = "NEEDS_VERIFICATION"
            result["failure_reason"] = (
                f"Permissive/optative mood in '{best_raw}'. "
                "Permissive clefting is marked NEEDS_VERIFICATION."
            )
            return result

        # -- Conditional mood: needs verification --
        if vc == CLASS_CONDITIONAL:
            result["rp_tag"]         = ""
            result["status"]         = "NEEDS_VERIFICATION"
            result["failure_reason"] = (
                f"Conditional mood in '{best_raw}'. "
                "Conditional clefting is marked NEEDS_VERIFICATION."
            )
            return result

        # -- Habitual negative: needs cleft examples to confirm --
        if vc == CLASS_HABITUAL_NEG:
            result["rp_tag"]         = ""
            result["status"]         = "NEEDS_VERIFICATION"
            result["failure_reason"] = (
                f"Habitual-aspect negative in '{best_raw}'. "
                "FST path exists but needs confirmed cleft examples before enabling."
            )
            return result

        # -- Build targets for verified classes (past/present affirmative, negative) --
        if vc == CLASS_TENSE_NEG:
            targets = _targets_tense_negative(best_raw, va)
            rp_display = va.rp_tag_negative
        elif vc == CLASS_TENSE_POS and va.rp_tag_positive:
            targets = _targets_tense_positive(best_raw, va)
            rp_display = va.rp_tag_positive
        else:
            result["status"] = "NEEDS_VERIFICATION"
            result["failure_reason"] = (
                f"Unverified or unsupported verb structure in analysis '{best_raw}'."
            )
            return result

        result["rp_tag"] = rp_display

        if not targets:
            result["failure_reason"] = (
                f"Could not determine tense from analysis '{best_raw}' to build generation target."
            )
            return result

        # Step 6: Generate
        generated_form = None
        used_target    = None

        for target in targets:
            gen_results = self._generator.generate(target)
            if not gen_results:
                continue
            # Prefer canonical ത് form; exclude തു് variant
            canonical  = [(f, w) for f, w in gen_results if not f.endswith("\u0d24\u0d41\u0d4d")]
            candidates = canonical if canonical else gen_results
            candidates.sort(key=lambda x: x[1])
            generated_form = candidates[0][0]
            used_target    = target
            break

        result["gen_target"] = used_target or targets[0]

        if generated_form is None:
            result["failure_reason"] = (
                f"mlmorph Generator returned no result for targets: {targets}. "
                f"Analysis: '{best_raw}'."
            )
            return result

        # Step 7: Derive stem for display
        result["derived_stem"] = (
            generated_form[:-2] if generated_form.endswith("\u0d24\u0d4d") else generated_form
        )

        # Step 8: Validate by re-analysis
        is_valid  = _is_valid_nominalized_verb(generated_form, self._analyser)
        norm_analyses = self._analyser.analyse(generated_form)
        norm_raw      = norm_analyses[0][0] if norm_analyses else ""
        if is_valid:
            result["status"] = "VALID"
            result["normalized"] = generated_form
        else:
            result["status"] = "UNRESOLVED"
            result["normalized"] = None
            result["failure_reason"] = (
                f"Generated '{generated_form}' but re-analysis did not confirm "
                f"<adv-clause-rp-*><n><deriv>. Re-analysis: '{norm_raw}'."
            )

        return result


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def print_result(r: dict) -> None:
    print(f"Original           : {r['original']}")
    print(f"Lemma              : {r['lemma']}")
    print(f"mlmorph Analysis   : {r['analysis']}")
    print(f"Verb Class         : {r['verb_class']}")
    print(f"RP / Morph Tag     : {r['rp_tag']}")
    print(f"Gen Target         : {r['gen_target']}")
    print(f"Derived Stem       : {r['derived_stem']}")
    print(f"Normalized         : {r['normalized']}")
    print(f"Normalized Analysis: {r['normalized_analysis']}")
    print(f"Status             : {r['status']}")
    if r["failure_reason"]:
        print(f"Note               : {r['failure_reason']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python verb_normalizer.py <Malayalam-verb>")
        sys.exit(1)

    verb = sys.argv[1]
    normalizer = VerbNormalizer()
    result = normalizer.normalize(verb)
    print_result(result)
