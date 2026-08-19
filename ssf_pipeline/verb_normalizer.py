"""
verb_normalizer.py - Standalone Malayalam verb normalizer.

Input : One Malayalam finite verb (e.g. വായിച്ചു).
Output: The nominalized clefting form ending in ത്  (e.g. വായിച്ചത്).

Approach
--------
1. Analyse the surface verb with mlmorph to extract the raw analysis string.
2. Parse the analysis to determine the verb class:
     CLASS_TENSE_POS    – simple finite verb (past/present/future)
     CLASS_TENSE_NEG    – negative finite verb (<neg> present)
     CLASS_HABITUAL_NEG – habitual-aspect negative (<habitual-aspect>ഇല്ല<neg>)
     CLASS_OBLIGATIVE   – obligative mood (<imperative-mood>)
     CLASS_PERMISSIVE   – permissive/promissive mood (<permissive-mood>/<promissive-mood>)
     CLASS_CONDITIONAL  – conditional mood (<conditional-mood>)
     CLASS_EXISTENTIAL  – existential copula (<aff> tag, ഉണ്ട്)
     CLASS_ALREADY_NORM – already nominalized (<n><deriv> present)
     CLASS_NON_VERB     – no <v> tag found
3. For each class, determine the FST generation target:
     TENSE_POS    → lemma<v>[voice]<adv-clause-rp-{past|present}><n><deriv>
     TENSE_NEG    → lemma<v><adv-clause-rp-{past|present}-neg><n><deriv>
     HABITUAL_NEG → lemma<v><cont-perfect-aspect-neg><adv-clause-rp-past><n><deriv>
     OBLIGATIVE   → lemma<v><cvb-adv-part-simul>അണ്ടുക<v><cvb-adv-part-absolute><n><deriv>
     PERMISSIVE   → no FST path – UNRESOLVED
     CONDITIONAL  → no FST path – UNRESOLVED
     EXISTENTIAL  → irregular lexical substitute: ഉള്ളത് – UNRESOLVED (documented)
4. Generate via mlmorph Generator. Prefer canonical ത് over തു് variant.
5. Validate via re-analysis: confirm <adv-clause-rp-*><n><deriv> is present.

No individual words, suffix lists, or word-specific replacements are used.
All decisions are driven by the tag set returned by mlmorph.
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
RP_FUTURE  = "adv-clause-rp-future"   # no FST <n><deriv> path; kept for lookup

RP_PAST_NEG    = "adv-clause-rp-past-neg"
RP_PRESENT_NEG = "adv-clause-rp-present-neg"

TAG_NOMINAL          = "<n><deriv>"
TAG_NEG              = "neg"
TAG_AFF              = "aff"                  # existential copula tag
TAG_IMPERATIVE       = "imperative-mood"       # obligative surface tag in mlmorph
TAG_PERMISSIVE       = "permissive-mood"
TAG_PROMISSIVE       = "promissive-mood"
TAG_CONDITIONAL      = "conditional-mood"
TAG_HABITUAL_ASPECT  = "habitual-aspect"
TAG_CONT_PERFECT_NEG = "cont-perfect-aspect-neg"

# The obligative compound: verb<cvb-adv-part-simul> + അണ്ടുക<v><cvb-adv-part-absolute>
OBLIGATIVE_AUX_BLOCK = "അണ്ടുക<v><cvb-adv-part-absolute>"
TAG_CVB_SIMUL        = "cvb-adv-part-simul"

TENSE_TAGS  = {TAG_PAST, TAG_PRESENT, TAG_FUTURE}
VOICE_TAGS  = {"passive-voice", "causative-voice"}
NON_FINITE_TAGS = {
    RP_PAST, RP_PRESENT, RP_FUTURE,
    RP_PAST_NEG, RP_PRESENT_NEG,
    "cvb-adv-part-past", "cvb-adv-part-absolute", TAG_CVB_SIMUL,
}

# Verb class labels
CLASS_TENSE_POS    = "tense-positive"
CLASS_TENSE_NEG    = "tense-negative"
CLASS_HABITUAL_NEG = "habitual-negative"
CLASS_OBLIGATIVE   = "obligative"
CLASS_PERMISSIVE   = "permissive"
CLASS_CONDITIONAL  = "conditional"
CLASS_EXISTENTIAL  = "existential"
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

        # Existential copula: ഉണ്ട് → <aff> tag (no <v>)
        self.is_existential = TAG_AFF in self.tags and not self.has_verb

        # Non-finite tag (participial) already present
        self.non_finite_tag = next(
            (t for t in self.suffix_tags if t in NON_FINITE_TAGS), None
        )

        # Mood classification (after the last <v>)
        self.is_obligative  = TAG_IMPERATIVE  in self.suffix_tags
        self.is_permissive  = TAG_PERMISSIVE  in self.suffix_tags or TAG_PROMISSIVE in self.suffix_tags
        self.is_conditional = TAG_CONDITIONAL in self.suffix_tags

        # Habitual-aspect + neg
        self.is_habitual_neg = TAG_HABITUAL_ASPECT in self.suffix_tags and self.has_neg

        # Lemma: text before first <
        self.lemma = raw.split("<")[0] if "<" in raw else raw

    def verb_class(self) -> str:
        if self.already_normalized:
            return CLASS_ALREADY_NORM
        if self.is_existential:
            return CLASS_EXISTENTIAL
        if not self.has_verb:
            return CLASS_NON_VERB
        if self.is_habitual_neg:
            return CLASS_HABITUAL_NEG
        if self.is_obligative:
            return CLASS_OBLIGATIVE
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
        if self.tense in (TAG_PRESENT, TAG_FUTURE):
            # Malayalam has no FST future-RP <n><deriv> path;
            # present-RP is the grammatically appropriate substitute.
            return RP_PRESENT
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


def _targets_habitual_negative(raw: str, va: VerbAnalysis) -> list:
    """
    Habitual-aspect negative: <habitual-aspect>ഇല്ല<neg>
    → <cont-perfect-aspect-neg><adv-clause-rp-past><n><deriv>

    This is the FST path that generates forms like
      വിളിക്കാതിരിക്കുന്നത് (continuous non-occurrence).
    The study equates this with the clefted form of habitual negatives.
    """
    last_v_pos = raw.rfind("<v>")
    prefix     = raw[:last_v_pos + len("<v>")]
    target     = prefix + f"<{TAG_CONT_PERFECT_NEG}><{RP_PAST}>" + TAG_NOMINAL
    return [target]


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


# No FST paths for permissive and conditional — they return [] to signal UNRESOLVED.

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# All RP tag names that count as valid in re-analysis
_ALL_RP_TAGS = [
    RP_PAST, RP_PRESENT, RP_FUTURE,
    RP_PAST_NEG, RP_PRESENT_NEG,
    TAG_CONT_PERFECT_NEG,
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

        # -- Existential (ഉണ്ട്) --
        if vc == CLASS_EXISTENTIAL:
            result["rp_tag"]         = ""
            result["failure_reason"] = (
                "Existential copula ഉണ്ട് has no regular FST nominalization path. "
                "The study identifies ഉള്ളത് as the irregular lexical substitute. "
                "mlmorph analyses ഉള്ളത് as ഉള്ളത്<n> (a lexical noun, not a derived form). "
                "Pattern class: <aff> copula → irregular."
            )
            return result

        # -- Non-verb --
        if vc == CLASS_NON_VERB:
            result["failure_reason"] = (
                f"Analysis '{best_raw}' contains no <v> tag; input may not be a verb."
            )
            return result

        # -- Permissive --
        if vc == CLASS_PERMISSIVE:
            result["rp_tag"]         = ""
            result["failure_reason"] = (
                f"Verb has permissive/promissive mood (<permissive-mood> or <promissive-mood>) "
                f"in analysis '{best_raw}'. "
                "mlmorph FST has no direct <n><deriv> path for permissive verbs. "
                "The study's target form (e.g. ചെയ്യാവുന്നത്) is not analysable by mlmorph; "
                "its generation would require a separate potential-modal auxiliary chain "
                "not represented in the current FST. "
                "Pattern class: <v><permissive-mood> → no <adv-clause-rp-*><n><deriv> path."
            )
            return result

        # -- Conditional --
        if vc == CLASS_CONDITIONAL:
            result["rp_tag"]         = ""
            result["failure_reason"] = (
                f"Verb has conditional mood (<conditional-mood>) in analysis '{best_raw}'. "
                "The study classifies conditional verbs as an exception with no available "
                "direct cleft-normalization form. "
                "mlmorph FST has no <adv-clause-rp-*><n><deriv> path for <conditional-mood>. "
                "Pattern class: <v><conditional-mood> → no nominalization path."
            )
            return result

        # -- Build targets for remaining classes --
        if vc == CLASS_HABITUAL_NEG:
            targets = _targets_habitual_negative(best_raw, va)
            rp_display = f"{TAG_CONT_PERFECT_NEG}>{RP_PAST}"
        elif vc == CLASS_OBLIGATIVE:
            targets = _targets_obligative(best_raw, va)
            rp_display = f"{TAG_CVB_SIMUL} + {OBLIGATIVE_AUX_BLOCK}"
        elif vc == CLASS_TENSE_NEG:
            targets = _targets_tense_negative(best_raw, va)
            rp_display = va.rp_tag_negative
        else:   # CLASS_TENSE_POS
            targets = _targets_tense_positive(best_raw, va)
            rp_display = va.rp_tag_positive or ""

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
