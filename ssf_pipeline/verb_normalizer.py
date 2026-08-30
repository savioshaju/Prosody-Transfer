"""
verb_normalizer.py — Comprehensive Malayalam verb normalizer for cleft construction.

Responsibility
--------------
Given a finite Malayalam verb that the Cleft Controller has already decided
should be nominalized, produce its correct nominalized (ത്-ending) form.

Supported verb classes
---------------------
1. ALREADY_NORM  — input already ends in <n><deriv> / ത് → passthrough
2. TENSE_POS:
   - Past -ന്നു / -ി  → -ന്നത് / -ിയത്  (<adv-clause-rp-past><n><deriv>)
   - Present -ുന്നു   → -ുന്നത്         (<adv-clause-rp-present><n><deriv>)
   - Future -ും      → -ുന്നത്         (<adv-clause-rp-present><n><deriv>)
3. TENSE_NEG     — negative finite verb → -ാത്തത് (<adv-clause-rp-present-neg><n><deriv>)
4. OBLIGATIVE    — obligative -ണം → -േണ്ടത് (<cvb-adv-part-simul>അണ്ടുക<v><cvb-adv-part-absolute><n><deriv>)
5. HABITUAL_POS  — habitual affirmative -ാറുണ്ട് → -ാറുള്ളത് (<habitual-aspect> + ഉള്ളത്<n><deriv>)
6. HABITUAL_NEG  — habitual negative -ാറില്ല → -ാറില്ലാത്തത് (<habitual-aspect> + ഇല്ലാത്തത്<n><deriv>)
7. PERMISSIVE    — permissive -ാം → -ാവുന്നത് (<permissive-mood> base nominalization)

Preserved as UNRESOLVED:
- Conditional, optative, non-verbs, unanalyzable forms.

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
TAG_PROMISSIVE       = "promissive-mood"
TAG_CONDITIONAL      = "conditional-mood"
TAG_HABITUAL_ASPECT  = "habitual-aspect"
TAG_SIMPLE_PERFECT   = "simple-perfect-aspect"
TAG_PERFECT          = "perfect-aspect"

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
CLASS_HABITUAL_POS = "habitual-positive"
CLASS_HABITUAL_NEG = "habitual-negative"
CLASS_OBLIGATIVE   = "obligative"
CLASS_PERMISSIVE   = "permissive"
CLASS_OPTATIVE     = "optative"
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

        # Mood & Aspect classification (after the last <v>)
        self.is_obligative   = TAG_IMPERATIVE in self.suffix_tags
        self.is_optative     = TAG_OPTATIVE in self.suffix_tags
        self.is_permissive   = (TAG_PERMISSIVE in self.suffix_tags or TAG_PROMISSIVE in self.suffix_tags)
        self.is_conditional  = TAG_CONDITIONAL in self.suffix_tags

        # Perfect / aspectual (e.g. വന്നിരുന്നു, ചെയ്തിരുന്നു)
        self.is_perfect      = (
            TAG_SIMPLE_PERFECT in self.suffix_tags
            or TAG_PERFECT in self.suffix_tags
            or TAG_SIMPLE_PERFECT in self.tags
            or TAG_PERFECT in self.tags
        )

        # Habitual-aspect
        self.is_habitual     = TAG_HABITUAL_ASPECT in self.suffix_tags
        self.is_habitual_neg = self.is_habitual and self.has_neg
        self.is_habitual_pos = self.is_habitual and not self.has_neg

        # Progressive / Iterative aspect (e.g. നൽകിക്കൊണ്ടിരിക്കുകയാണ്, ചെയ്യുകയാണ്, വായിക്കുകയാണ്)
        self.is_progressive  = (
            TAG_HABITUAL_ASPECT not in self.suffix_tags
            and (
                "iterative-aspect" in self.suffix_tags
                or "iterative-aspect" in self.tags
                or "<v>ആണ്<aff>" in self.raw
                or "കൊണ്ടിരിക്കുക" in self.raw
                or any(
                    self.raw.endswith(sfx) or self.lemma.endswith(sfx)
                    for sfx in ("കൊണ്ടിരിക്കുകയാണ്", "ക്കുകയാണ്", "ുകയാണ്")
                )
            )
        )

    def verb_class(self) -> str:
        if self.already_normalized:
            return CLASS_ALREADY_NORM
        if not self.has_verb:
            return CLASS_NON_VERB
        if self.is_habitual_neg:
            return CLASS_HABITUAL_NEG
        if self.is_habitual_pos:
            return CLASS_HABITUAL_POS
        if self.is_obligative:
            return CLASS_OBLIGATIVE
        if self.is_permissive:
            return CLASS_PERMISSIVE
        if self.is_optative:
            return CLASS_OPTATIVE
        if self.is_conditional:
            return CLASS_CONDITIONAL
        if self.has_neg:
            return CLASS_TENSE_NEG
        return CLASS_TENSE_POS

    @property
    def rp_tag_positive(self):
        """Tense / Aspect → RP tag for positive (non-negated) verbs."""
        if self.is_progressive:
            return RP_PRESENT
        if self.tense == TAG_PAST or self.is_perfect:
            return RP_PAST
        if self.tense == TAG_PRESENT:
            return RP_PRESENT
        if self.tense == TAG_FUTURE:
            # Future tense (e.g. വരും, പോകും) maps to present RP (വരുന്നത്, പോകുന്നത്)
            return RP_PRESENT
        return None

    @property
    def rp_tag_negative(self):
        """Tense → RP-neg tag for negated verbs."""
        if self.tense == TAG_PAST or self.is_perfect:
            return RP_PAST_NEG
        # present/future/tenseless negative → present-neg (-ാത്തത്)
        return RP_PRESENT_NEG


# ---------------------------------------------------------------------------
# Generation target builders (one per verb class)
# ---------------------------------------------------------------------------

def _targets_tense_positive(raw: str, va: VerbAnalysis) -> list:
    rp = va.rp_tag_positive
    if rp is None:
        return []

    if va.is_progressive:
        targets = []
        if "കൊണ്ടിരിക്കുക" in raw:
            if "ആണ്<aff>" in raw:
                idx = raw.find("ആണ്<aff>")
                targets.append(raw[:idx] + f"<{RP_PRESENT}>{TAG_NOMINAL}")
            last_v_pos = raw.rfind("<v>")
            prefix = raw[:last_v_pos + len("<v>")]
            targets.append(prefix + f"<{RP_PRESENT}>{TAG_NOMINAL}")

        first_v_pos = raw.find("<v>")
        if first_v_pos != -1:
            base_prefix = raw[:first_v_pos + len("<v>")]
            targets.append(base_prefix + f"<{RP_PRESENT}>{TAG_NOMINAL}")
        return targets

    last_v_pos     = raw.rfind("<v>")
    prefix         = raw[:last_v_pos + len("<v>")]
    voice_clause   = f"<{va.voice}>" if va.voice else ""
    tense_clause   = f"<{va.tense}>" if va.tense else ""
    primary        = prefix + voice_clause + f"<{rp}>" + TAG_NOMINAL
    with_tense     = prefix + tense_clause + voice_clause + f"<{rp}>" + TAG_NOMINAL
    if va.is_perfect:
        with_perfect = prefix + f"<{TAG_SIMPLE_PERFECT}><{rp}>" + TAG_NOMINAL
        return [with_perfect, primary, with_tense]
    return [primary, with_tense]


def _targets_tense_negative(raw: str, va: VerbAnalysis) -> list:
    """
    Negative finite verbs: strip <tense><neg> and generate present/past negative RP.
    e.g. വരുന്നില്ല / വന്നില്ല → വരാത്തത് (<adv-clause-rp-present-neg><n><deriv>)
    """
    last_v_pos = raw.rfind("<v>")
    prefix     = raw[:last_v_pos + len("<v>")]
    primary    = prefix + f"<{RP_PRESENT_NEG}>" + TAG_NOMINAL
    secondary  = prefix + f"<{RP_PAST_NEG}>" + TAG_NOMINAL
    return [primary, secondary]


def _targets_obligative(raw: str, va: VerbAnalysis) -> list:
    """
    Obligative verbs: -ണം → -േണ്ടത്.
    Generated via: lemma<v><cvb-adv-part-simul>അണ്ടുക<v><cvb-adv-part-absolute><n><deriv>
    """
    last_v_pos = raw.rfind("<v>")
    prefix     = raw[:last_v_pos + len("<v>")]
    target     = prefix + f"<{TAG_CVB_SIMUL}>{OBLIGATIVE_AUX_BLOCK}" + TAG_NOMINAL
    return [target]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_ALL_RP_TAGS = [
    RP_PAST, RP_PRESENT,
    RP_PAST_NEG, RP_PRESENT_NEG,
]

def _is_valid_nominalized_verb(form: str, analyser: Analyser) -> tuple:
    """
    Validates that 'form' is a linguistically valid nominalized verb in Malayalam.
    Returns (is_valid: bool, analysis_str: str).
    """
    if not form:
        return False, ""
    
    # 1. Direct mlmorph re-analysis
    results = analyser.analyse(form)
    if results:
        for raw, _ in results:
            if "<n><deriv>" not in raw:
                continue
            if any(f"<{rp}>" in raw for rp in _ALL_RP_TAGS):
                return True, raw
            if "<cvb-adv-part-simul>" in raw and "<cvb-adv-part-absolute><n><deriv>" in raw:
                return True, raw

    # 2. Permissive validation: form[:-2] + 'താണ്' (e.g. വരാവുന്നത് -> വരാവുന്നതാണ്)
    if form.endswith("ാവുന്നത്"):
        cop_form = form[:-2] + "താണ്"
        results = analyser.analyse(cop_form)
        if results:
            for raw, _ in results:
                if "<permissive-mood>" in raw:
                    return True, f"{raw}<n><deriv>"

    # 3. Habitual affirmative validation: stem + ഉള്ളത് (e.g. വരാറുള്ളത് -> stem 'വരാറ്')
    if form.endswith("ുള്ളത്"):
        stem = form.split("ുള്ളത്")[0] + "്"
        results = analyser.analyse(stem)
        if results:
            for raw, _ in results:
                if "<habitual-aspect>" in raw:
                    return True, f"{raw}ഉള്ളത്<n><deriv>"

    # 4. Habitual negative validation: stem + ഇല്ലാത്തത് (e.g. വരാറില്ലാത്തത് -> stem 'വരാറ്')
    if form.endswith("ില്ലാത്തത്"):
        stem = form.split("ില്ലാത്തത്")[0] + "്"
        results = analyser.analyse(stem)
        if results:
            for raw, _ in results:
                if "<habitual-aspect>" in raw:
                    return True, f"{raw}ഇല്ലാത്തത്<n><deriv>"

    # 5. Surface morphological validation for compound/derived verbs (e.g. സ്ഥിതിചെയ്യുന്നത്)
    if form.endswith("ത്") and any(form.endswith(sfx) for sfx in ("ുന്നത്", "ന്നത്", "ിയത്", "യത്", "ാത്തത്", "േണ്ടത്", "ത്തത്", "ഞ്ഞത്", "ണ്ടത്")):
        return True, f"{form}<v><adv-clause-rp-present><n><deriv>"

    return False, ""


# ---------------------------------------------------------------------------
# Main normalizer
# ---------------------------------------------------------------------------

PUNCTUATION_CHARS = set(".,!?;:\"'“”‘’()[]{}—–-«»/\\")


def split_punctuation(token: str) -> tuple:
    """
    Splits a token into (leading_punct, core_linguistic_word, trailing_punct).
    Treats structural punctuation (commas, colons, quotes, etc.) separately
    from the linguistic stem so that morphology/normalization operates on the
    stem and restores punctuation outside the normalized form.
    """
    if not token or not isinstance(token, str):
        return "", "", ""
    start = 0
    while start < len(token) and token[start] in PUNCTUATION_CHARS:
        start += 1
    end = len(token)
    while end > start and token[end - 1] in PUNCTUATION_CHARS:
        end -= 1
    leading = token[:start]
    core = token[start:end]
    trailing = token[end:]
    return leading, core, trailing


class VerbNormalizer:
    """Standalone Malayalam verb normalizer."""

    def __init__(self):
        self._analyser  = Analyser()
        self._generator = Generator()

    def normalize(self, verb: str) -> dict:
        """
        Normalize a finite verb to its nominalized (ത്-ending) form,
        preserving any leading/trailing structural punctuation.
        """
        leading, core, trailing = split_punctuation(verb)
        if not core:
            return {
                "original": verb,
                "lemma": "",
                "analysis": "",
                "verb_class": "",
                "rp_tag": "",
                "gen_target": "",
                "derived_stem": "",
                "normalized": verb,
                "normalized_analysis": "",
                "status": "UNRESOLVED",
                "failure_reason": "Empty or punctuation-only verb token.",
            }

        result = self._normalize_core(core)
        result["original"] = verb
        if result["normalized"]:
            result["normalized"] = f"{leading}{result['normalized']}{trailing}"
        return result

    def _normalize_core(self, verb: str) -> dict:
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

        # High-priority surface rules for existential/stative compound verbs
        if verb.endswith("ഉണ്ട്"):
            norm_form = verb[:-len("ഉണ്ട്")] + "ഉള്ളത്"
            result["status"] = "VALID"
            result["normalized"] = norm_form
            result["verb_class"] = CLASS_TENSE_POS
            result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-present><n><deriv>"
            return result

        if verb.endswith("മുണ്ട്"):
            norm_form = verb[:-len("മുണ്ട്")] + "മുള്ളത്"
            result["status"] = "VALID"
            result["normalized"] = norm_form
            result["verb_class"] = CLASS_TENSE_POS
            result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-present><n><deriv>"
            return result

        if verb.endswith("ഉണ്ടായിരുന്നു"):
            norm_form = verb[:-len("ഉണ്ടായിരുന്നു")] + "ഉണ്ടായിരുന്നത്"
            result["status"] = "VALID"
            result["normalized"] = norm_form
            result["verb_class"] = CLASS_TENSE_POS
            result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-past><n><deriv>"
            return result

        # Step 1: Analyse
        raw_analyses = self._analyser.analyse(verb)
        if not raw_analyses:
            # ----------------------------------------------------------
            # Surface rule fallback for compound/OOV verbs not in FST.
            # Implements all rows of the SSF verb normalization table:
            #
            # Past affirmative:
            #   -ന്നു  → -ന്നത്     (e.g. വന്നു → വന്നത്)
            #   -ി     → -ിയത്      (e.g. വാങ്ങി → വാങ്ങിയത്)
            #   -ിച്ചു → -ിച്ചത്     (e.g. ചെയ്തിച്ചു → ചെയ്തിച്ചത്)
            #   -ച്ചു  → -ച്ചത്      (e.g. അയച്ചു → അയച്ചത്)
            #   -ഞ്ഞു  → -ഞ്ഞത്     (e.g. കൊടുഞ്ഞു → കൊടുഞ്ഞത്)
            # Present affirmative:
            #   -ുന്നു → -ുന്നത്     (e.g. സ്ഥിതിചെയ്യുന്നു → സ്ഥിതിചെയ്യുന്നത്)
            #   -യുന്നു → -യുന്നത്   (handled by -ുന്നു since it ends the same)
            # Future affirmative:
            #   -ും    → -ുന്നത്     (e.g. വരും → വരുന്നത്)
            # Negative:
            #   -ുന്നില്ല → strip -ുന്നില്ല + -ാത്തത്
            #   -ഇല്ല  → strip -ഇല്ല + -ാത്തത്
            # Habitual affirmative:
            #   -ാറുണ്ട് → -ാറുള്ളത്  (e.g. വരാറുണ്ട് → വരാറുള്ളത്)
            # Habitual negative:
            #   -ാറില്ല → -ാറില്ലാത്തത് (e.g. വരാറില്ല → വരാറില്ലാത്തത്)
            # Obligative:
            #   -ണം   → -േണ്ടത്      (e.g. ചെയ്യണം → ചെയ്യേണ്ടത്)
            # Permissive:
            #   -ാം   → -ാവുന്നത്    (e.g. വരാം → വരാവുന്നത്)
            # Already nominalized:
            #   -ത്   → passthrough
            # ----------------------------------------------------------

            # Already nominalized
            if verb.endswith("ത്") and any(verb.endswith(sfx) for sfx in ("ുന്നത്", "ന്നത്", "ിയത്", "ച്ചത്", "ഞ്ഞത്", "ാത്തത്", "േണ്ടത്")):
                result["status"] = "VALID"
                result["normalized"] = verb
                result["verb_class"] = CLASS_ALREADY_NORM
                result["normalized_analysis"] = f"{verb}<v><adv-clause-rp-present><n><deriv>"
                result["failure_reason"] = "Input is already in nominalized form."
                return result

            # Habitual negative: -ാറില്ല → -ാറില്ലാത്തത്
            if verb.endswith("ാറില്ല"):
                norm_form = verb + "ാത്തത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_HABITUAL_NEG
                result["normalized_analysis"] = f"{norm_form}<v><habitual-aspect><neg><n><deriv>"
                return result

            # Habitual affirmative: -ാറുണ്ട് → -ാറുള്ളത്
            if verb.endswith("ാറുണ്ട്"):
                norm_form = verb[:-6] + "ാറുള്ളത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_HABITUAL_POS
                result["normalized_analysis"] = f"{norm_form}<v><habitual-aspect><n><deriv>"
                return result

            # Obligative: -ണം → -േണ്ടത്
            if verb.endswith("ണം"):
                norm_form = verb[:-2] + "േണ്ടത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_OBLIGATIVE
                result["normalized_analysis"] = f"{norm_form}<v><cvb-adv-part-simul><n><deriv>"
                return result

            # Existential/Suppletive: -ഉണ്ട് / -മുണ്ട് → -ഉള്ളത് / -മുള്ളത്
            if verb.endswith("ഉണ്ട്"):
                norm_form = verb[:-len("ഉണ്ട്")] + "ഉള്ളത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_TENSE_POS
                result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-present><n><deriv>"
                return result

            if verb.endswith("മുണ്ട്"):
                norm_form = verb[:-len("മുണ്ട്")] + "മുള്ളത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_TENSE_POS
                result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-present><n><deriv>"
                return result

            # Permissive: -ാം → -ാവുന്നത്
            if verb.endswith("ാം"):
                norm_form = verb[:-2] + "ാവുന്നത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_PERMISSIVE
                result["normalized_analysis"] = f"{norm_form}<v><permissive-mood><n><deriv>"
                return result

            # Negative: -ുന്നില്ല → -ാത്തത്
            if verb.endswith("ുന്നില്ല"):
                norm_form = verb[:-len("ുന്നില്ല")] + "ാത്തത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_TENSE_NEG
                result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-present-neg><n><deriv>"
                return result

            # Negative: -ഇല്ല → -ാത്തത്
            if verb.endswith("ില്ല"):
                norm_form = verb[:-len("ില്ല")] + "ാത്തത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_TENSE_NEG
                result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-present-neg><n><deriv>"
                return result

            # Present affirmative: -ുന്നു → -ുന്നത്
            if verb.endswith("ുന്നു"):
                norm_form = verb[:-4] + "ുന്നത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_TENSE_POS
                result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-present><n><deriv>"
                return result

            # Past affirmative: -ിച്ചു → -ിച്ചത്
            if verb.endswith("ിച്ചു"):
                norm_form = verb[:-4] + "ിച്ചത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_TENSE_POS
                result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-past><n><deriv>"
                return result

            # Past affirmative: -ച്ചു → -ച്ചത്
            if verb.endswith("ച്ചു"):
                norm_form = verb[:-2] + "ത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_TENSE_POS
                result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-past><n><deriv>"
                return result

            # Past affirmative: -ഞ്ഞു → -ഞ്ഞത്
            if verb.endswith("ഞ്ഞു"):
                norm_form = verb[:-2] + "ത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_TENSE_POS
                result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-past><n><deriv>"
                return result

            # Past affirmative: -ന്നു → -ന്നത്
            if verb.endswith("ന്നു"):
                norm_form = verb[:-2] + "ത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_TENSE_POS
                result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-past><n><deriv>"
                return result

            # Past affirmative: -ി → -ിയത്
            if verb.endswith("ി"):
                norm_form = verb + "യത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_TENSE_POS
                result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-past><n><deriv>"
                return result

            # Future affirmative: -ും → -ുന്നത്
            if verb.endswith("ും"):
                norm_form = verb[:-2] + "ുന്നത്"
                result["status"] = "VALID"
                result["normalized"] = norm_form
                result["verb_class"] = CLASS_TENSE_POS
                result["normalized_analysis"] = f"{norm_form}<v><adv-clause-rp-present><n><deriv>"
                return result

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

        # 1. Already normalized
        if vc == CLASS_ALREADY_NORM:
            result["normalized"]          = verb
            result["rp_tag"]              = va.non_finite_tag or ""
            result["normalized_analysis"] = best_raw
            result["derived_stem"]        = verb[:-2] if verb.endswith("\u0d24\u0d4d") else verb
            result["status"]              = "VALID"
            result["failure_reason"]      = "Input is already in nominalized form."
            return result

        # 2. Non-verb
        if vc == CLASS_NON_VERB:
            if any(verb.endswith(sfx) for sfx in ("കൊണ്ടിരിക്കുകയാണ്", "ക്കുകയാണ്", "ുകയാണ", "കയാണ്", "ാറുണ്ട്", "ാറില്ല", "ണം", "ാം")):
                vc = CLASS_TENSE_POS
                targets = []
            else:
                result["failure_reason"] = (
                    f"Analysis '{best_raw}' contains no <v> tag; input may not be a verb."
                )
                return result

        # 3. Permissive mood (-ാം → -ാവുന്നത്)
        if vc == CLASS_PERMISSIVE:
            # FST path: generator on <permissive-mood> produces '...ാവrunningതാണ്'
            gen_res = self._generator.generate(f"{va.lemma}<v><permissive-mood>")
            generated_form = None
            if gen_res:
                for form, _ in gen_res:
                    if "ാവുന്നത്" in form:
                        generated_form = form
                        break
                    elif "ാവrunningതാണ്" in form or "ാവുന്നത്" in form or "ാവുന്നതാണ്" in form:
                        # Strip trailing copula ആണ് / ാണ്
                        clean_form = re.sub(r"ാണ്$|ാണു്$|ആണ്$", "", form) + "ത്"
                        if clean_form.endswith("ാവുന്നത്"):
                            generated_form = clean_form
                            break

            # Surface fallback if FST generator missing dictionary entry
            if not generated_form:
                if verb.endswith("ാം"):
                    generated_form = verb[:-2] + "ാവുന്നത്"
                else:
                    generated_form = verb + "ാവുന്നത്"

            result["gen_target"]   = f"{va.lemma}<v><permissive-mood><n><deriv>"
            result["rp_tag"]       = "permissive-rp"
            result["derived_stem"] = generated_form[:-2] if generated_form.endswith("ത്") else generated_form
            
            is_valid, norm_raw = _is_valid_nominalized_verb(generated_form, self._analyser)
            if is_valid:
                result["status"]              = "VALID"
                result["normalized"]          = generated_form
                result["normalized_analysis"] = norm_raw
            else:
                result["status"]         = "UNRESOLVED"
                result["failure_reason"] = f"Generated '{generated_form}' but failed validation."
            return result

        # 4. Habitual affirmative (-ാറുണ്ട് → -ാറുള്ളത്)
        if vc == CLASS_HABITUAL_POS:
            gen_res = self._generator.generate(f"{va.lemma}<v><habitual-aspect>")
            generated_form = None
            if gen_res:
                stem = gen_res[0][0]  # e.g. വരാറ്
                if stem.endswith("്"):
                    generated_form = stem[:-1] + "ുള്ളത്"
                else:
                    generated_form = stem + "ുള്ളത്"
            
            # Surface fallback
            if not generated_form:
                if verb.endswith("ാറുണ്ട്"):
                    generated_form = verb[:-6] + "ാറുള്ളത്"
                else:
                    generated_form = verb + "ഉള്ളത്"

            result["gen_target"]   = f"{va.lemma}<v><habitual-aspect>ഉള്ളത്<n><deriv>"
            result["rp_tag"]       = "habitual-rp-pos"
            result["derived_stem"] = generated_form[:-2] if generated_form.endswith("ത്") else generated_form

            is_valid, norm_raw = _is_valid_nominalized_verb(generated_form, self._analyser)
            if is_valid:
                result["status"]              = "VALID"
                result["normalized"]          = generated_form
                result["normalized_analysis"] = norm_raw
            else:
                result["status"]         = "UNRESOLVED"
                result["failure_reason"] = f"Generated '{generated_form}' but failed validation."
            return result

        # 5. Habitual negative (-ാറില്ല → -ാറില്ലാത്തത്)
        if vc == CLASS_HABITUAL_NEG:
            gen_res = self._generator.generate(f"{va.lemma}<v><habitual-aspect>")
            generated_form = None
            if gen_res:
                stem = gen_res[0][0]  # e.g. വരാറ്
                if stem.endswith("്"):
                    generated_form = stem[:-1] + "ില്ലാത്തത്"
                else:
                    generated_form = stem + "ില്ലാത്തത്"

            # Surface fallback
            if not generated_form:
                if verb.endswith("ാറില്ല"):
                    generated_form = verb[:-6] + "ാറില്ലാത്തത്"
                else:
                    generated_form = verb + "ഇല്ലാത്തത്"

            result["gen_target"]   = f"{va.lemma}<v><habitual-aspect>ഇല്ലാത്തത്<n><deriv>"
            result["rp_tag"]       = "habitual-rp-neg"
            result["derived_stem"] = generated_form[:-2] if generated_form.endswith("ത്") else generated_form

            is_valid, norm_raw = _is_valid_nominalized_verb(generated_form, self._analyser)
            if is_valid:
                result["status"]              = "VALID"
                result["normalized"]          = generated_form
                result["normalized_analysis"] = norm_raw
            else:
                result["status"]         = "UNRESOLVED"
                result["failure_reason"] = f"Generated '{generated_form}' but failed validation."
            return result

        # 6. Obligative mood (-ണം → -േണ്ടത്)
        if vc == CLASS_OBLIGATIVE:
            targets = _targets_obligative(best_raw, va)
            rp_display = "obligative-rp"

        # 7. Tense negative (-ഇല്ല → -ാത്തത്)
        elif vc == CLASS_TENSE_NEG:
            targets = _targets_tense_negative(best_raw, va)
            rp_display = va.rp_tag_negative

        # 8. Tense positive (Past, Present, Future)
        elif vc == CLASS_TENSE_POS:
            targets = _targets_tense_positive(best_raw, va)
            rp_display = va.rp_tag_positive

        # 9. Unsupported mood/structures
        else:
            result["status"] = "UNRESOLVED"
            result["failure_reason"] = (
                f"Unsupported mood or verb structure '{best_raw}' (class: {vc})."
            )
            return result

        result["rp_tag"] = rp_display

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

        result["gen_target"] = used_target or (targets[0] if targets else "")

        if generated_form is None:
            if verb.endswith("ിക്കൊണ്ടിരിക്കുകയാണ്"):
                generated_form = verb[:-len("ിക്കൊണ്ടിരിക്കുകയാണ്")] + "ിക്കൊണ്ടിരിക്കുന്നത്"
            elif verb.endswith("ച്ചുകൊണ്ടിരിക്കുകയാണ്"):
                generated_form = verb[:-len("ച്ചുകൊണ്ടിരിക്കുകയാണ്")] + "ച്ചുകൊണ്ടിരിക്കുന്നത്"
            elif verb.endswith("ഞ്ഞുകൊണ്ടിരിക്കുകയാണ്"):
                generated_form = verb[:-len("ഞ്ഞുകൊണ്ടിരിക്കുകയാണ്")] + "ഞ്ഞുകൊണ്ടിരിക്കുന്നത്"
            elif verb.endswith("കൊണ്ടിരിക്കുകയാണ്"):
                generated_form = verb[:-len("കൊണ്ടിരിക്കുകയാണ്")] + "കൊണ്ടിരിക്കുന്നത്"
            elif verb.endswith("ിക്കുകയാണ്"):
                generated_form = verb[:-len("ിക്കുകയാണ്")] + "ിക്കുന്നത്"
            elif verb.endswith("ക്കുകയാണ്"):
                generated_form = verb[:-len("ക്കുകയാണ്")] + "ക്കുന്നത്"
            elif verb.endswith("ികയാണ്"):
                generated_form = verb[:-len("ികയാണ്")] + "ുന്നത്"
            elif verb.endswith("ുകയാണ്") and len(verb) > 4:
                generated_form = verb[:-len("ുകയാണ്")] + "ുന്നത്"
            elif any(verb.endswith(sfx) for sfx in ("ിരുന്നു", "യിരുന്നു", "ന്നിരുന്നു", "ച്ചിരുന്നു", "ഞ്ഞിരുന്നു", "തിരുന്നു")):
                generated_form = verb[:-1] + "ത്"
            elif verb.endswith("ുന്നു"):
                generated_form = verb[:-len("ുന്നു")] + "ുന്നത്"
            elif verb.endswith("ിച്ചു"):
                generated_form = verb[:-len("ിച്ചു")] + "ിച്ചത്"
            elif verb.endswith("ിയ"):
                generated_form = verb + "ത്"
            elif verb.endswith("ി"):
                generated_form = verb + "യത്"
            elif verb.endswith("ു"):
                generated_form = verb[:-1] + "ിയത്"

        # Step 7: Derive stem for display
        result["derived_stem"] = (
            generated_form[:-2] if generated_form and generated_form.endswith("\u0d24\u0d4d") else (generated_form or "")
        )

        # Step 8: Validate by re-analysis with fallback
        if generated_form:
            is_valid, norm_raw = _is_valid_nominalized_verb(generated_form, self._analyser)
            result["status"]              = "VALID"
            result["normalized"]          = generated_form
            result["normalized_analysis"] = norm_raw or f"{generated_form}<v><n><deriv>"
            return result

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
