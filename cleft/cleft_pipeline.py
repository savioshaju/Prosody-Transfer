import re
import logging
from typing import Optional, Tuple, List, Dict, Any

from .malayalam_pipeline import MalayalamPipeline
from .copula_pipeline import AnalysisLayer, CopulaLayer
from verb_norm.verb_normalizer import VerbNormalizer
from aligner.alignment_utils import extract_alignment_and_prosody, strip_punctuation

logger = logging.getLogger(__name__)

FF_TAG = "<FF>"
# Format: <FF>word<FF> or <FF>word</FF>
FF_PATTERN = re.compile(r"<FF>(.*?)(?:</FF>|<FF>)", re.UNICODE)

# ---------------------------------------------------------------------------
# Lexical & category sets for Phase 1 Eligibility checking
# ---------------------------------------------------------------------------

INDEFINITE_PRONOUNS = {
    "ആരെങ്കിലും", "എന്തെങ്കിലും", "എവിടെയെങ്കിലും", "എപ്പോഴെങ്കിലും",
    "ആരും", "ഒന്നും", "എവിടെയും", "എപ്പോഴും",
    "ഏതെങ്കിലും", "ചിലർ", "ആരേലും", "എന്തേലും", "ഏതേലും",
    "ആരെയും", "എന്തിനെയും", "ആർക്കും", "ഒന്നിനും", "ആർക്കെങ്കിലും"
}

DEFINITE_PRONOUN_LEMMAS = {
    "ഞാൻ", "നീ", "അവൻ", "അവൾ", "അവർ", "അത്",
    "ഞങ്ങൾ", "നിങ്ങൾ", "നാം", "ഇവൻ", "ഇവൾ", "ഇവർ", "ഇത്",
    "താങ്കൾ", "അങ്ങ്"
}

TIME_WORDS = {
    "ഇന്നലെ", "ഇന്ന്", "നാളെ", "രാവിലെ", "വൈകുന്നേരം", "ഇപ്പോൾ",
    "അന്ന്", "പിറ്റേന്ന്", "നേരത്തെ", "എപ്പോൾ", "പണ്ട്", "ഇപ്പോഴേ",
    "രാത്രി", "പകൽ", "ഉച്ചയ്ക്ക്", "അസ്തമയത്ത്"
}

TEMPORAL_CLAUSE_SUFFIXES = (
    "തിനുശേഷം",
    "തിനുമുമ്പ്",
    "പ്പോൾ",
    "ുമ്പോൾ",
    "മ്പോൾ",
)

MANNER_SUFFIXES = (
    # Standard manner/instrumental postpositions
    "ആയി", "ആയിട്ട്", "ഓടെ", "ആംവണ്ണം", "പ്രകാരം",
    # Instrumental case suffix
    "കൊണ്ട്", "ആലൂടെ", "ിലൂടെ", "ലൂടെ",
    # Conjunctive participle (CVB) endings — CVBs are manner/sequential adjuncts in Malayalam
    "ഞ്ഞ്", "ഞ്ഞ", "ിച്ച്", "ച്ച്", "ത്ത്", "്ത്", "ട്ട്",
    # Sociative (coordination-with focus, e.g. 'in coordination with India')
    "ഉമായി", "ോടൊപ്പം",
)

ADJECTIVE_PREDICATES = {
    "സുന്ദരം", "മനോഹരം", "നല്ല", "വലിയ", "ചെറിയ", "മിടുക്കൻ", "മിടുക്കി",
    "മോശം", "കേമം", "ധീരം", "സത്യം", "ശുദ്ധം", "വ്യക്തം", "ശരി", "തെറ്റ്"
}


# ---------------------------------------------------------------------------
# SECTION 1: CLEFT vs PE ELIGIBILITY DECISION FUNCTION
# ---------------------------------------------------------------------------

def _has_nominal_head_in_qt(focused_word: str) -> bool:
    """
    Check if a quantifier/ordinal expression is part of a compound noun or NP
    containing a nominal head (e.g. രണ്ടാംഭാഗം -> ഭാഗം, ഒന്നാംഘട്ടം -> ഘട്ടം).
    """
    clean_words = [strip_punctuation(w) for w in (focused_word or "").split() if strip_punctuation(w)]
    if not clean_words:
        return False

    noun_head_suffixes = (
        "ഭാഗം", "ഭാഗങ്ങൾ", "ഘട്ടം", "ഘട്ടങ്ങൾ", "അധ്യായം", "അധ്യായങ്ങൾ",
        "സ്ഥാനം", "വർഷം", "തീയതി", "ദിവസം", "മാസം", "വകുപ്പ്", "വകുപ്പുകൾ",
        "പേർ", "ആൾ", "ആളുകൾ", "എണ്ണം", "പങ്ക്", "ഇനം", "വരി", "പേജ്",
        "ഘടകം", "ഘടകങ്ങൾ", "വിഭാഗം", "വിഭാഗങ്ങൾ", "തലമുറ", "ക്ലാസ്",
        "ശതമാനം", "ശതമാനത്തിൽ", "ഹെക്ടർ", "ഹെക്ടറിൽ", "മാത്രം", "മാത്രമേ",
        "ൽ", "യിൽ", "ത്തിൽ", "ത്ത്", "നിന്ന്", "കൊണ്ട്", "ക്ക്", "ന്", "ന്റെ", "ുടെ"
    )

    # 1. Any word ends with a nominal head suffix (e.g. രണ്ടാംഭാഗം, ഒന്നാംഘട്ടം)
    for w in clean_words:
        if any(w.endswith(sfx) for sfx in noun_head_suffixes):
            return True

    # 2. Multi-word phrase with non-digit lexical words
    if len(clean_words) > 1 and any(
        not (w.isdigit() or w in ("ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്", "എട്ട്", "ഒമ്പത്", "പത്ത്"))
        for w in clean_words
    ):
        return True

    return False


def _extract_head_and_appositives(focus_phrase: str) -> tuple:
    """
    Split a focus phrase into (prefix, head_word, appositive_suffix).

    Handles:
      - Trailing parentheticals/brackets: "രണ്ടാംഭാഗം (വകുപ്പുകൾ 17-24)"
        -> prefix="", head_word="രണ്ടാംഭാഗം", appositive="(വകുപ്പുകൾ 17-24)"
      - Complex NP with parenthetical: "നാല് ചതുരങ്ങളുള്ള കെട്ടിടം (ചുവരുകൾ ഉള്ള)"
        -> prefix="നാല് ചതുരങ്ങളുള്ള", head_word="കെട്ടിടം", appositive="(ചുവരുകൾ ഉള്ള)"
      - Standard multi-word NP: "നാല് ചതുരങ്ങളുള്ള കെട്ടിടം"
        -> prefix="നാല് ചതുരങ്ങളുള്ള", head_word="കെട്ടിടം", appositive=""
      - Single word: "അച്ഛൻ"
        -> prefix="", head_word="അച്ഛൻ", appositive=""
    """
    cleaned = (focus_phrase or "").strip()
    if not cleaned:
        return "", "", ""

    # 1. Check for trailing parenthetical / bracketed appositive
    m = re.match(r"^(.*?)\s*(\([^\)]*\)|\[[^\]]*\])\s*$", cleaned)
    if m and m.group(1).strip():
        core_part = m.group(1).strip()
        appositive = m.group(2).strip()
        core_words = core_part.split()
        if len(core_words) > 1:
            return " ".join(core_words[:-1]), core_words[-1], appositive
        elif len(core_words) == 1:
            return "", core_words[0], appositive
        else:
            return "", "", appositive

    # 2. Standard multi-word constituent
    words = cleaned.split()
    if len(words) > 1:
        return " ".join(words[:-1]), words[-1], ""
    elif len(words) == 1:
        return "", words[0], ""
    return "", "", ""


# ---------------------------------------------------------------------------
# POSTPOSITIONS, DEGREE MODIFIERS & NUMERAL SETS
# ---------------------------------------------------------------------------

POSTPOSITIONS = frozenset({
    "കൂടെ", "ഒപ്പം", "ശേഷം", "മുമ്പ്", "മുൻപ്", "കുറിച്ച്", "സംബന്ധിച്ച്",
    "കൊണ്ട്", "വഴി", "ഉൾപ്പെടെ", "പകരം", "പകരമായി", "നേതൃത്വത്തിൽ",
    "പ്രകാരം", "കാരണം", "അനുസരിച്ച്", "പോലെ", "വരെ", "എതിരെ",
    "തുടങ്ങി", "ആയി", "ആയിട്ട്", "കുറിച്ചുള്ള", "നേരെ", "മാത്രം", "മാത്രമേ"
})

DEGREE_MODIFIERS = frozenset({
    "വളരെ", "കൂടുതൽ", "തീരെ", "അല്പം", "ഒരല്പം", "ഏറ്റവും", "അത്യധികം",
    "ഏറെ", "കുറച്ച്", "വല്ലാതെ", "അധികം", "തീർത്തും"
})

NUMERAL_WORDS = frozenset({
    "ഒരു", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്", "എട്ട്", "ഒമ്പത്", "പത്ത്",
    "നൂറ്", "ആയിരം", "ചില", "പല", "അനേകം", "ധാരാളം", "ഏതാനും", "എല്ലാ", "മുഴുവൻ"
})

STANDALONE_FREQUENCY_ADVERBS = frozenset({
    "കുറച്ച്", "അല്പം", "ഏറെ", "കൂടുതൽ", "ചിലത്", "ഒരല്പം", "ഒരുപാട്", "ധാരാളം"
})


def find_focus_span_indices(tokens, focus_word: str) -> Optional[Tuple[int, int]]:
    """
    Find the start and end token indices of the focus expression in a sequence of tokens.
    tokens can be a list of IR Token objects or string words.
    """
    if not tokens or not focus_word:
        return None

    clean_forms = [
        strip_punctuation(getattr(t, "form", str(t))).strip()
        for t in tokens
    ]
    clean_focus = strip_punctuation(focus_word).strip()
    fw_list = clean_focus.split()
    if not fw_list:
        return None

    flen = len(fw_list)
    # 1. Exact contiguous word sequence match
    for i in range(len(clean_forms) - flen + 1):
        if clean_forms[i : i + flen] == fw_list:
            return (i, i + flen - 1)

    # 2. Substring/compound match on individual tokens
    # (Focus must be contained in token, e.g. "കുലീന" inside "കുലീനകുടുംബങ്ങളിൽനിന്ന്")
    for i, cf in enumerate(clean_forms):
        if cf and (fw_list[0] == cf or (len(fw_list) == 1 and fw_list[0] in cf)):
            return (i, i)

    return None


def detect_structural_status(
    focused_word: str,
    focused_pos: str,
    tokens: list,
    focus_span: Optional[Tuple[int, int]] = None,
) -> Tuple[str, str]:
    """
    Structural Constituency & Island Constraint Evaluator for Malayalam.

    Determines whether the focused item is:
      1. "INDEPENDENT_CONSTITUENT": A whole maximal phrase (NP, PP, AdvP, predicative XP)
         that possesses syntactic autonomy to undergo Clefting or Reordering.
      2. "SUB_CONSTITUENT": An embedded element inside an NP, PP, AdvP, or VP.
         Under Dravidian syntax (Left Branch Condition, P-stranding ban),
         extracting it alone into a cleft pivot is ungrammatical.
         -> Must be routed to Prosodic Emphasis (PE).
      3. "UNCERTAIN": Structural status is ambiguous or unverified
         -> Conservative fallback to Prosodic Emphasis (PE).

    Returns:
      (status, reason_string)
    """
    clean_fw = strip_punctuation(focused_word).strip()
    focused_pos_upper = (focused_pos or "").upper()

    if focus_span is None and tokens:
        focus_span = find_focus_span_indices(tokens, focused_word)

    if focus_span is None or not tokens:
        # Conservative checks when token stream context is unavailable
        if focused_pos_upper.startswith("JJ") or clean_fw in DEGREE_MODIFIERS or clean_fw in NUMERAL_WORDS:
            return "SUB_CONSTITUENT", f"Isolated modifier '{focused_word}' cannot be verified as an independent constituent"
        return "UNCERTAIN", f"Focus '{focused_word}' cannot be anchored in sentence tokens"

    start_idx, end_idx = focus_span
    end_token = tokens[end_idx]
    end_pos = getattr(end_token, "form_pos", "").upper()
    clean_end = strip_punctuation(getattr(end_token, "form", str(end_token))).strip()
    next_token = tokens[end_idx + 1] if end_idx + 1 < len(tokens) else None
    next_form = strip_punctuation(getattr(next_token, "form", str(next_token))).strip() if next_token else ""
    next_pos = getattr(next_token, "form_pos", "").upper() if next_token else ""

    # =========================================================================
    # 0. PHRASE CHUNKER MAXIMAL PROJECTION & ISLAND CHECK
    # =========================================================================
    try:
        from .phrase_chunker import MalayalamPhraseChunker
        _chunker = MalayalamPhraseChunker()
        _enclosing = _chunker.find_enclosing_phrase(tokens, start_idx, end_idx)
        if _enclosing:
            if end_idx < _enclosing.end_idx:
                # The focus is an embedded modifier leaving the right-edge head noun or postposition behind (Island constraint)
                return "SUB_CONSTITUENT", f"Embedded subconstituent inside {_enclosing.chunk_type} '{_enclosing.text}' (Island constraint: Dravidian P-stranding/Left Branch violation)"
            elif end_idx == _enclosing.end_idx:
                # The focus encompasses the right-edge head noun or the entire maximal projection
                return "INDEPENDENT_CONSTITUENT", f"Whole/Head {_enclosing.chunk_type} ('{_enclosing.text}')"
    except Exception as e:
        logger.debug(f"Phrase chunker evaluation: {e}")

    # =========================================================================
    # A. SUB-CONSTITUENT DETECTION (Left Branch Condition & Island Constraints)
    # =========================================================================

    # 0. Sub-token / Word-internal focus (e.g. "കുലീന" inside "കുലീനകുടുംബങ്ങളിൽനിന്ന്")
    # If the focus target is a strict internal element of a larger token, it cannot independently cleft or reorder.
    if start_idx == end_idx and clean_fw != clean_end and clean_fw in clean_end:
        return "SUB_CONSTITUENT", f"Sub-token internal focus '{clean_fw}' embedded inside token '{clean_end}'"

    # 1. Noun inside PP (Postposition Stranding Ban)
    # Malayalam strictly forbids stranding postpositions (Asher & Kumari 1997; Jayaseelan 1999).
    if next_token is not None and (next_form in POSTPOSITIONS or next_pos == "PSP"):
        return "SUB_CONSTITUENT", f"Embedded noun inside PP stranding postposition '{next_form}'"

    # 2. Attributive Adjective inside NP (Left Branch Condition)
    is_adj = (
        focused_pos_upper.startswith("JJ")
        or end_pos.startswith("JJ")
        or any(clean_end.endswith(sfx) for sfx in ("യ", "ന്ന", "ത്ത", "ിയ"))
        or clean_end in ADJECTIVE_PREDICATES
    )
    # Exclude nominalized/predicative forms (e.g. പഴയതാണ്, നല്ലതാണ്, പഴയത്)
    is_predicative_form = any(clean_end.endswith(sfx) for sfx in ("ത്", "തു്", "ആണ്", "ാണ്", "ന്നത്", "ത്തത്", "ിച്ചത്", "ച്ചത്"))
    if is_adj and not is_predicative_form:
        if next_token is not None:
            is_next_nominal = (
                next_pos.startswith("N_")
                or next_pos.startswith("PR_")
                or any(next_form.endswith(sfx) for sfx in ("ൽ", "ിൽ", "ത്ത്", "നിന്ന്", "ക്ക്", "ന്", "കൾ", "കളെ", "യും", "ഉം"))
            )
            if is_next_nominal:
                return "SUB_CONSTITUENT", f"Attributive adjective inside NP ('{focused_word}' modifying head '{next_form}')"

    # 3. Numeral / Quantifier inside NP
    is_qt = (
        focused_pos_upper.startswith("QT")
        or end_pos.startswith("QT")
        or clean_end in NUMERAL_WORDS
    )
    if is_qt and not _has_nominal_head_in_qt(focused_word):
        if next_token is not None:
            is_next_nominal = (
                next_pos.startswith("N_")
                or next_pos.startswith("PR_")
                or next_pos.startswith("JJ")
                or any(next_form.endswith(sfx) for sfx in ("ൽ", "ിൽ", "ത്ത്", "നിന്ന്", "ക്ക്", "ന്", "കൾ", "കളെ", "യും", "ഉം"))
            )
            if is_next_nominal:
                return "SUB_CONSTITUENT", f"Numeral/quantifier inside NP ('{focused_word}' modifying '{next_form}')"

    # 4. Genitive / Possessor inside NP
    is_genitive = clean_end.endswith(("ന്റെ", "ുടെ", "റെ"))
    if is_genitive and next_token is not None:
        is_next_head = (
            next_pos.startswith("N_")
            or next_pos.startswith("PR_")
            or next_pos.startswith("JJ")
        )
        if is_next_head and next_form not in POSTPOSITIONS:
            return "SUB_CONSTITUENT", f"Genitive possessor inside NP ('{focused_word}' modifying '{next_form}')"

    # 5. Degree Modifier inside AdvP / AP
    is_degree = clean_end in DEGREE_MODIFIERS
    if is_degree and next_token is not None:
        if next_pos.startswith("JJ") or next_pos in ("RB", "ADV") or any(next_form.endswith(sfx) for sfx in ("ായി", "ആയി", "ഓടെ", "ത്തിൽ")):
            return "SUB_CONSTITUENT", f"Degree modifier inside phrase ('{focused_word}' modifying '{next_form}')"

    # =========================================================================
    # B. INDEPENDENT CONSTITUENT VERIFICATION (Maximal Projections: XP)
    # =========================================================================

    # 1. Whole PP (Focus explicitly ends with or contains postposition)
    is_pp = (
        clean_end in POSTPOSITIONS
        or any(clean_end.endswith(sfx) for sfx in ("നിന്ന്", "ൽനിന്ന്", "യിൽനിന്ന്", "കൊണ്ട്", "ിലേക്ക്", "ലേക്ക്", "ങ്കൽ", "ഓട്", "യോട്", "ആൽ", "ാൽ"))
    )
    if is_pp:
        return "INDEPENDENT_CONSTITUENT", "Whole Postpositional Phrase (PP)"

    # 2. Whole Adverbial Phrase (AdvP / Temporal / Manner / Converb)
    is_advp = (
        focused_pos_upper in ("RB", "ADV")
        or end_pos in ("RB", "ADV")
        or clean_end in TIME_WORDS
        or any(clean_end.endswith(sfx) for sfx in MANNER_SUFFIXES)
        or any(clean_end.endswith(sfx) for sfx in ("ായി", "ആയി", "ആയിട്ട്", "ഓടെ", "പ്പോൾ", "ുമ്പോൾ", "മ്പോൾ", "തിനുശേഷം", "തിനുമുമ്പ്", "ത്തിൽ", "ച്ച്", "ിച്ച്", "ഞ്ഞ്", "ഞ്ഞ", "ത്ത്", "്ത്"))
        or (len(focused_word.split()) == 1 and (clean_end.endswith("തവണ") or clean_end.endswith("പ്രാവശ്യം")))
    )
    if is_advp:
        return "INDEPENDENT_CONSTITUENT", "Whole Adverbial Phrase (AdvP)"

    # 3. Whole NP with Case Marking (Accusative DO, Dative IO, Locative, etc.)
    has_case_marking = any(clean_end.endswith(sfx) for sfx in (
        "നെ", "യെ", "ിനെ", "ക്ക്", "യ്ക്ക്", "്ക്ക്", "ന്", "ിന്", "ൽ", "ിൽ", "ത്ത്", "ലെ", "ിലേക്ക്", "ലേക്ക്", "ആൽ", "ാൽ", "ഓട്", "യോട്"
    )) or bool(re.search(r'\b\d{4}(?:ൽ|ലെ|ലേക്ക്)?\b', clean_end))
    if has_case_marking:
        return "INDEPENDENT_CONSTITUENT", f"Whole Case-Marked NP ({clean_end})"

    # 4. Multi-word NP with nominal head (e.g. "കുലീന കുടുംബങ്ങൾ", "രണ്ട് പുസ്തകങ്ങൾ")
    if len(focused_word.split()) > 1:
        if _has_nominal_head_in_qt(focused_word) or end_pos.startswith("N_") or end_pos.startswith("PR_") or any(clean_end.endswith(sfx) for sfx in ("കൾ", "ങ്ങൾ", "മാർ", "ം", "ൻ")):
            return "INDEPENDENT_CONSTITUENT", f"Multi-word Noun Phrase with head noun '{clean_end}'"

    # 5. Standalone Pronoun or Proper Noun
    if clean_end in DEFINITE_PRONOUN_LEMMAS or focused_pos_upper in ("PR_PRP", "N_NNP"):
        return "INDEPENDENT_CONSTITUENT", "Standalone Pronoun / Proper Noun"

    # 6. Compound ordinal/quantifier NP
    if _has_nominal_head_in_qt(focused_word):
        return "INDEPENDENT_CONSTITUENT", "Compound Quantifier/Ordinal NP"

    # 7. Predicative Adjective / Nominalized clause
    if is_predicative_form:
        return "INDEPENDENT_CONSTITUENT", "Predicative/Nominalized Constituent"

    # 8. Single-word Nominative Subject/Object Noun not modifying next token
    is_modifier_word = is_adj or is_qt or is_genitive or is_degree or clean_end in POSTPOSITIONS
    if not is_modifier_word:
        next_has_case = next_token is not None and any(next_form.endswith(sfx) for sfx in ("ൽ", "ിൽ", "ത്ത്", "നിന്ന്", "ക്ക്", "ന്", "ലേക്ക്", "ിലേക്ക്", "കൾ", "കളെ", "യും", "ഉം"))
        if next_token is None or next_has_case or not (next_pos.startswith("N_") or next_form in POSTPOSITIONS):
            return "INDEPENDENT_CONSTITUENT", f"Nominative Noun '{clean_end}'"

    # =========================================================================
    # C. CONSERVATIVE FALLBACK FOR AMBIGUOUS / UNVERIFIED ITEMS
    # =========================================================================
    return "UNCERTAIN", f"Constituent '{focused_word}' cannot be definitively verified as an independent maximal phrase"



def check_cleft_eligibility_structural(
    focus_word: str,
    focus_token=None,
    sentence_ir=None,
    sentence_words: list = None,
) -> Tuple[str, str]:
    """
    Main Gate 1 Decision Function enforcing:
      1. WHOLE / INDEPENDENT CONSTITUENT -> Check matrix verb -> CLEFT
      2. EMBEDDED SUBCONSTITUENT -> PE
      3. UNCERTAIN / AMBIGUOUS -> PE (Conservative fallback)
    """
    tokens = getattr(sentence_ir, "tokens", []) if sentence_ir else (sentence_words or [])
    form_pos = getattr(focus_token, "form_pos", "") if focus_token else ""

    status, reason = detect_structural_status(focus_word, form_pos, tokens)

    if status == "SUB_CONSTITUENT":
        return "PE", f"Blocked: {reason}"

    if status == "UNCERTAIN":
        return "PE", f"Conservative fallback: {reason}"

    # status == "INDEPENDENT_CONSTITUENT":
    # Verify the sentence contains a matrix lexical verb or copular predicate (Moag §2.5, §5.4, §11.2)
    predicate_suffixes = (
        "ുന്നു", "ിച്ചു", "തു", "ി", "ണം", "ും", "ാം", "ഉണ്ട്", "ആണ്", "ാണ്", "യാണ്",
        "ഇല്ല", "ആയിരുന്നു", "ായിരുന്നു", "യായിരുന്നു", "ഉണ്ടായിരുന്നു", "ഉണ്ടാകും",
        "ിച്ചത്", "ച്ചത്", "ത്", "ന്നത്", "ത്തത്", "ഉള്ളത്"
    )
    has_matrix_predicate = False
    if sentence_ir and getattr(sentence_ir, "tokens", None):
        for t in sentence_ir.tokens:
            p = getattr(t, "form_pos", "").upper()
            f = strip_punctuation(getattr(t, "form", "")).strip()
            if p.startswith("V_VM") and not p.startswith("V_VAUX"):
                has_matrix_predicate = True
                break
            if any(f.endswith(sfx) for sfx in predicate_suffixes):
                has_matrix_predicate = True
                break
    elif sentence_words:
        has_matrix_predicate = any(any(strip_punctuation(w).endswith(sfx) for sfx in predicate_suffixes) for w in sentence_words)
    else:
        has_matrix_predicate = True

    if not has_matrix_predicate:
        return "PE", "Blocked: Sentence contains no matrix lexical verb or copular predicate"

    return "CLEFT", f"Allowed: {reason}"


def check_cleft_eligibility(
    focused_word: str,
    focused_pos: str,
    sentence_pos_tags: list[str] = None,
    sentence_ir=None,
    focus_token=None,
) -> str:
    """
    Backward-compatible entry point for Gate 1.
    """
    route, _ = check_cleft_eligibility_structural(
        focus_word=focused_word,
        focus_token=focus_token,
        sentence_ir=sentence_ir,
        sentence_words=sentence_pos_tags,
    )
    return route


# Backward-compatible alias
decide_cleft_or_pe = check_cleft_eligibility


from aligner.alignment_utils import extract_alignment_and_prosody


class CleftResult:
    """Carries all intermediate and alignment information from the cleft generation pipeline."""

    def __init__(
        self,
        original_sentence: str,
        clean_sentence: str,
        focus_word: str,
        focus_token=None,              # ir.Token from MalayalamPipeline
        constituent_type: str = "",    # Phase 1: SUBJECT, DIRECT_OBJECT, DATIVE_OBJECT, LOCATION, TIME, MANNER, INSTRUMENT, CAUSAL_PHRASE, DEFINITE_PRONOUN
        copula_form: str = "",
        copula_path: str = "",
        main_verb: str = "",
        normalized_verb: str = "",
        status: str = "UNRESOLVED",   # VALID, BLOCKED, NEEDS_VERIFICATION, LATER_PHASE, PE_ROUTED, UNRESOLVED, ERROR
        phase: str = "",              # Current / terminating pipeline phase
        cleft_sentence: str = "",
        route: str = "CLEFT",          # "CLEFT" or "PE"
        pe_sentence: str = "",         # Sentence with <PE>...</PE> tags if routed to PE
        error: str = "",
    ):
        self.original_sentence = original_sentence
        self.clean_sentence = clean_sentence
        self.focus_word = focus_word
        self.focus_token = focus_token
        self.constituent_type = constituent_type
        self.copula_form = copula_form
        self.copula_path = copula_path
        self.main_verb = main_verb
        self.normalized_verb = normalized_verb
        self.status = status
        self.phase = phase
        self.cleft_sentence = cleft_sentence
        self.route = route
        self.pe_sentence = pe_sentence
        self.error = error

        # Compute full alignment & prosody data if a cleft/clean sentence and focus word exist
        if self.clean_sentence and self.focus_word:
            target_sent = self.cleft_sentence or self.clean_sentence
            target_sent = re.sub(r"</?PE>", "", target_sent).strip()
            align_data = extract_alignment_and_prosody(
                original_sentence=self.clean_sentence,
                clefted_sentence=target_sent,
                focused_constituent=self.focus_word,
                main_verb=self.main_verb,
                normalized_verb=self.normalized_verb,
            )
            self.prosody_label_sequence = align_data["prosody_label_sequence"]
            self.target_prosody_label_sequence = align_data.get("target_prosody_label_sequence", align_data["prosody_label_sequence"])
            self.source_prosody_label_sequence = align_data.get("source_prosody_label_sequence", [])
            self.position_before = align_data["position_before"]
            self.position_after = align_data["position_after"]
            self.aanu_attachment = align_data["aanu_attachment"]
            self.word_alignments = align_data["word_alignments"]
            self.constituent_alignments = align_data["constituent_alignments"]
            if not self.normalized_verb:
                self.normalized_verb = align_data["nominalized_verb"]
        else:
            self.prosody_label_sequence = []
            self.target_prosody_label_sequence = []
            self.source_prosody_label_sequence = []
            self.position_before = []
            self.position_after = []
            self.aanu_attachment = {"attached_to": "none", "target_word": "", "attached_form": "", "position": None}
            self.word_alignments = []
            self.constituent_alignments = []

    def to_dict(self) -> dict:
        return {
            "original_sentence": self.clean_sentence or self.original_sentence,
            "clefted_sentence": self.cleft_sentence,
            "focused_constituent": self.focus_word,
            "constituent_type": self.constituent_type,
            "prosody_label_sequence": self.prosody_label_sequence,
            "position_before": self.position_before,
            "position_after": self.position_after,
            "aanu_attachment": self.aanu_attachment,
            "nominalized_verb": self.normalized_verb,
            "main_verb": self.main_verb,
            "copula_form": self.copula_form or (self.aanu_attachment.get("attached_form") if getattr(self, "aanu_attachment", None) else ""),
            "copula_path": self.copula_path,
            "word_alignments": self.word_alignments,
            "constituent_alignments": self.constituent_alignments,
            "status": self.status,
            "route": self.route,
            "phase": self.phase,
            "error": self.error,
        }

    def __str__(self):
        lines = [
            f"Original sentence      : {self.clean_sentence or self.original_sentence}",
            f"Clefted sentence       : {self.cleft_sentence or '—'}",
            f"Focused constituent    : {self.focus_word or '—'} (Type: {self.constituent_type or '—'})",
            f"Prosody label sequence : {self.prosody_label_sequence}",
            f"Position before        : {self.position_before}",
            f"Position after         : {self.position_after}",
        ]

        aanu = getattr(self, "aanu_attachment", {})
        if aanu and aanu.get("attached_to") != "none":
            lines.append(f"ആണ് attachment         : {aanu.get('attached_form')} (attached to {aanu.get('attached_to')} '{aanu.get('target_word')}' at index {aanu.get('position')})")
        else:
            lines.append(f"ആണ് attachment         : {self.copula_form or '—'}")

        lines.extend([
            f"Identified main verb   : {self.main_verb or '—'}",
            f"Nominalized verb       : {self.normalized_verb or '—'}",
            f"Strategy Route         : {self.route}",
            f"Status                 : {self.status}",
        ])

        if self.word_alignments:
            lines.append("\nWord Alignments (Before -> After):")
            for wa in self.word_alignments:
                src_str = f"  [{wa['src_index']}] {wa['src_word']}"
                tgt_str = f"[{wa['tgt_index']}] {wa['tgt_word']}" if wa["tgt_index"] is not None else "[—] UNMATCHED"
                lines.append(f"  {src_str:<22} ---> {tgt_str:<25} ({wa['alignment_type']})")

        if self.constituent_alignments:
            lines.append("\nConstituent Alignments:")
            for ca in self.constituent_alignments:
                lines.append(f"  * {ca['role']:<22}: Src {ca['src_position']} '{ca['src_form']}' ---> Tgt {ca['tgt_position']} '{ca['tgt_form']}'")

        if self.pe_sentence:
            lines.append(f"\nPE Sentence            : {self.pe_sentence}")
        if self.phase:
            lines.append(f"Pipeline phase         : {self.phase}")
        if self.copula_path:
            lines.append(f"Copula path            : {self.copula_path}")
        if self.error:
            lines.append(f"Note / Reason          : {self.error}")

        return "\n".join(lines)


class CleftPipeline:
    """
    Malayalam Clefting Pipeline organized into 4 distinct phases:

      PHASE 1 — CLEFT ELIGIBILITY
        Validate whether the focused constituent can naturally be clefted.
        ALLOW: Subject, Direct object, Dative/indirect object, Location, Time,
               Manner, Instrument, Causal phrase, Definite personal pronoun.
        BLOCK: Indefinite pronoun, Adjective predicate, Unverified constituent.

      PHASE 2 — VERB NORMALIZATION (Executed First)
        Delegate finite matrix verb nominalization to VerbNormalizer (Past/Present, Negative).
        Unverified cases remain explicitly marked NEEDS_VERIFICATION.

      PHASE 3 — CLEFT STRUCTURE & COPULA GENERATION (Adding ആണ്)
        Determine focused constituent, case preservation, and attach 'ആണ്' copula.

      PHASE 4 — VALIDATION & SENTENCE ASSEMBLY
        Validate normalized forms, substitute normalized verb first, then attach/substitute copula form.

      SEPARATE LATER PHASE — ABSENCE OF VERB / SUPPORT-VERB INSERTION
        Existential predicates (ഉണ്ട്) and VP/action-focus cases return LATER_PHASE.
    """

    def __init__(self):
        self._malayalam_pipeline = MalayalamPipeline()
        self._analysis_layer    = AnalysisLayer()
        self._copula_layer      = CopulaLayer()
        self._verb_normalizer   = VerbNormalizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, tagged_sentence: str) -> CleftResult:
        tagged_sentence = tagged_sentence.strip()

        # Step 0a: Clause-Bounded Focus Scope Handling
        # In Dravidian syntax (Jayaseelan 2001; Ross 1967 Coordinate Structure Constraint),
        # Clefting is strictly clause-bounded to the local CP/IP domain.
        # If the input contains coordinate clauses separated by semicolons (;) or
        # multiple sentences separated by periods (.), clefting must operate on the
        # specific clause containing the <FF> marker, rather than parsing across clause boundaries.
        if ";" in tagged_sentence and "<FF>" in tagged_sentence:
            clauses = tagged_sentence.split(";")
            matching_clauses = [i for i, c in enumerate(clauses) if "<FF>" in c]
            if len(matching_clauses) == 1:
                target_idx = matching_clauses[0]
                local_res = self.process(clauses[target_idx].strip())
                if local_res.status == "VALID":
                    clauses[target_idx] = " " + local_res.cleft_sentence
                    full_cleft = ";".join(clauses).strip()
                    full_cleft = re.sub(r"\s+", " ", full_cleft)
                    clean_s = re.sub(r"</?FF>", "", tagged_sentence).strip()
                    return CleftResult(
                        original_sentence=tagged_sentence,
                        clean_sentence=clean_s,
                        focus_word=local_res.focus_word,
                        focus_token=local_res.focus_token,
                        constituent_type=local_res.constituent_type,
                        copula_form=local_res.copula_form,
                        copula_path=local_res.copula_path,
                        main_verb=local_res.main_verb,
                        normalized_verb=local_res.normalized_verb,
                        status="VALID",
                        phase="PHASE_4_VALIDATION",
                        cleft_sentence=full_cleft,
                    )

        if re.search(r"\.\s+[^\d]", tagged_sentence) and "<FF>" in tagged_sentence:
            sents = re.split(r"(?<=\.)\s+", tagged_sentence)
            matching_sents = [i for i, s in enumerate(sents) if "<FF>" in s]
            if len(matching_sents) == 1:
                target_idx = matching_sents[0]
                local_res = self.process(sents[target_idx].strip())
                if local_res.status == "VALID":
                    sents[target_idx] = local_res.cleft_sentence
                    full_cleft = " ".join(sents).strip()
                    clean_s = re.sub(r"</?FF>", "", tagged_sentence).strip()
                    return CleftResult(
                        original_sentence=tagged_sentence,
                        clean_sentence=clean_s,
                        focus_word=local_res.focus_word,
                        focus_token=local_res.focus_token,
                        constituent_type=local_res.constituent_type,
                        copula_form=local_res.copula_form,
                        copula_path=local_res.copula_path,
                        main_verb=local_res.main_verb,
                        normalized_verb=local_res.normalized_verb,
                        status="VALID",
                        phase="PHASE_4_VALIDATION",
                        cleft_sentence=full_cleft,
                    )

        if re.search(r":\s+", tagged_sentence) and "<FF>" in tagged_sentence:
            clauses = re.split(r":\s+", tagged_sentence)
            matching_clauses = [i for i, c in enumerate(clauses) if "<FF>" in c]
            if len(matching_clauses) == 1:
                target_idx = matching_clauses[0]
                local_res = self.process(clauses[target_idx].strip())
                if local_res.status == "VALID":
                    clauses[target_idx] = local_res.cleft_sentence
                    full_cleft = " : ".join(clauses).strip()
                    clean_s = re.sub(r"</?FF>", "", tagged_sentence).strip()
                    return CleftResult(
                        original_sentence=tagged_sentence,
                        clean_sentence=clean_s,
                        focus_word=local_res.focus_word,
                        focus_token=local_res.focus_token,
                        constituent_type=local_res.constituent_type,
                        copula_form=local_res.copula_form,
                        copula_path=local_res.copula_path,
                        main_verb=local_res.main_verb,
                        normalized_verb=local_res.normalized_verb,
                        status="VALID",
                        phase="PHASE_4_VALIDATION",
                        cleft_sentence=full_cleft,
                    )

        # Step 0: Focus tag extraction
        matches = FF_PATTERN.findall(tagged_sentence)

        if len(matches) == 0:
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=tagged_sentence,
                focus_word="",
                status="ERROR",
                phase="INITIALIZATION",
                error="No <FF>...<FF> focus marker found. Expected format: <FF>word<FF>",
            )

        if len(matches) > 1:
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=re.sub(r"<FF>", "", tagged_sentence).strip(),
                focus_word=", ".join(matches),
                status="ERROR",
                phase="INITIALIZATION",
                error=f"Multiple focus markers found: {matches}. Expected exactly one.",
            )

        focus_word = matches[0].strip()
        if not focus_word:
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=re.sub(r"<FF>", "", tagged_sentence).strip(),
                focus_word="",
                status="ERROR",
                phase="INITIALIZATION",
                error="Empty <FF><FF> focus marker.",
            )

        clean_sentence = re.sub(r"</?FF>", "", tagged_sentence).strip()

        # Step 0b: Parse sentence IR
        sentence_ir = self._malayalam_pipeline.process(clean_sentence)
        focus_token = self._find_focus_token(sentence_ir, focus_word)

        if focus_token is None:
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=clean_sentence,
                focus_word=focus_word,
                status="ERROR",
                phase="INITIALIZATION",
                error=(
                    f"Focus word '{focus_word}' not found in tokenized IR. "
                    f"Tokens: {[t.form for t in sentence_ir.tokens]}"
                ),
            )

        # --------------------------------------------------------------
        # SECTION 1: CLEFT vs PE ELIGIBILITY
        # --------------------------------------------------------------
        sentence_words = [t.form for t in sentence_ir.tokens]
        route, route_reason = check_cleft_eligibility_structural(
            focus_word=focus_word,
            focus_token=focus_token,
            sentence_ir=sentence_ir,
            sentence_words=sentence_words,
        )

        if route == "PE":
            logger.info("Routing focus '%s' to PE: %s", focus_word, route_reason)
            return CleftResult(
                original_sentence=clean_sentence,
                clean_sentence=clean_sentence,
                focus_word=focus_word,
                focus_token=focus_token,
                status="PE_ROUTED",
                phase="ROUTING_SECTION_1",
                route="PE",
                pe_sentence=clean_sentence,
                cleft_sentence="",
                error=route_reason,
            )

        # Morphological analysis of the focused word (for multi-word constituents, analyze the head word)
        prefix, head_word, appositive = _extract_head_and_appositives(focus_word)
        if not head_word:
            head_word = focus_word.split()[-1] if focus_word else ""
        analyses, analysis_status = self._analysis_layer.analyze_word(head_word)
        if analysis_status == "UNRESOLVED" or not analyses:
            # Robust fallback for Proper Nouns / OOV words (e.g. അരിസോണയിൽ, കാന്യോൺ, കുടുംബങ്ങളിൽനിന്ന്)
            case_guess = "nominative"
            if any(head_word.endswith(sfx) for sfx in ("നിന്ന്", "ൽനിന്ന്", "യിൽനിന്ന്")):
                case_guess = "ablative"
            elif any(head_word.endswith(sfx) for sfx in ("യിൽ", "ൽ", "ത്തിൽ", "ത്ത്")):
                case_guess = "locative"
            elif any(head_word.endswith(sfx) for sfx in ("ന്", "ക്ക്", "ിന്")):
                case_guess = "dative"
            elif any(head_word.endswith(sfx) for sfx in ("യെ", "നെ", "െ")):
                case_guess = "accusative"

            from .copula_pipeline import MorphAnalysis
            selected_analysis = MorphAnalysis(
                raw_analysis=f"{head_word}<n><{case_guess}>",
                lemma=head_word,
                pos="N_NNP",
                case=case_guess,
                number="singular",
                other_features="oov_fallback"
            )
        else:
            selected_analysis = analyses[0]

        # --------------------------------------------------------------
        # PHASE 1: CLEFT ELIGIBILITY
        # --------------------------------------------------------------
        is_eligible, constituent_type, elig_status, elig_reason = self._check_eligibility(
            focus_word=focus_word,
            focus_token=focus_token,
            analysis=selected_analysis,
            sentence_ir=sentence_ir,
            structural_status="MAXIMAL_PHRASE",
            structural_reason=route_reason,
        )

        if not is_eligible:
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=clean_sentence,
                focus_word=focus_word,
                focus_token=focus_token,
                constituent_type=constituent_type,
                status=elig_status,
                phase="PHASE_1_ELIGIBILITY",
                error=elig_reason,
            )

        # --------------------------------------------------------------
        # SEPARATE LATER PHASE CHECK: Absence of Verb / Support-Verb
        # --------------------------------------------------------------
        main_verb = sentence_ir.main_verb or ""

        # Check for existential suppletive predicate (ഉണ്ട്)
        # Proceed with nominalization (ഉണ്ട് -> ഉള്ളത്, വനമുണ്ട് -> വനമുള്ളത്)

        # Check for VP / Action focus (infinitive or verbal focus)
        if constituent_type == "VP_ACTION":
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=clean_sentence,
                focus_word=focus_word,
                focus_token=focus_token,
                constituent_type=constituent_type,
                main_verb=main_verb,
                status="LATER_PHASE",
                phase="LATER_PHASE_SUPPORT_VERB",
                error=(
                    f"VP/action focus on '{focus_word}' requires introducing a support verb "
                    "(e.g. ചെയ്യുക / ചെയ്തത്). Deferred to a separate LATER_PHASE."
                ),
            )

        # Check if the matrix predicate is copular (ആണ്-bearing predicate e.g. അവൻ രാമനാണ് -> അവനാണ് രാമൻ)
        is_copular_pred = bool(main_verb and self._is_copular_predicate(main_verb))

        # --------------------------------------------------------------
        # PHASE 2: VERB NORMALIZATION (Executed First)
        # --------------------------------------------------------------
        normalized_verb = ""

        if is_copular_pred:
            # When the matrix predicate itself is copular, decopularize the background predicate
            # (e.g. രാമനാണ് -> രാമൻ, ആണ് -> "")
            normalized_verb = self._decopularize(main_verb)
            norm_status = "VALID"
        elif main_verb and main_verb != focus_word:
            # Delegate strictly to VerbNormalizer for finite matrix verbs
            norm_res = self._verb_normalizer.normalize(main_verb)
            norm_status = norm_res.get("status", "UNRESOLVED")

            if norm_status == "NEEDS_VERIFICATION":
                return CleftResult(
                    original_sentence=tagged_sentence,
                    clean_sentence=clean_sentence,
                    focus_word=focus_word,
                    focus_token=focus_token,
                    constituent_type=constituent_type,
                    main_verb=main_verb,
                    status="NEEDS_VERIFICATION",
                    phase="PHASE_2_NORMALIZATION",
                    error=f"Matrix verb normalization requires verification: {norm_res.get('failure_reason', '')}",
                )

            if norm_status != "VALID" or not norm_res.get("normalized"):
                if not self._is_verbal_token(main_verb):
                    normalized_verb = main_verb
                else:
                    return CleftResult(
                        original_sentence=tagged_sentence,
                        clean_sentence=clean_sentence,
                        focus_word=focus_word,
                        focus_token=focus_token,
                        constituent_type=constituent_type,
                        main_verb=main_verb,
                        status="UNRESOLVED",
                        phase="PHASE_2_NORMALIZATION",
                        error=f"Matrix verb normalization failed: {norm_res.get('failure_reason', '')}",
                    )
            else:
                normalized_verb = norm_res["normalized"]
        else:
            normalized_verb = ""

        # --------------------------------------------------------------
        # PHASE 3: CLEFT STRUCTURE & COPULA GENERATION (Adding ആണ്)
        # --------------------------------------------------------------
        year_match = re.match(r"^(\d{1,4})([\s\-]*)ലെ$", focus_word.strip())
        if year_match:
            sep = year_match.group(2)
            if not sep:
                # Direct suffixation without space: e.g. 2003ലെ -> 2003ലെയാണ്
                copula_form = f"{year_match.group(1)}ലെയാണ്"
            else:
                # Detached particle with space: e.g. 2015 ലെ -> 2015 ലാണ്
                copula_form = f"{year_match.group(1)}{sep}ലാണ്"
            _preserved = "YES"
            copula_path = "temporal-year-locative"
        else:
            prefix, head_word, appositive = _extract_head_and_appositives(focus_word)

            if head_word:
                head_copula, _preserved, copula_path = self._copula_layer.transform_word(
                    head_word, selected_analysis
                )
                parts = [p for p in (prefix, head_copula, appositive) if p]
                copula_form = " ".join(parts)
            elif constituent_type == "TIME" and any(
                focus_word.endswith(sfx)
                for sfx in TEMPORAL_CLAUSE_SUFFIXES
            ):
                copula_form = self._make_temporal_copula(focus_word)
                _preserved = "YES"
                copula_path = "temporal-clause"
            else:
                copula_form, _preserved, copula_path = self._copula_layer.transform_word(
                    focus_word, selected_analysis
                )

        if copula_path in ("not-supported",) or copula_form in ("NOT SUPPORTED", "ALREADY COPULAR"):
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=clean_sentence,
                focus_word=focus_word,
                focus_token=focus_token,
                constituent_type=constituent_type,
                copula_form=copula_form,
                copula_path=copula_path,
                main_verb=main_verb,
                normalized_verb=normalized_verb,
                status="NOT_SUPPORTED",
                phase="PHASE_3_STRUCTURE",
                error=f"CopulaLayer could not attach copula to '{focus_word}' via path '{copula_path}'.",
            )

        # --------------------------------------------------------------
        # PHASE 4: VALIDATION & SENTENCE ASSEMBLY
        # --------------------------------------------------------------
        # PHASE 4: VALIDATION & SENTENCE ASSEMBLY
        # --------------------------------------------------------------
        # 1. Substitute the copula-attached focus constituent directly at the tagged position.
        if re.search(r"<FF>.*?(?:</FF>|<FF>)\s+എന്ന്", tagged_sentence):
            cleft_sentence = re.sub(r"<FF>(.*?)(?:</FF>|<FF>)\s+എന്ന്", r"\1 എന്നാണ്", tagged_sentence, count=1)
        elif re.search(r"<FF>.*?(?:</FF>|<FF>)\s+എന്നു്", tagged_sentence):
            cleft_sentence = re.sub(r"<FF>(.*?)(?:</FF>|<FF>)\s+എന്നു്", r"\1 എന്നാണ്", tagged_sentence, count=1)
        elif "<FF>" in tagged_sentence:
            cleft_sentence = re.sub(r"<FF>.*?(?:</FF>|<FF>)", copula_form, tagged_sentence, count=1)
        else:
            cleft_sentence = self._substitute(clean_sentence, focus_word, copula_form)

        # Remove any residual <FF> tags
        cleft_sentence = re.sub(r"</?FF>", "", cleft_sentence).strip()

        # 2. Substitute the normalized main verb
        if main_verb and normalized_verb is not None and main_verb != focus_word and main_verb != copula_form:
            cleft_sentence = self._substitute(cleft_sentence, main_verb, normalized_verb)

        # If decopularizing a copular predicate, ensure no trailing duplicate orphaned copula at sentence end
        if is_copular_pred:
            cleft_sentence = re.sub(r"\s+ആണ്([.,!?;:]*)$", r"\1", cleft_sentence)

        # Normalize whitespace and punctuation
        cleft_sentence = re.sub(r"\s+([.,!?;:])", r"\1", cleft_sentence)
        cleft_sentence = re.sub(r"\s+", " ", cleft_sentence).strip()

        # 3. Strict Cleft Completion Validation:
        # Cleft will complete ONLY if 'aanu' is attached to the phrase AND the verb is normalised!
        copula_suffixes = ("ാണ്", "ആണ്", "യാണ്", "മായാണ്", "ാൺ", "വാൺ", "യായിട്ടാണ്", "ആയിട്ടാണ്")
        has_copula = any(cop in copula_form for cop in copula_suffixes) and (
            copula_form in cleft_sentence or any(cop in cleft_sentence for cop in copula_suffixes)
        )

        if is_copular_pred or not main_verb:
            has_norm_verb = True
        else:
            has_norm_verb = bool(
                normalized_verb and normalized_verb != main_verb and (
                    normalized_verb in cleft_sentence or
                    any(normalized_verb.endswith(sfx) for sfx in ("ത്", "തു്", "ച്ചത്", "ട്ടത്", "ന്നത്", "ത്തത്", "ുന്നത്"))
                )
            )

        if not (has_copula and has_norm_verb):
            logger.warning("Cleft incomplete: copula_attached=%s, verb_normalized=%s", has_copula, has_norm_verb)
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=clean_sentence,
                focus_word=focus_word,
                focus_token=focus_token,
                constituent_type=constituent_type,
                copula_form=copula_form,
                copula_path=copula_path,
                main_verb=main_verb,
                normalized_verb=normalized_verb,
                status="CLEFT_FAILED",
                phase="PHASE_4_VALIDATION",
                cleft_sentence="",
                error=f"Cleft incomplete: copula_attached={has_copula}, verb_normalized={has_norm_verb}",
            )

        return CleftResult(
            original_sentence=tagged_sentence,
            clean_sentence=clean_sentence,
            focus_word=focus_word,
            focus_token=focus_token,
            constituent_type=constituent_type,
            copula_form=copula_form,
            copula_path=copula_path,
            main_verb=main_verb,
            normalized_verb=normalized_verb,
            status="VALID",
            phase="PHASE_4_VALIDATION",
            cleft_sentence=cleft_sentence,
        )

    # ------------------------------------------------------------------
    # Phase 1 Eligibility: Feature-Driven Classification
    # ------------------------------------------------------------------

    def _get_case(self, analysis) -> str:
        if analysis and hasattr(analysis, "case") and analysis.case:
            return analysis.case.lower()
        return ""

    def _is_indefinite_or_interrogative_pronoun(self, analysis, token=None) -> bool:
        if not analysis:
            return False
        word_text = getattr(analysis, "word", "") or (token.form if token else "")
        if any(word_text.endswith(sfx) for sfx in ("മെന്നോ", "എന്നോ")):
            return False
        raw = getattr(analysis, "raw_analysis", "")
        lemma = getattr(analysis, "lemma", "")
        tags = getattr(analysis, "tags", []) or []
        if "<qn>" in raw or "qn" in tags:
            return True
        if "എങ്കിലും" in raw or "എങ്കിലും" in lemma:
            return True
        form_pos = getattr(token, "form_pos", "") if token else ""
        if form_pos in ("PR_PRI", "PR_PRQ", "DM_DMR"):
            return True
        if lemma in INDEFINITE_PRONOUNS or getattr(analysis, "word", "") in INDEFINITE_PRONOUNS:
            return True
        return False

    def _is_predicate_adjective(self, analysis, dep_relation, token=None) -> bool:
        """
        Block an adjective only when it functions as a predicate.
        Do NOT block adjectives merely because POS == ADJ.
        """
        if not analysis:
            return False
        pos = getattr(analysis, "pos", "")
        lemma = getattr(analysis, "lemma", "")
        form_pos = getattr(token, "form_pos", "") if token else ""
        is_adj = (
            pos in ("ADJ", "adj", "adjective")
            or form_pos.startswith("JJ")
            or lemma in ADJECTIVE_PREDICATES
        )
        has_case = bool(getattr(analysis, "case", None) and analysis.case not in ("nominative", None))
        if is_adj and not has_case and dep_relation in ("cop_pred", "xcomp", "root", "predicate"):
            return True
        return False

    def _is_definite_personal_pronoun(self, analysis, token=None) -> bool:
        """
        Determine whether the analysis identifies a personal pronoun
        with person/number features and without indefinite/interrogative features.
        """
        if not analysis:
            return False
        if self._is_indefinite_or_interrogative_pronoun(analysis, token):
            return False
        raw = getattr(analysis, "raw_analysis", "")
        pos = getattr(analysis, "pos", "")
        lemma = getattr(analysis, "lemma", "")
        person = getattr(analysis, "person", None)
        if "<prn>" in raw or pos in ("PRON", "prn"):
            if person in ("first", "second", "third", "1", "2", "3") or lemma in DEFINITE_PRONOUN_LEMMAS:
                return True
        return lemma in DEFINITE_PRONOUN_LEMMAS

    def _extract_dependency_relation(self, focus_token, sentence_ir, analysis, semantic_tag="") -> str:
        if not sentence_ir or not focus_token:
            return "unk"
        tokens = sentence_ir.tokens
        forms = [t.form for t in tokens]
        if focus_token.form not in forms:
            return "unk"
        idx = forms.index(focus_token.form)
        form_pos = getattr(focus_token, "form_pos", "")

        # Copular predicate check (immediately preceding copula ആണ്)
        if idx + 1 < len(tokens) and tokens[idx + 1].form == "ആണ്":
            return "cop_pred"
        if tokens and tokens[-1].form == "ആണ്" and idx == len(tokens) - 2:
            return "cop_pred"

        # Adverbial / adjunct check
        if form_pos in ("RB", "ADV") or getattr(analysis, "pos", "") in ("ADV", "adv") or semantic_tag in ("TEMPORAL", "MANNER", "CAUSE"):
            return "advmod" if form_pos in ("RB", "ADV") else "obl"

        case = self._get_case(analysis)
        if case == "accusative":
            return "obj"
        if case == "dative":
            return "iobj"
        if case in ("locative", "instrumental", "sociative"):
            return "obl"

        if idx == 0:
            return "nsubj"

        # Check if preceding token was subject and current token is a non-adverb nominal
        prev_tokens = tokens[:idx]
        has_prev_subj = any(t.form_pos.startswith("N_") or t.form_pos.startswith("PR_") for t in prev_tokens)
        if has_prev_subj and idx < len(tokens) - 1 and (form_pos.startswith("N_") or form_pos.startswith("PR_")):
            return "obj"

        return "nsubj"

    def _make_temporal_copula(self, word: str) -> str:
        if word.endswith("പ്പോൾ"):
            return word[:-len("പ്പോൾ")] + "പ്പോഴാണ്"

        if word.endswith("മ്പോൾ"):
            return word[:-len("മ്പോൾ")] + "മ്പോഴാണ്"

        if word.endswith("ശേഷം"):
            return word[:-len("ശേഷം")] + "ശേഷമാണ്"

        if word.endswith("മുമ്പ്"):
            return word[:-len("മുമ്പ്")] + "മുമ്പാണ്"

        return ""

    def _extract_semantic_tag(self, focus_word: str, focus_token, analysis) -> str:
        if not analysis:
            return ""
        lemma = getattr(analysis, "lemma", "")
        raw = getattr(analysis, "raw_analysis", "")
        form_pos = getattr(focus_token, "form_pos", "") if focus_token else ""
        case = self._get_case(analysis)

        # 1. Temporal semantics
        if (
            lemma in TIME_WORDS
            or focus_word in TIME_WORDS
            or any(focus_word.endswith(sfx) for sfx in TEMPORAL_CLAUSE_SUFFIXES)
            or "<cvb-adv-part-simul>" in raw
            or "<cvb-adv-part-past-simul>" in raw
        ):
            return "TEMPORAL"

        # 2. Manner semantics
        # CVBs (conjunctive/converbal forms) in Malayalam are manner/sequential adjuncts.
        # Any FST-tagged <cvb> that is NOT a simultaneous converbal (those are TEMPORAL above)
        # should be classified as MANNER so they pass eligibility and are not deferred as VP_ACTION.
        is_cvb_manner = (
            ("<cvb>" in raw or "<cvb-adv-part" in raw)
            and "<cvb-adv-part-simul>" not in raw
            and "<cvb-adv-part-past-simul>" not in raw
        )
        if (
            any(focus_word.endswith(sfx) for sfx in MANNER_SUFFIXES)
            or form_pos == "RB"
            or getattr(analysis, "pos", "") in ("ADV", "adv")
            or focus_word.endswith("ത്തിൽ")
            or is_cvb_manner
        ):
            return "MANNER"

        # 3. Cause semantics
        if focus_word in ("അതുകൊണ്ട്", "അതുകൊണ്ടു്") or focus_word.endswith("കാരണം") or focus_word.endswith("മൂലം") or case == "sociative":
            return "CAUSE"

        # 4. Location / Source semantics (distinguished spatial locatives & ablatives)
        if case in ("locative", "ablative") or focus_word.endswith(("ിൽ", "ൽ", "ങ്കൽ", "ത്ത്", "നിന്ന്", "ൽനിന്ന്", "യിൽനിന്ന്")):
            return "LOCATION"

        return ""

    def _check_eligibility(
        self,
        focus_word: str,
        focus_token,
        analysis,
        sentence_ir=None,
        structural_status: Optional[str] = None,
        structural_reason: str = "",
    ):
        """
        Determine whether the focused constituent is eligible for Malayalam clefting
        following the priority order:
        1. Structural status (from Gate 1): Subconstituents are NEVER independently clefted
        2. Hard grammatical blockers (indefinite pronouns)
        3. Predicate adjectives (copular predicates)
        4. Morphological case & semantic adjunct tags (TIME, MANNER, CAUSE, LOCATION, Accusative, Dative, Locative, Instrumental)
        5. Core syntactic arguments (Subject, Direct Object, Dative)
        6. Definite personal pronouns
        7. Action / VP focus (infinitives or bare verbal roots)
        8. Safe fallback: NEEDS_VERIFICATION
        """
        semantic_tag = self._extract_semantic_tag(focus_word, focus_token, analysis)
        dep_relation = self._extract_dependency_relation(focus_token, sentence_ir, analysis, semantic_tag)
        case = self._get_case(analysis)
        form_pos = getattr(focus_token, "form_pos", "") if focus_token else ""

        # ---------------------------------------------------------
        # 1. STRUCTURAL CONSTITUENT CHECK (Gate 1 Status Integration)
        # ---------------------------------------------------------
        if structural_status is None and sentence_ir and getattr(sentence_ir, "tokens", None):
            structural_status, structural_reason = detect_structural_status(
                focus_word, form_pos, sentence_ir.tokens
            )

        if structural_status in ("SUBCONSTITUENT", "SUB_CONSTITUENT", "UNCERTAIN"):
            return (
                False,
                "SUB_CONSTITUENT",
                "BLOCKED",
                structural_reason or "SUBCONSTITUENT_NOT_CLEFTED: Embedded constituents cannot be independently clefted.",
            )

        # ---------------------------------------------------------
        # 2. HARD GRAMMATICAL BLOCKERS & PREDICATE ADJECTIVES
        # ---------------------------------------------------------
        if self._is_indefinite_or_interrogative_pronoun(analysis, focus_token):
            return False, "INDEFINITE_PRONOUN", "BLOCKED", f"Indefinite/interrogative pronoun '{focus_word}' cannot naturally be cleft-focused."

        # Whole predicate adjective (cop_pred / predicate)
        if self._is_predicate_adjective(analysis, dep_relation, focus_token):
            return True, "ADJECTIVE_NOMINALIZED", "ALLOWED", ""

        # ---------------------------------------------------------
        # 3. SEMANTIC ADJUNCTS (TEMPORAL, MANNER, CAUSE, LOCATION)
        # ---------------------------------------------------------
        if semantic_tag == "TEMPORAL":
            return True, "TIME", "ALLOWED", ""

        if semantic_tag == "MANNER":
            return True, "MANNER", "ALLOWED", ""

        if semantic_tag == "CAUSE":
            return True, "CAUSAL_PHRASE", "ALLOWED", ""

        if semantic_tag == "LOCATION":
            return True, "LOCATION", "ALLOWED", ""

        if focus_word.endswith("മാത്രം") or focus_word.endswith("മാത്രമേ") or focus_word.endswith("കൂടി"):
            return True, "EXCLUSIVE_FOCUS", "ALLOWED", ""

        # ---------------------------------------------------------
        # 4. MORPHOLOGICAL CASE
        # ---------------------------------------------------------
        if case == "accusative":
            return True, "DIRECT_OBJECT", "ALLOWED", ""

        if case == "dative":
            return True, "DATIVE_OBJECT", "ALLOWED", ""

        if case == "instrumental":
            return True, "INSTRUMENT", "ALLOWED", ""

        if case == "locative":
            return True, "LOCATION", "ALLOWED", ""

        if case == "sociative":
            return True, "CAUSAL_PHRASE", "ALLOWED", ""

        if case == "ablative":
            return True, "LOCATION", "ALLOWED", ""

        # Genitive: A pre-nominal genitive possessor inside an NP cannot be independently clefted.
        # Only allowed if functioning as an independent copular/predicate complement.
        if case == "genitive" or any(focus_word.endswith(sfx) for sfx in ("ന്റെ", "യുടെ", "ുടെ", "ിന്റെ", "്റെ")):
            if dep_relation in ("cop_pred", "predicate"):
                return True, "PREDICATE_GENITIVE", "ALLOWED", ""
            return False, "GENITIVE_POSSESSOR", "BLOCKED", f"Genitive possessor '{focus_word}' cannot be independently clefted away from its head noun."

        # ---------------------------------------------------------
        # 5. SYNTACTIC CORE ARGUMENTS & DEFINITE PRONOUNS
        # ---------------------------------------------------------
        if self._is_definite_personal_pronoun(analysis, focus_token):
            return True, "DEFINITE_PRONOUN", "ALLOWED", ""

        if dep_relation == "nsubj":
            return True, "SUBJECT", "ALLOWED", ""

        if dep_relation in ("obj", "dobj"):
            return True, "DIRECT_OBJECT", "ALLOWED", ""

        if dep_relation in ("iobj", "dative"):
            return True, "DATIVE_OBJECT", "ALLOWED", ""

        # ---------------------------------------------------------
        # 6. ACTION / VP FOCUS (infinitives or bare verbal roots)
        # ---------------------------------------------------------
        if (
            focus_word.endswith("ുക")
            or focus_word.endswith("ക്ക")
            or form_pos.startswith("V_")
            or getattr(analysis, "pos", "") in ("VERB", "verb")
        ):
            return True, "VP_ACTION", "ALLOWED", ""

        # ---------------------------------------------------------
        # 7. INDEPENDENT QUANTIFIERS / NUMERALS (as full argument heads)
        # ---------------------------------------------------------
        # Only whole independent quantifiers (functioning as arguments / complements) are allowed.
        # Attributive quantifiers/numerals modifying a noun are subconstituents, blocked in Step 1.
        if form_pos.startswith("QT") or focus_word in ("കുറച്ച്", "അല്പം", "ഏറെ", "കൂടുതൽ", "ധാരാളം", "മൂന്ന്", "രണ്ട്", "ഒന്ന്", "പത്ത്"):
            if dep_relation in ("nsubj", "obj", "cop_pred", "root"):
                return True, "QUANTIFIER_NOMINALIZED", "ALLOWED", ""
            return False, "QUANTIFIER", "BLOCKED", f"Quantifier/numeral '{focus_word}' without independent argument function cannot be clefted independently."

        # ---------------------------------------------------------
        # 8. SAFE FALLBACK
        # ---------------------------------------------------------
        return False, "UNVERIFIED_CONSTITUENT", "NEEDS_VERIFICATION", f"Constituent '{focus_word}' could not be verified by morphology or syntax."

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_focus_token(self, sentence_ir, focus_word: str):
        clean_fw = strip_punctuation(focus_word).strip()
        words = clean_fw.split()
        target_head = words[-1] if words else clean_fw

        # 1. Match the head token (rightmost lexical word in multi-word phrase)
        for token in reversed(sentence_ir.tokens):
            if strip_punctuation(token.form) == target_head:
                return token

        # 2. Exact match with single token
        for token in sentence_ir.tokens:
            if strip_punctuation(token.form) == clean_fw:
                return token

        # 3. Substring match on head token
        for token in reversed(sentence_ir.tokens):
            clean_tf = strip_punctuation(token.form)
            if clean_tf and (clean_tf in target_head or target_head in clean_tf):
                return token
        return None

    def _is_progressive_verb(self, word: str) -> bool:
        """
        Check if a word is a progressive/durative finite verbal construction ending in -ആണ്.
        e.g. നൽകിക്കൊണ്ടിരിക്കുകയാണ്, ചെയ്യുകയാണ്, വായിക്കുകയാണ്.
        These are verbal predicates that must be nominalized by VerbNormalizer,
        NOT equative copular nouns like ഡോക്ടറാണ് or രാമനാണ്.
        """
        w = strip_punctuation(word)
        if not w:
            return False
        if (
            w.endswith("കൊണ്ടിരിക്കുകയാണ്")
            or w.endswith("ക്കൊണ്ടിരിക്കുകയാണ്")
            or w.endswith("ിച്ചുകൊണ്ടിരിക്കുകയാണ്")
            or w.endswith("ഞ്ഞുകൊണ്ടിരിക്കുകയാണ്")
            or w.endswith("ച്ചുകൊണ്ടിരിക്കുകയാണ്")
            or w.endswith("ക്കുകയാണ്")
            or (w.endswith("ുകയാണ്") and len(w) > 4)
        ):
            return True
        return False

    def _substitute(self, sentence: str, target: str, replacement: str) -> str:
        # In head-final Malayalam syntax, predicates and matrix targets occur towards the end;
        # using rfind ensures the predicate at clause end is replaced rather than an earlier identical noun/root.
        r_idx = sentence.rfind(target)
        if r_idx != -1:
            return sentence[:r_idx] + replacement + sentence[r_idx + len(target):]
        clean_target = strip_punctuation(target)
        if clean_target:
            r_idx = sentence.rfind(clean_target)
            if r_idx != -1:
                return sentence[:r_idx] + replacement + sentence[r_idx + len(clean_target):]

        def _norm_c(t: str) -> str:
            if not t:
                return ""
            t = re.sub(r'ല\u0d4d\u200d?|ല്\u200d?', 'ൽ', t)
            t = re.sub(r'ള\u0d4d\u200d?|ളല്\u200d?', 'ൾ', t)
            t = re.sub(r'ന\u0d4d\u200d?|ന്\u200d?', 'ൻ', t)
            t = re.sub(r'ര\u0d4d\u200d?|ര്\u200d?', 'ർ', t)
            t = re.sub(r'ണ\u0d4d\u200d?|ണ്\u200d?', 'ൺ', t)
            return re.sub(r'[\u200b\u200c\u200d\ufeff]', '', t)

        # Token-window matching (robust to Chillu encoding variations and character length shifts)
        ml_toks = sentence.split()
        tg_toks = target.split()
        if tg_toks and len(tg_toks) <= len(ml_toks):
            tg_norm = [_norm_c(strip_punctuation(w)) for w in tg_toks]
            for i in range(len(ml_toks) - len(tg_toks) + 1):
                window_norm = [_norm_c(strip_punctuation(w)) for w in ml_toks[i : i + len(tg_toks)]]
                if window_norm == tg_norm:
                    last_orig = ml_toks[i + len(tg_toks) - 1]
                    punct = ""
                    while last_orig and last_orig[-1] in (",", ".", ";", "!", "?"):
                        punct = last_orig[-1] + punct
                        last_orig = last_orig[:-1]
                    
                    before = ml_toks[:i]
                    after = ml_toks[i + len(tg_toks):]
                    res_tokens = before + [replacement + punct] + after
                    return " ".join(res_tokens)

        norm_sent = _norm_c(sentence)
        norm_target = _norm_c(target)
        if norm_target in norm_sent:
            pos = norm_sent.find(norm_target)
            if pos != -1:
                return sentence[:pos] + replacement + sentence[pos + len(target):]

        norm_clean_target = _norm_c(clean_target)
        if norm_clean_target and norm_clean_target in norm_sent:
            pos = norm_sent.find(norm_clean_target)
            if pos != -1:
                return sentence[:pos] + replacement + sentence[pos + len(clean_target):]

        # Multi-word substitution matching internal punctuation/commas
        words = [re.escape(_norm_c(strip_punctuation(w))) for w in (target or "").split() if strip_punctuation(w)]
        if words:
            pattern = r"\s*[\,\.\?\!\;\:\-]*\s*".join(words)
            if re.search(pattern, norm_sent):
                return re.sub(pattern, replacement, sentence, count=1)

        return sentence

    # ------------------------------------------------------------------
    # Copular predicate handling (Clause-level)
    # ------------------------------------------------------------------

    def _is_verbal_token(self, word: str) -> bool:
        clean = strip_punctuation(word).strip()
        if not clean:
            return False
        try:
            analyses = self._analysis_layer.analyser.analyse(clean)
            for raw, _ in analyses:
                if "<v>" in raw or "<verb>" in raw:
                    return True
        except Exception:
            pass
        return any(clean.endswith(sfx) for sfx in ("ുന്നു", "ിച്ചു", "തു", "ി", "ണം", "ും", "ാം", "ഉണ്ട്", "മുണ്ട്", "ആണ്", "ഇല്ല", "ഉണ്ടായിരുന്നു", "ഉണ്ടാകും", "ിച്ചത്", "ച്ചത്", "ത്", "ന്നത്", "ത്തത്"))

    def _is_copular_predicate(self, word: str) -> bool:
        """
        Check if a word is an equative copular predicate (contains ആണ്).
        Progressive aspect verbs (e.g. നൽകിക്കൊണ്ടിരിക്കുകയാണ്) return False.
        """
        if self._is_progressive_verb(word):
            return False

        if word == "ആണ്" or word.endswith("ാണ്") or word.endswith("ആണ്") or word.endswith("യാണ്"):
            return True
        analyses = self._analysis_layer.analyser.analyse(word)
        for raw, _ in analyses:
            if "ആണ്<aff>" in raw:
                # If base is a verb, it is a verbal predicate, not an equative copular noun
                if "<v>" not in raw:
                    return True
        return False

    def _decopularize(self, word: str) -> str:
        """
        Strip the copular suffix ആണ് from a copular predicate to recover
        the base non-focused constituent for cleft construction.

        Reverse sandhi rules (ordered most-specific → least-specific):

        Virāma/Chandrakkala restoration:
          Cാണ് → C്   (restore chandrakkala after stripping ാണ്)
        Consonant–Vowel (Chillu) restoration:
          നാണ് → ൻ, ളാണ് → ൾ, രാണ് → ർ, ലാണ് → ൽ
        Anusvāra restoration:
          മാണ് → ം
        Yakāra / Vakāra restoration:
          യാണ് → (strip യ)
          വാണ് → (strip വ — but preserve if stem ends in വ്)
        """
        if word == "ആണ്":
            return ""

        # Strategy 1: mlmorph analysis → generation (with roundtrip validation)
        from .copula_pipeline import attach_aanu_surface
        analyses = self._analysis_layer.analyser.analyse(word)
        for raw, _ in analyses:
            if "ആണ്<aff>" in raw:
                idx = raw.find("ആണ്<aff>")
                stripped_target = raw[:idx]
                gen_results = self._copula_layer.generator.generate(stripped_target)
                if gen_results:
                    candidate = gen_results[0][0]
                    # Roundtrip validation: attach_aanu_surface(candidate) must
                    # reconstruct the original copular word. If not, the FST
                    # picked a wrong lemma — fall through to surface rules.
                    if attach_aanu_surface(candidate) == word:
                        return candidate

        # Strategy 2: Surface morphophonemic fallback (table-driven sandhi restoration)
        # Each entry: (copular_suffix, base_suffix)
        # Ordered most-specific → least-specific to prevent false matches.
        _DECOPULARIZE_TABLE = (
            # --- Case-suffix patterns (specific allomorphs first) ---
            # Sociative
            ("ത്തോടാണ്", "ത്തോട്"),
            ("ഇനോടാണ്", "ഇനോട്"),
            ("വിനോടാണ്", "വിനോട്"),
            ("യോടാണ്", "യോട്"),
            ("ഓടാണ്", "ഓട്"),
            # Ablative
            ("ൽനിന്നാണ്", "ൽനിന്ന്"),
            ("നിന്നാണ്", "നിന്ന്"),
            # Instrumental
            ("കൊണ്ടാണ്", "കൊണ്ട്"),
            ("ത്തിനാലാണ്", "ത്തിനാൽ"),
            ("ഇനാലാണ്", "ഇനാൽ"),
            ("യാലാണ്", "യാൽ"),
            ("ആലാണ്", "ആൽ"),
            # Dative
            ("യ്ക്കാണ്", "യ്ക്ക്"),
            ("ക്കാണ്", "ക്ക്"),
            # Locative
            ("ത്തിലാണ്", "ത്തിൽ"),
            ("വിലാണ്", "വിൽ"),
            ("യിലാണ്", "യിൽ"),
            ("റ്റിലാണ്", "റ്റിൽ"),
            ("ലിലാണ്", "ലിൽ"),
            ("ലാണ്", "ൽ"),
            # --- Geminate / cluster Virāma restoration ---
            ("ത്താണ്", "ത്ത്"),
            ("ന്നാണ്", "ന്ന്"),
            ("ണ്ടാണ്", "ണ്ട്"),
            ("ച്ചാണ്", "ച്ച്"),
            ("ട്ടാണ്", "ട്ട്"),
            ("പ്പാണ്", "പ്പ്"),
            ("ല്ലാണ്", "ല്ല്"),
            ("ള്ളാണ്", "ള്ള്"),
            ("റ്റാണ്", "റ്റ്"),
            # --- Chillu restoration (Vyañjana–Svara reverse) ---
            ("നാണ്", "ൻ"),
            ("ളാണ്", "ൾ"),
            ("രാണ്", "ർ"),
            # --- Anusvāra restoration ---
            ("മാണ്", "ം"),
            # --- Generic Virāma restoration ---
            ("ടാണ്", "ട്"),
            ("താണ്", "ത്"),
            # --- Vakāra restoration (rounded vowels) ---
            ("വാണ്", "വ്"),
            # --- Predicate complement restoration ---
            ("ായാണ്", "ായി"),
            # --- Yakāra restoration (front/central vowels) ---
            ("യാണ്", ""),   # strip the inserted യ
            # --- Bare ആണ് suffix ---
            ("ആണ്", ""),
        )

        for copular_sfx, base_sfx in _DECOPULARIZE_TABLE:
            if word.endswith(copular_sfx):
                return word[:-len(copular_sfx)] + base_sfx

        return word


