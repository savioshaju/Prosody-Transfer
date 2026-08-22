import re
import logging

from .malayalam_pipeline import MalayalamPipeline
from .copula_pipeline import AnalysisLayer, CopulaLayer
from .verb_normalizer import VerbNormalizer

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

MANNER_SUFFIXES = ("ആയി", "ആയിട്ട്", "ഓടെ", "ആംവണ്ണം", "പ്രകാരം")

ADJECTIVE_PREDICATES = {
    "സുന്ദരം", "മനോഹരം", "നല്ല", "വലിയ", "ചെറിയ", "മിടുക്കൻ", "മിടുക്കി",
    "മോശം", "കേമം", "ധീരം", "സത്യം", "ശുദ്ധം", "വ്യക്തം", "ശരി", "തെറ്റ്"
}


# ---------------------------------------------------------------------------
# SECTION 1: CLEFT vs PE ELIGIBILITY DECISION FUNCTION
# ---------------------------------------------------------------------------

def check_cleft_eligibility(focused_word: str, focused_pos: str, sentence_pos_tags: list[str]) -> str:
    """
    Given the focused constituent and sentence-level POS information,
    decide whether the sentence enters the CLEFT pipeline or the PE pipeline.

    Inputs:
      - focused_word: str
      - focused_pos: str
      - sentence_pos_tags: list[str]

    Output:
      - "CLEFT" or "PE"

    Decision rules:
      IF focused POS = QT_QTO or frequency expression (e.g. രണ്ടുതവണ)
              → PE
      ELSE IF focused POS = JJ
              → PE
      ELSE IF sentence contains only V_VAUX (no lexical V_VM_*)
              → PE
      ELSE IF focused constituent is a supported nominal/adverbial constituent
              AND sentence contains a lexical V_VM_*
              → CLEFT
      ELSE
              → PE
    """
    focused_pos_upper = (focused_pos or "").upper()
    sentence_pos_upper = [tag.upper() for tag in (sentence_pos_tags or [])]

    # 1. Quantifiers / Numerals / Frequency expressions
    if (
        focused_pos_upper.startswith("QT")
        or focused_pos_upper == "QT_QTO"
        or focused_word.endswith("തവണ")
        or focused_word.endswith("പ്രാവശ്യം")
    ):
        return "PE"

    # 2. Adjectives
    if focused_pos_upper.startswith("JJ") or focused_pos_upper == "JJ":
        return "PE"

    # 3. Check for lexical main verb V_VM_* in sentence
    has_lexical_vm = any(
        tag.startswith("V_VM") and not tag.startswith("V_VAUX")
        for tag in sentence_pos_upper
    )
    has_only_vaux = (
        any(tag == "V_VAUX" or tag.startswith("V_VAUX") for tag in sentence_pos_upper)
        and not has_lexical_vm
    )

    if not has_lexical_vm or has_only_vaux:
        return "PE"

    # 4. Supported nominal / adverbial / VP constituent + lexical V_VM_*
    is_supported_constituent = (
        focused_pos_upper.startswith("N_")       # N_NN, N_NNP, N_NST, etc.
        or focused_pos_upper.startswith("PR_")   # PR_PRP, PR_PRI, etc.
        or focused_pos_upper.startswith("DM_")   # DM_DMR, DM_DMQ
        or focused_pos_upper in ("RB", "ADV")    # Adverbs / Temporal / Manner
        or focused_pos_upper.startswith("V_")    # VP infinitive / action focus
    )

    if is_supported_constituent and has_lexical_vm:
        return "CLEFT"

    # 5. Default fallback
    return "PE"


# Backward-compatible alias
decide_cleft_or_pe = check_cleft_eligibility


class CleftResult:
    """Carries all intermediate information from the cleft generation pipeline."""

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

    def __str__(self):
        lines = [
            f"Original sentence   : {self.original_sentence}",
            f"Focused word        : {self.focus_word or '—'}",
            f"Strategy Route      : {self.route}",
            f"Constituent type    : {self.constituent_type or '—'}",
            f"Focused-word copula : {self.copula_form or '—'}",
            f"Identified main verb: {self.main_verb or '—'}",
            f"Normalized verb     : {self.normalized_verb or '—'}",
            f"Final cleft sentence: {self.cleft_sentence or '—'}",
            f"Status              : {self.status}",
        ]
        if self.pe_sentence:
            lines.append(f"PE Sentence         : {self.pe_sentence}")
        if self.phase:
            lines.append(f"Pipeline phase      : {self.phase}")
        if self.copula_path:
            lines.append(f"Copula path         : {self.copula_path}")
        if self.error:
            lines.append(f"Note / Reason       : {self.error}")
        return "\n".join(lines)


class CleftPipeline:
    """
    Malayalam Clefting Pipeline organized into 4 distinct phases:

      PHASE 1 — CLEFT ELIGIBILITY
        Validate whether the focused constituent can naturally be clefted.
        ALLOW: Subject, Direct object, Dative/indirect object, Location, Time,
               Manner, Instrument, Causal phrase, Definite personal pronoun.
        BLOCK: Indefinite pronoun, Adjective predicate, Unverified constituent.

      PHASE 2 — CLEFT STRUCTURE
        Determine focused constituent, background clause, word order,
        copula placement, and clausal transformations.

      PHASE 3 — VERB NORMALIZATION
        Delegate finite verb nominalization to VerbNormalizer (Past/Present, Negative).
        Unverified cases remain explicitly marked NEEDS_VERIFICATION.

      PHASE 4 — VALIDATION & ASSEMBLY
        Validate normalized forms and assemble final clefted sentence.

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
        sentence_pos_tags = [t.form_pos for t in sentence_ir.tokens]
        route = check_cleft_eligibility(focus_word, focus_token.form_pos, sentence_pos_tags)

        if route == "PE":
            pe_sentence = re.sub(r"<FF>(.*?)(?:</FF>|<FF>)", r"<PE>\1</PE>", tagged_sentence)
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=clean_sentence,
                focus_word=focus_word,
                focus_token=focus_token,
                status="PE_ROUTED",
                phase="ROUTING_SECTION_1",
                route="PE",
                pe_sentence=pe_sentence,
                cleft_sentence=pe_sentence,
            )

        # Morphological analysis of the focused word
        analyses, analysis_status = self._analysis_layer.analyze_word(focus_word)
        if analysis_status == "UNRESOLVED" or not analyses:
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=clean_sentence,
                focus_word=focus_word,
                focus_token=focus_token,
                status="UNRESOLVED",
                phase="PHASE_1_ELIGIBILITY",
                route="CLEFT",
                error=f"mlmorph could not analyze focus word '{focus_word}'.",
            )

        selected_analysis = analyses[0]

        # --------------------------------------------------------------
        # PHASE 1: CLEFT ELIGIBILITY
        # --------------------------------------------------------------
        is_eligible, constituent_type, elig_status, elig_reason = self._check_eligibility(
            focus_word, focus_token, selected_analysis, sentence_ir
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
        if main_verb == "ഉണ്ട്" or focus_word == "ഉണ്ട്":
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
                    "Sentence contains an existential/suppletive predicate (ഉണ്ട്). "
                    "Support-verb insertion and existential clefting (ഉണ്ട് → ഉള്ളത്) "
                    "is deferred to a separate LATER_PHASE."
                ),
            )

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

        # --------------------------------------------------------------
        # PHASE 2: CLEFT STRUCTURE & COPULA GENERATION
        # --------------------------------------------------------------
        if constituent_type == "TIME" and any(
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
                status="NOT_SUPPORTED",
                phase="PHASE_2_STRUCTURE",
                error=f"CopulaLayer could not attach copula to '{focus_word}' via path '{copula_path}'.",
            )

        # Check if the matrix predicate is copular (ആണ്-bearing predicate e.g. അവൻ രാമനാണ് -> അവനാണ് രാമൻ)
        is_copular_pred = bool(main_verb and self._is_copular_predicate(main_verb))

        # --------------------------------------------------------------
        # PHASE 3: VERB NORMALIZATION
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
                    copula_form=copula_form,
                    copula_path=copula_path,
                    main_verb=main_verb,
                    status="NEEDS_VERIFICATION",
                    phase="PHASE_3_NORMALIZATION",
                    error=f"Matrix verb normalization requires verification: {norm_res.get('failure_reason', '')}",
                )

            if norm_status != "VALID" or not norm_res.get("normalized"):
                return CleftResult(
                    original_sentence=tagged_sentence,
                    clean_sentence=clean_sentence,
                    focus_word=focus_word,
                    focus_token=focus_token,
                    constituent_type=constituent_type,
                    copula_form=copula_form,
                    copula_path=copula_path,
                    main_verb=main_verb,
                    status="UNRESOLVED",
                    phase="PHASE_3_NORMALIZATION",
                    error=f"Matrix verb normalization failed: {norm_res.get('failure_reason', '')}",
                )

            normalized_verb = norm_res["normalized"]
        else:
            normalized_verb = ""

        # --------------------------------------------------------------
        # PHASE 4: VALIDATION & SENTENCE ASSEMBLY
        # --------------------------------------------------------------
        cleft_sentence = self._substitute(clean_sentence, focus_word, copula_form)

        if main_verb and normalized_verb is not None and main_verb != focus_word and main_verb != copula_form:
            cleft_sentence = self._substitute(cleft_sentence, main_verb, normalized_verb)

        # Normalize whitespace and punctuation
        cleft_sentence = re.sub(r"\s+([.,!?;:])", r"\1", cleft_sentence)
        cleft_sentence = re.sub(r"\s+", " ", cleft_sentence).strip()

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
        """
        Determine from morphological/lexical features whether
        the token is an interrogative or indefinite pronoun.
        """
        if not analysis:
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
        if (
            any(focus_word.endswith(sfx) for sfx in MANNER_SUFFIXES)
            or form_pos == "RB"
            or getattr(analysis, "pos", "") in ("ADV", "adv")
            or focus_word.endswith("ത്തിൽ")
        ):
            return "MANNER"

        # 3. Cause semantics
        if focus_word in ("അതുകൊണ്ട്", "അതുകൊണ്ടു്") or focus_word.endswith("കാരണം") or focus_word.endswith("മൂലം") or case == "sociative":
            return "CAUSE"

        # 4. Location semantics (distinguished spatial locatives)
        if case == "locative" or focus_word.endswith(("ിൽ", "ൽ", "ങ്കൽ", "ത്ത്")):
            return "LOCATION"

        return ""

    def _check_eligibility(self, focus_word: str, focus_token, analysis, sentence_ir=None):
        """
        Determine whether the focused constituent is eligible for Malayalam clefting
        following the priority order:
        1. Hard grammatical blockers
        2. Dependency/syntactic role
        3. Morphological case
        4. Semantic interpretation of ambiguous adjuncts
        5. Personal-pronoun features
        6. Safe fallback: NEEDS_VERIFICATION
        """
        semantic_tag = self._extract_semantic_tag(focus_word, focus_token, analysis)
        dep_relation = self._extract_dependency_relation(focus_token, sentence_ir, analysis, semantic_tag)

        # ---------------------------------------------------------
        # TEMPORAL ADJUNCT CHECK
        # ---------------------------------------------------------
        # A verbal form can function as a temporal adjunct.
        # Example:
        #   വന്നതിനുശേഷം
        #   ആരംഭിക്കുന്നതിനുമുമ്പ്
        #
        # These must be treated as TIME, not VP_ACTION.
        if semantic_tag == "TEMPORAL":
            return True, "TIME", "ALLOWED", ""

        # ---------------------------------------------------------
        # ACTION / VP FOCUS
        # ---------------------------------------------------------
        form_pos = getattr(focus_token, "form_pos", "") if focus_token else ""
        if (
            focus_word.endswith("ുക")
            or focus_word.endswith("ക്ക")
            or form_pos.startswith("V_")
            or getattr(analysis, "pos", "") in ("VERB", "verb")
        ):
            return True, "VP_ACTION", "ALLOWED", ""

        # ---------------------------------------------------------
        # 1. HARD BLOCKERS
        # ---------------------------------------------------------
        if self._is_indefinite_or_interrogative_pronoun(analysis, focus_token):
            return False, "INDEFINITE_PRONOUN", "BLOCKED", f"Indefinite/interrogative pronoun '{focus_word}' cannot naturally be cleft-focused."

        if self._is_predicate_adjective(analysis, dep_relation, focus_token):
            return False, "ADJECTIVE_PREDICATE", "BLOCKED", f"Adjective predicate '{focus_word}' cannot be cleft-focused directly without nominalization."

        # ---------------------------------------------------------
        # 2. SYNTACTIC CORE ARGUMENTS
        # ---------------------------------------------------------
        if dep_relation == "nsubj":
            if self._is_definite_personal_pronoun(analysis, focus_token):
                return True, "DEFINITE_PRONOUN", "ALLOWED", ""
            return True, "SUBJECT", "ALLOWED", ""

        if dep_relation in ("obj", "dobj"):
            return True, "DIRECT_OBJECT", "ALLOWED", ""

        if dep_relation in ("iobj", "dative"):
            return True, "DATIVE_OBJECT", "ALLOWED", ""

        # ---------------------------------------------------------
        # 3. MORPHOLOGICAL CASE
        # ---------------------------------------------------------
        case = self._get_case(analysis)

        if case == "accusative":
            return True, "DIRECT_OBJECT", "ALLOWED", ""

        if case == "dative":
            return True, "DATIVE_OBJECT", "ALLOWED", ""

        # ---------------------------------------------------------
        # 4. SEMANTICALLY AMBIGUOUS ADJUNCTS
        # ---------------------------------------------------------
        if semantic_tag == "TEMPORAL":
            return True, "TIME", "ALLOWED", ""

        if semantic_tag == "MANNER":
            return True, "MANNER", "ALLOWED", ""

        if semantic_tag == "CAUSE":
            return True, "CAUSAL_PHRASE", "ALLOWED", ""

        if case == "instrumental" and semantic_tag != "CAUSE":
            return True, "INSTRUMENT", "ALLOWED", ""

        if case == "locative":
            if semantic_tag == "LOCATION":
                return True, "LOCATION", "ALLOWED", ""
            return False, "AMBIGUOUS_LOCATIVE", "NEEDS_VERIFICATION", f"Locative constituent '{focus_word}' is semantically ambiguous without location/manner/time context."

        # ---------------------------------------------------------
        # 5. DEFINITE PERSONAL PRONOUN
        # ---------------------------------------------------------
        if self._is_definite_personal_pronoun(analysis, focus_token):
            return True, "DEFINITE_PRONOUN", "ALLOWED", ""

        # ---------------------------------------------------------
        # 6. SAFE FALLBACK
        # ---------------------------------------------------------
        return False, "UNVERIFIED_CONSTITUENT", "NEEDS_VERIFICATION", f"Constituent '{focus_word}' could not be verified by morphology or syntax."

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_focus_token(self, sentence_ir, focus_word: str):
        for token in sentence_ir.tokens:
            if token.form == focus_word:
                return token
        return None

    def _substitute(self, sentence: str, target: str, replacement: str) -> str:
        if target in sentence:
            return sentence.replace(target, replacement, 1)
        return sentence

    # ------------------------------------------------------------------
    # Copular predicate handling (Clause-level)
    # ------------------------------------------------------------------

    def _is_copular_predicate(self, word: str) -> bool:
        """Check if a word is a copular predicate (contains ആണ്)."""
        if word == "ആണ്" or word.endswith("ാണ്") or word.endswith("ആണ്") or word.endswith("യാണ്"):
            return True
        analyses = self._analysis_layer.analyser.analyse(word)
        for raw, _ in analyses:
            if "ആണ്<aff>" in raw:
                return True
        return False

    def _decopularize(self, word: str) -> str:
        """
        Strip the copular suffix ആണ് from a copular predicate to recover
        the base non-focused constituent for cleft construction.

        e.g. രാമനാണ് → രാമൻ, വീടാണ് → വീട്, മരമാണ് → മരം
        """
        if word == "ആണ്":
            return ""

        # Strategy 1: mlmorph analysis → generation
        analyses = self._analysis_layer.analyser.analyse(word)
        for raw, _ in analyses:
            if "ആണ്<aff>" in raw:
                idx = raw.find("ആണ്<aff>")
                stripped_target = raw[:idx]
                gen_results = self._copula_layer.generator.generate(stripped_target)
                if gen_results:
                    return gen_results[0][0]

        # Strategy 2: Morphophonemic fallback (sandhi restoration)
        if word.endswith("ക്കാണ്"):
            return word[:-5] + "ക്ക്"
        if word.endswith("യിലാണ്"):
            return word[:-5] + "ിൽ"
        if word.endswith("റ്റിലാണ്"):
            return word[:-6] + "റ്റിൽ"
        if word.endswith("ലിലാണ്"):
            return word[:-6] + "ലിൽ"
        if word.endswith("ലാണ്"):
            return word[:-4] + "ൽ"
        if word.endswith("നാണ്"):
            return word[:-4] + "ൻ"
        if word.endswith("ളാണ്"):
            return word[:-4] + "ൾ"
        if word.endswith("രാണ്"):
            return word[:-4] + "ർ"
        if word.endswith("മാണ്"):
            return word[:-4] + "ം"
        if word.endswith("വാണ്"):
            return word[:-4] + "വ്"
        if word.endswith("ടാണ്"):
            return word[:-4] + "ട്"
        if word.endswith("താണ്"):
            return word[:-4] + "ത്"
        if word.endswith("റ്റാണ്"):
            return word[:-4] + "റ്റ്"
        if word.endswith("പ്പാണ്"):
            return word[:-4] + "പ്പ്"
        if word.endswith("ത്താണ്"):
            return word[:-4] + "ത്ത്"
        if word.endswith("യാണ്"):
            return word[:-4]
        if word.endswith("ആണ്"):
            return word[:-3]

        return word
