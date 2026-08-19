import re
import logging

from .malayalam_pipeline import MalayalamPipeline
from .copula_pipeline import AnalysisLayer, CopulaLayer

logger = logging.getLogger(__name__)

FF_TAG = "<FF>"
# Format: <FF>word<FF>  (symmetric — same tag used as opener and closer)
FF_PATTERN = re.compile(r"<FF>(.*?)<FF>", re.UNICODE)


class CleftResult:
    """Carries all intermediate information from the cleft generation step."""

    def __init__(
        self,
        original_sentence: str,
        clean_sentence: str,
        focus_word: str,
        focus_token=None,       # ir.Token from MalayalamPipeline
        copula_form: str = "",
        copula_path: str = "",
        main_verb: str = "",
        normalized_verb: str = "",
        status: str = "UNRESOLVED",
        cleft_sentence: str = "",
        error: str = "",
    ):
        self.original_sentence = original_sentence
        self.clean_sentence = clean_sentence
        self.focus_word = focus_word
        self.focus_token = focus_token
        self.copula_form = copula_form
        self.copula_path = copula_path
        self.main_verb = main_verb
        self.normalized_verb = normalized_verb
        self.status = status
        self.cleft_sentence = cleft_sentence
        self.error = error

    def __str__(self):
        lines = [
            f"Original sentence   : {self.original_sentence}",
            f"Focused word        : {self.focus_word or '—'}",
            f"Focused-word copula : {self.copula_form or '—'}",
            f"Identified main verb: {self.main_verb or '—'}",
            f"Normalized verb     : {self.normalized_verb or '—'}",
            f"Final cleft sentence: {self.cleft_sentence or '—'}",
            f"Status              : {self.status}",
        ]
        if self.copula_path:
            lines.append(f"Copula path         : {self.copula_path}")
        if self.error:
            lines.append(f"Error               : {self.error}")
        return "\n".join(lines)


class CleftPipeline:
    """
    Connects MalayalamPipeline (sentence-level IR + VerbNormalizer) with CopulaLayer
    (word-level copula generation) to produce end-to-end cleft sentences.

    Usage:
        pipeline = CleftPipeline()
        result = pipeline.process("രാമൻ <FF>പുസ്തകം<FF> വായിച്ചു.")
        print(result)
    """

    def __init__(self):
        self._malayalam_pipeline = MalayalamPipeline()
        self._analysis_layer    = AnalysisLayer()
        self._copula_layer      = CopulaLayer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, tagged_sentence: str) -> CleftResult:
        """
        Process a tagged sentence and return a CleftResult.

        Flow:
          1. Detect <FF>...</FF> and extract the focused word/constituent.
          2. Strip markers to obtain the clean surface sentence.
          3. Run MalayalamPipeline on the clean sentence to:
             - tokenize
             - obtain POS and morphological analyses
             - identify the main finite verb
             - normalize the main verb using VerbNormalizer
          4. Locate the focused token in the IR.
          5. Send focused word to CopulaPipeline (AnalysisLayer + CopulaLayer)
             to generate the focused word's copular form.
          6. Construct the final clefted sentence by:
             - replacing the focused constituent with its copular form
             - replacing the main finite verb with its normalized form
        """
        tagged_sentence = tagged_sentence.strip()

        # Step 1: Validate and extract focus span
        matches = FF_PATTERN.findall(tagged_sentence)

        if len(matches) == 0:
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=tagged_sentence,
                focus_word="",
                status="ERROR",
                error="No <FF>...<FF> focus marker found in the sentence. "
                      "Expected format: <FF>word<FF>",
            )

        if len(matches) > 1:
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=re.sub(r"<FF>", "", tagged_sentence).strip(),
                focus_word=", ".join(matches),
                status="ERROR",
                error=f"Multiple focus markers found: {matches}. Expected exactly one.",
            )

        focus_word = matches[0].strip()
        if not focus_word:
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=re.sub(r"<FF>", "", tagged_sentence).strip(),
                focus_word="",
                status="ERROR",
                error="Empty <FF><FF> focus marker.",
            )

        # Step 2: Strip markers → clean sentence
        clean_sentence = tagged_sentence.replace(FF_TAG + focus_word + FF_TAG, focus_word).strip()
        clean_sentence = re.sub(r"<FF>", "", clean_sentence).strip()

        # Step 3: Run full sentence IR pipeline (tokenization, POS, morph, main-verb identification & normalization)
        sentence_ir = self._malayalam_pipeline.process(clean_sentence)
        main_verb = sentence_ir.main_verb or ""
        normalized_verb = sentence_ir.normalized_verb or main_verb

        # Step 4: Find the focused token in the IR
        focus_token = self._find_focus_token(sentence_ir, focus_word)
        if focus_token is None:
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=clean_sentence,
                focus_word=focus_word,
                main_verb=main_verb,
                normalized_verb=normalized_verb,
                status="ERROR",
                error=(
                    f"Focus word '{focus_word}' not found in the tokenized IR. "
                    f"Tokens: {[t.form for t in sentence_ir.tokens]}"
                ),
            )

        # Step 5: Run copula pipeline on the focused surface form
        analyses, analysis_status = self._analysis_layer.analyze_word(focus_word)

        if analysis_status == "UNRESOLVED" or not analyses:
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=clean_sentence,
                focus_word=focus_word,
                focus_token=focus_token,
                main_verb=main_verb,
                normalized_verb=normalized_verb,
                status="UNRESOLVED",
                copula_path="not-supported",
                error=f"mlmorph could not analyze focus word '{focus_word}' even with fallback.",
            )

        selected = analyses[0]
        copula_form, _preserved, copula_path = self._copula_layer.transform_word(
            focus_word, selected
        )

        if copula_path in ("not-supported",) or copula_form in ("NOT SUPPORTED", "ALREADY COPULAR"):
            return CleftResult(
                original_sentence=tagged_sentence,
                clean_sentence=clean_sentence,
                focus_word=focus_word,
                focus_token=focus_token,
                copula_form=copula_form,
                copula_path=copula_path,
                main_verb=main_verb,
                normalized_verb=normalized_verb,
                status=analysis_status if analysis_status != "VALID" else "NOT_SUPPORTED",
                error=f"CopulaLayer returned '{copula_form}' via path '{copula_path}'.",
            )

        # Step 6: Construct clefted sentence
        # 6a. Replace focus_word with copula_form
        cleft_sentence = self._substitute(clean_sentence, focus_word, copula_form)
        
        # 6b. If main_verb is present and distinct from focus_word, replace main_verb with normalized_verb
        if main_verb and normalized_verb and main_verb != focus_word and main_verb != copula_form:
            cleft_sentence = self._substitute(cleft_sentence, main_verb, normalized_verb)

        return CleftResult(
            original_sentence=tagged_sentence,
            clean_sentence=clean_sentence,
            focus_word=focus_word,
            focus_token=focus_token,
            copula_form=copula_form,
            copula_path=copula_path,
            main_verb=main_verb,
            normalized_verb=normalized_verb,
            status=analysis_status,   # VALID or AMBIGUOUS
            cleft_sentence=cleft_sentence,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_focus_token(self, sentence_ir, focus_word: str):
        """
        Finds the first IR Token whose surface form matches the focus word.
        Returns None if not found.
        """
        for token in sentence_ir.tokens:
            if token.form == focus_word:
                return token
        return None

    def _substitute(self, sentence: str, target: str, replacement: str) -> str:
        """
        Replaces the first occurrence of target with replacement in the sentence.
        """
        if target in sentence:
            return sentence.replace(target, replacement, 1)
        return sentence
