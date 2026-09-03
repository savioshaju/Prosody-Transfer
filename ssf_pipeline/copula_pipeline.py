import os
import sys
import logging
import string

_venv_site = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "venv", "Lib", "site-packages")
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.append(_venv_site)

from mlmorph import Analyser, Generator

logger = logging.getLogger(__name__)

# Active Malayalam case suffixes used by detect_css() to preserve case during copula attachment
# Extended with all allomorphs from the SSF manual (both atomic chillu and virama encodings)
CASE_SUFFIXES = {
    "ablative": ["ഇൽനിന്ന്", "ൽനിന്ന്", "ഇല്നിന്ന്", "ല്നിന്ന്", "നിന്ന്", "നിന്നു"],
    "instrumental": ["കൊണ്ട്", "ത്തിനാൽ", "ഇനാൽ", "യാൽ", "ആൽ", "ത്തിനാല്", "ഇനാല്", "യാല്", "ആല്"],
    "genitive": ["ഇന്റെ", "ന്റെ", "്റെ", "യുടെ", "ുടെ", "ിന്റെ", "ിന്റെ"],
    "locative": ["ത്തിൽ", "തിൽ", "വിൽ", "യിൽ", "റ്റിൽ", "ില്", "തിൽ", "ത്തിൽ", "ത്ത്", "ൽ", "ല്", "യിൽ", "വിൽ", "റ്റിൽ", "ില്", "തില്", "ത്തില്", "വില്", "യില്", "റ്റില്", "ിൽ"],
    "dative": ["യ്ക്ക്", "ക്ക്", "ഇന്", "ന്", "ഇൻ", "ൻ"],
    "accusative": ["യെ", "നെ", "െ"],
    "sociative": ["ത്തോട്", "ഇനോട്", "വിനോട്", "യോട്", "ഓട്"]
}

PUNCTUATION_CHARS = set(".,!?;:\"'“”‘’()[]{}—–-«»/\\")


def split_punctuation(token: str) -> tuple:
    """
    Splits a token into (leading_punct, core_linguistic_word, trailing_punct).
    Treats structural punctuation (commas, colons, quotes, brackets, etc.)
    separately from the linguistic stem, ensuring morphology applies to the stem
    and restores punctuation outside the affixed form.

    Examples:
      - 'പച്ചമത്സ്യം,'  -> ('', 'പച്ചമത്സ്യം', ',')
      - 'ഉയരങ്ങളിൽ:'   -> ('', 'ഉയരങ്ങളിൽ', ':')
      - '“രൂപത്തിൽ”'   -> ('“', 'രൂപത്തിൽ', '”')
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


def _attach_aanu_surface_core(word: str) -> str:
    """
    Direct deterministic surface phonological attachment of 'ആണ്' to a clean single Malayalam word stem.
    """
    if not word:
        return "ആണ്"
    # Already copular — passthrough
    if word.endswith("ആണ്") or word.endswith("ാണ്"):
        return word

    # ---------------------------------------------------------------
    # Disjunctive quote / clause nominalization: -എന്നോ, -മെന്നോ → -എന്നതിനെയാണ്
    # ---------------------------------------------------------------
    if word.endswith("എന്നോ"):
        return word[:-len("എന്നോ")] + "എന്നതിനെയാണ്"
    if word.endswith("മെന്നോ"):
        return word[:-len("മെന്നോ")] + "മെന്നതിനെയാണ്"

    # ---------------------------------------------------------------
    # 3. Chillu Consonants (Vyañjana-Svara Sandhi)
    #    Anusvāra: ം → മാണ്  (e.g. സത്യം → സത്യമാണ്, കാടുകൾ → കാടുകളാണ്)
    # ---------------------------------------------------------------
    if word.endswith("ുന്നു"):
        return word[:-len("ുന്നു")] + "ുന്നതാണ്"

    # ---------------------------------------------------------------
    # 2b. ആദേശം — Anusvāra Resolution: ം → മാണ്
    # ---------------------------------------------------------------
    if word.endswith("ം"):
        return word[:-1] + "മാണ്"

    # ---------------------------------------------------------------
    # 3. വ്യഞ്ജന–സ്വര സന്ധി — Chillu / Virama Consonant–Vowel Joining
    # ---------------------------------------------------------------
    if word.endswith("ൽ") or word.endswith("ല്"):
        stem = word[:-1] if word.endswith("ൽ") else word[:-2]
        return stem + "ലാണ്"
    if word.endswith("ൻ") or word.endswith("ന്"):
        stem = word[:-1] if word.endswith("ൻ") else word[:-2]
        return stem + "നാണ്"
    if word.endswith("ൾ") or word.endswith("ള്"):
        stem = word[:-1] if word.endswith("ൾ") else word[:-2]
        return stem + "ളാണ്"
    if word.endswith("ർ") or word.endswith("ര്"):
        stem = word[:-1] if word.endswith("ർ") else word[:-2]
        return stem + "രാണ്"
    if word.endswith("ൺ") or word.endswith("ണ്"):
        stem = word[:-1] if word.endswith("ൺ") else word[:-2]
        return stem + "ണാണ്"
    if word.endswith("ൿ"):
        return word[:-1] + "കാണ്"

    # ---------------------------------------------------------------
    # 4. Virāma / Chandrakkala Deletion: C് + ആ → Cാണ്
    #    The final chandrakkala ് is removed and the vowel-sign ാ attaches.
    #    Ordered: specific consonant clusters first, then generic.
    # ---------------------------------------------------------------
    # Specific geminate/cluster patterns (most common in case suffixes)
    _VIRAMA_CLUSTERS = (
        ("ത്ത്", "ത്താണ്"),   # locative ത്ത്  → ത്താണ്  (e.g. നിലത്ത്→നിലത്താണ്)
        ("ട്ട്", "ട്ടാണ്"),   # e.g. കൊട്ട്→കൊട്ടാണ്
        ("ക്ക്", "ക്കാണ്"),   # dative ക്ക്   → ക്കാണ്  (e.g. അവനുക്ക്→...ക്കാണ്)
        ("പ്പ്", "പ്പാണ്"),   # e.g. ചെപ്പ്→ചെപ്പാണ്
        ("ന്ന്", "ന്നാണ്"),   # ablative നിന്ന് → നിന്നാണ്
        ("ണ്ട്", "ണ്ടാണ്"),   # e.g. കൊണ്ട്→കൊണ്ടാണ്  (instrumental)
        ("ച്ച്", "ച്ചാണ്"),   # e.g. പുച്ച്→പുച്ചാണ്
        ("ല്ല്", "ല്ലാണ്"),
        ("ള്ള്", "ള്ളാണ്"),
    )
    for suffix, replacement in _VIRAMA_CLUSTERS:
        if word.endswith(suffix):
            return word[:-len(suffix)] + replacement

    # Generic Virāma deletion: any consonant + ് → consonant + ാണ്
    if word.endswith("്"):
        return word[:-1] + "ാണ്"

    # ---------------------------------------------------------------
    # Predicate complement suffix -ായി → -ായാണ്
    # (e.g. വ്യക്തിയായി → വ്യക്തിയായാണ്, ആയി → ആയാണ്)
    # Must precede Yakāra-āgamam to avoid producing *-ായിയാണ്.
    # ---------------------------------------------------------------
    if word.endswith("ായി"):
        return word[:-1] + "ാണ്"

    # ---------------------------------------------------------------
    # 1. ആഗമം — Vowel-based Insertion
    # ---------------------------------------------------------------
    # 1b. വകാരാഗമം (Vakāra-āgamam): rounded/back vowels → വ insertion
    #     ഉ, ഊ, ഒ, ഓ, ഔ (+ vowel signs ു, ൂ, ൊ, ോ, ൌ, ൗ) + ആണ് → + വാണ്
    if any(word.endswith(v) for v in ("ു", "ൂ", "ൊ", "ോ", "ൌ", "ൗ", "ഉ", "ഊ", "ഒ", "ഓ", "ഔ")):
        return word + "വാണ്"

    # 1a. യകാരാഗമം (Yakāra-āgamam): front/central vowels → യ insertion
    #     അ, ആ, ഇ, ഈ, എ, ഏ, ഐ (+ vowel signs ാ, ി, ീ, െ, േ, ൈ) + ആണ് → + യാണ്
    #     This is the default for all remaining vowel-final words.
    return word + "യാണ്"


def attach_aanu_surface(word: str) -> str:
    """
    Direct deterministic surface phonological attachment of 'ആണ്' to a single Malayalam word,
    correctly preserving leading and trailing punctuation (e.g. 'ഉയരങ്ങളിൽ,' -> 'ഉയരങ്ങളിലാണ്,').
    """
    if not word:
        return "ആണ്"
    leading, core, trailing = split_punctuation(word)
    if not core:
        return word + "ആണ്"
    if core.endswith("ആണ്") or core.endswith("ാണ്"):
        return word
    copular_core = _attach_aanu_surface_core(core)
    return f"{leading}{copular_core}{trailing}"


def is_valid_sandhi(word: str, generated: str) -> bool:
    """
    Validates if a generated copula form obeys Malayalam Sandhi rules.
    Flags invalid attachments like 'ഐാണ്' or 'ഇാണ്' where a vowel-final stem
    directly appends 'ാണ്' without inserting 'യ' (Yakāra-āgamam) or 'വ' (Vakāra-āgamam).
    """
    if not generated or not word:
        return True
    
    clean_w = word.rstrip(".,!?;:\"'”’")
    clean_g = generated.rstrip(".,!?;:\"'”’")
    
    _ALL_VOWELS = (
        "അ", "ആ", "ഇ", "ഈ", "എ", "ഏ", "ഐ", "ഉ", "ഊ", "ഒ", "ഓ", "ഔ",
        "ാ", "ി", "ീ", "െ", "േ", "ൈ", "ു", "ൂ", "ൊ", "ോ", "ൌ", "ൗ"
    )
    for v in _ALL_VOWELS:
        if clean_w.endswith(v) and clean_g.endswith(v + "ാണ്"):
            return False
            
    return True


class MorphAnalysis:
    def __init__(self, raw_analysis, lemma, pos, case, number, other_features):
        self.raw_analysis = raw_analysis
        self.lemma = lemma
        self.pos = pos
        self.case = case
        self.number = number
        self.other_features = other_features

    def key(self):
        return (self.lemma, self.pos, self.case, self.number)

    def __str__(self):
        return self.raw_analysis


class AnalysisLayer:
    def __init__(self):
        self.analyser = Analyser()

    def _is_punctuation(self, word):
        extra_punc = "\u201c\u201d\u2018\u2019\u2014\u2013"
        return all(c in string.punctuation or c in extra_punc for c in word)

    def parse_raw_analysis(self, raw_analysis):
        """
        Parses a raw mlmorph analysis string like 'രാമൻ<n><genitive>' into a structured MorphAnalysis.
        """
        import re
        if "<" not in raw_analysis:
            return MorphAnalysis(raw_analysis, raw_analysis, "UNK", "nominative", "singular", "")

        # Extract all tags inside <...>
        tags = re.findall(r"<([^>]+)>", raw_analysis)
        
        # Extract all stems (non-tag parts)
        stems = [s for s in re.split(r"<[^>]+>", raw_analysis) if s]
        lemma = stems[-1] if stems else raw_analysis.split("<")[0]

        raw_pos = tags[0] if tags else "UNK"
        
        # Simplify POS category
        pos_map = {
            "n": "NOUN", "np": "NOUN", "sanskrit": "NOUN", "eng": "NOUN",
            "v": "VERB", "adj": "ADJ", "adv": "ADV",
            "prn": "PRON", "dem": "DEM", "quantifier": "QUANT", "postp": "POSTP"
        }
        pos = pos_map.get(raw_pos, raw_pos.upper())

        # Determine Case
        case = None
        for tag in tags:
            if tag in CASE_SUFFIXES.keys():
                case = tag
                break
        if not case:
            case = "nominative" if pos in ["NOUN", "PRON"] else None

        # Determine Number
        number = "plural" if "pl" in tags else ("singular" if pos in ["NOUN", "PRON"] else None)

        # Other Features
        skip_tags = {raw_pos, case, "pl"}
        other_features = "|".join([t for t in tags if t not in skip_tags])

        return MorphAnalysis(raw_analysis, lemma, pos, case, number, other_features)

    def analyze_word(self, word):
        """
        Runs mlmorph analysis on a single word, resolves ambiguity, and returns the selected analysis
        along with status ('VALID', 'AMBIGUOUS', 'UNRESOLVED').
        """
        if self._is_punctuation(word):
            analysis = MorphAnalysis(word, word, "PUNC", None, None, "")
            return [analysis], "VALID"

        raw_analyses = self.analyser.analyse(word)
        if not raw_analyses:
            # For words not in FST dictionary, create a base MorphAnalysis
            synthetic = MorphAnalysis(word, word, "NOUN", "nominative", "singular", "")
            return [synthetic], "VALID"

        # Parse all analyses
        parsed_analyses = [self.parse_raw_analysis(raw[0]) for raw in raw_analyses]

        # Priority scoring to prefer inflected forms while penalizing overly complex compound tags
        scored_analyses = []
        for analysis in parsed_analyses:
            score = 0
            if analysis.case and analysis.case != "nominative":
                score += 10
            if analysis.number == "plural":
                score += 5
                
            participle_tags = ["adv-clause-rp-past", "cvb-adv-part-absolute", "cvb-adv-part-past", "rp"]
            if any(t in analysis.raw_analysis for t in participle_tags):
                score += 5
                
            tag_count = analysis.raw_analysis.count('<')
            score -= tag_count * 0.5
            
            scored_analyses.append((score, analysis))

        # Group by unique morphological keys
        groups = {}
        for score, analysis in scored_analyses:
            k = analysis.key()
            if k not in groups or groups[k][0] < score:
                groups[k] = (score, analysis)

        if not groups:
            return [], "UNRESOLVED"

        max_score = max(item[0] for item in groups.values())
        best_groups = [item[1] for item in groups.values() if item[0] == max_score]

        if len(best_groups) == 1:
            return best_groups, "VALID"

        # Check if genuinely ambiguous (different lemma, case, or pos)
        lemmas = {g.lemma for g in best_groups}
        cases = {g.case for g in best_groups}
        poses = {g.pos for g in best_groups}

        if len(lemmas) > 1 or len(cases) > 1 or len(poses) > 1:
            return best_groups, "AMBIGUOUS"

        return [best_groups[0]], "VALID"


class CopulaLayer:
    def __init__(self):
        self.generator = Generator()

    def get_sandhi_css_candidates(self, css):
        """
        Computes possible sandhi-transformed substrings for a case suffix (CSS)
        when combined with the copula clitic 'ആണ്'.
        """
        if not css:
            return []
        
        if css.startswith("ഇ"):
            rest = css[1:]
            rest_candidates = self.get_sandhi_css_candidates(rest)
            if not rest_candidates:
                return [css, "ി"]
            return [css[0] + r for r in rest_candidates] + ["ി" + r for r in rest_candidates]

        pattern = css
        if pattern.endswith("ൽ") or pattern.endswith("ല്"):
            pattern = (pattern[:-1] if pattern.endswith("ൽ") else pattern[:-2]) + "ല"
        elif pattern.endswith("ൻ") or pattern.endswith("ന്"):
            pattern = (pattern[:-1] if pattern.endswith("ൻ") else pattern[:-2]) + "ന"
        elif pattern.endswith("ർ") or pattern.endswith("ര്"):
            pattern = (pattern[:-1] if pattern.endswith("ർ") else pattern[:-2]) + "ര"
        elif pattern.endswith("ൾ") or pattern.endswith("ള്"):
            pattern = (pattern[:-1] if pattern.endswith("ൾ") else pattern[:-2]) + "ള"
        elif pattern.endswith("ൺ") or pattern.endswith("ണ്"):
            pattern = (pattern[:-1] if pattern.endswith("ൺ") else pattern[:-2]) + "ണ"
        elif pattern.endswith("്"):
            pattern = pattern[:-1]
            
        return [pattern]

    def detect_css(self, word, selected_analysis):
        """
        Extracts original case suffix (CSS) using the selected analysis case.
        """
        case = selected_analysis.case
        if not case or case == "nominative":
            return ""

        suffixes = CASE_SUFFIXES.get(case, [])
        for suffix in suffixes:
            if word.endswith(suffix):
                return suffix

        lemma = selected_analysis.lemma
        if len(word) > len(lemma):
            return word[len(lemma):]
        return ""

    def transform_word(self, word, selected_analysis):
        """
        Applies direct morphological 'ആണ്' attachment to a single selected word.
        Preserves existing case/morphological realization of the input word and
        guarantees that structural punctuation (comma, colon, quotes) is retained
        outside the affixed form (e.g. 'ഉയരങ്ങളിൽ,' -> 'ഉയരങ്ങളിലാണ്,').

        Returns:
            copula_form (str)
            case_preserved (str): 'YES', 'NO', or 'UNKNOWN'
            copula_path (str): 'already-copular', 'case-preserving', 'normal'
        """
        if not word:
            return "ആണ്", "YES", "empty"

        leading, core, trailing = split_punctuation(word)
        if not core:
            return word + "ആണ്", "YES", "punct_only"

        raw_analysis = selected_analysis.raw_analysis if selected_analysis else ""

        # 1. Already copular check
        if "ആണ്" in core or (raw_analysis and "<aff>" in raw_analysis):
            return f"{leading}{core}{trailing}", "YES", "already-copular"

        # 1b. Predicate complement / adverbial suffix -ായി → -ായാണ്
        if core.endswith("ായി"):
            return f"{leading}{attach_aanu_surface(core)}{trailing}", "YES", "predicate-complement"

        # 1c. Focus particles: -മാത്രം → -മാത്രമാണ്, -മാത്രമേ → -മാത്രമേയുള്ളൂ
        if core.endswith("മാത്രം"):
            res = core[:-len("മാത്രം")] + "മാത്രമാണ്"
            return f"{leading}{res}{trailing}", "YES", "focus-particle"

        if core.endswith("മാത്രമേ"):
            res = core[:-len("മാത്രമേ")] + "മാത്രമേയുള്ളൂ"
            return f"{leading}{res}{trailing}", "YES", "focus-particle"

        # 1d. Finite present tense verbs: -ുന്നു + ആണ് → -ുന്നതാണ്
        #     (e.g. തുടരുന്നു → തുടരുന്നതാണ്, ചെയ്യുന്നു → ചെയ്യുന്നതാണ്)
        if core.endswith("ുന്നു"):
            res = core[:-len("ുന്നു")] + "ുന്നതാണ്"
            return f"{leading}{res}{trailing}", "YES", "verb-nominalized"

        # 2. Case-preserving copula generation
        css = self.detect_css(core, selected_analysis) if selected_analysis else ""
        if css:
            target_analysis = f"{raw_analysis}ആണ്<aff>"
            results = self.generator.generate(target_analysis)
            if results:
                generated_forms = []
                for form, weight in results:
                    if form not in generated_forms:
                        generated_forms.append(form)

                sandhi_patterns = self.get_sandhi_css_candidates(css)
                filtered = [g for g in generated_forms if any(p in g for p in sandhi_patterns)]
                if filtered and is_valid_sandhi(core, filtered[0]):
                    return f"{leading}{filtered[0]}{trailing}", "YES", "fst-case-preserving"
                if generated_forms and is_valid_sandhi(core, generated_forms[0]):
                    return f"{leading}{generated_forms[0]}{trailing}", "YES", "fst-case-preserving"

            # Direct surface fallback preserving case
            surface_form = attach_aanu_surface(core)
            return f"{leading}{surface_form}{trailing}", "YES", "fallback-case-preserving"

        # 3. Normal direct copula generation for non-case words
        if raw_analysis and "<" in raw_analysis:
            target_analysis = f"{raw_analysis}ആണ്<aff>"
            results = self.generator.generate(target_analysis)
            if results and is_valid_sandhi(core, results[0][0]):
                return f"{leading}{results[0][0]}{trailing}", "YES", "fst-normal"

        # Direct surface fallback
        surface_form = attach_aanu_surface(core)
        return f"{leading}{surface_form}{trailing}", "YES", "fallback-surface"
