import logging
import string
from mlmorph import Analyser, Generator

logger = logging.getLogger(__name__)

# Standard Malayalam case suffixes mapped to their case names
CASE_SUFFIXES = {
    "ablative": ["ഇൽനിന്ന്", "ൽനിന്ന്", "നിന്ന്", "നിന്നു"],
    "instrumental": ["കൊണ്ട്", "ആൽ", "യാൽ"],
    "genitive": ["ന്റെ", "ഉടെ", "നുടെ"],
    "locative": ["ഇൽ", "ൽ", "റ്റിൽ", "ങ്കൽ"],
    "dative": ["ന്", "ക്ക്", "അ്", "ക്"],
    "accusative": ["എ", "നെ", "യെ"],
    "sociative": ["ഓട്", "യോട്"]
}

# Pronoun person resolution mapping
PRONOUN_PERSONS = {
    "ഞാൻ": "first", "ഞങ്ങൾ": "first", "നാം": "first",
    "നീ": "second", "നിങ്ങൾ": "second",
    "അവൻ": "third", "അവൾ": "third", "അവർ": "third", "അത്": "third"
}


def attach_aanu_surface(word: str) -> str:
    """
    Direct deterministic surface phonological attachment of 'ആണ്' to a single Malayalam word.
    Used when FST generation does not have an explicit dictionary entry for the single word.
    
    Phonological rules:
    - Anusvara (-ം)           → -മാണ് (e.g. പുസ്തകം → പുസ്തകമാണ്)
    - Alveolar chillu (-ൻ)     → -നാണ് (e.g. അവൻ → അവനാണ്)
    - Lateral chillu (-ൽ)      → -ലാണ് (e.g. വീട്ടിൽ → വീട്ടിലാണ്)
    - Retroflex chillu (-ൾ)    → -ളാണ് (e.g. അവൾ → അവളാണ്)
    - Rhotic chillu (-ർ)       → -രാണ് (e.g. അവർ → അവരാണ്)
    - Retroflex nasal chillu (-ൺ) → -ണാണ് (e.g. കൺ → കണാണ്)
    - Velar chillu (-ൿ)        → -കാണ്
    - Labial / rounded vowels (-ു, -ൂ, -ൊ, -ോ, -വ്) → -വാണ് (e.g. ഗുരു → ഗുരുവാണ്, കാവ് → കാവാണ്)
    - Palatal vowels & all others (-ി, -ീ, -െ, -േ, etc.) → -യാണ് (e.g. കുട്ടി → കുട്ടിയാണ്, നാളെ → നാളെയാണ്)
    """
    if not word:
        return "ആണ്"
    if word.endswith("ആണ്") or word.endswith("ാണ്"):
        return word
    if word.endswith("ം"):
        return word[:-1] + "മാണ്"
    if word.endswith("ൻ"):
        return word[:-1] + "നാണ്"
    if word.endswith("ൽ"):
        return word[:-1] + "ലാണ്"
    if word.endswith("ൾ"):
        return word[:-1] + "ളാണ്"
    if word.endswith("ർ"):
        return word[:-1] + "രാണ്"
    if word.endswith("ൺ"):
        return word[:-1] + "ണാണ്"
    if word.endswith("ൿ"):
        return word[:-1] + "കാണ്"
    if word.endswith("വ്"):
        return word[:-1] + "വാണ്"
    if any(word.endswith(v) for v in ("ു", "ൂ", "ൊ", "ോ", "ൌ")):
        return word + "വാണ്"
    return word + "യാണ്"


class MorphAnalysis:
    def __init__(self, raw_analysis, lemma, pos, case, number, person, other_features):
        self.raw_analysis = raw_analysis
        self.lemma = lemma
        self.pos = pos
        self.case = case
        self.number = number
        self.person = person
        self.other_features = other_features

    def key(self):
        return (self.lemma, self.pos, self.case, self.number, self.person)

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
            return MorphAnalysis(raw_analysis, raw_analysis, "UNK", "nominative", "singular", None, "")

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

        # Determine Person
        person = None
        if "first" in tags:
            person = "first"
        elif "second" in tags:
            person = "second"
        elif "third" in tags:
            person = "third"
        elif pos == "PRON" and lemma in PRONOUN_PERSONS:
            person = PRONOUN_PERSONS[lemma]

        # Other Features
        skip_tags = {raw_pos, case, "pl", "first", "second", "third"}
        other_features = "|".join([t for t in tags if t not in skip_tags])

        return MorphAnalysis(raw_analysis, lemma, pos, case, number, person, other_features)

    def analyze_word(self, word):
        """
        Runs mlmorph analysis on a single word, resolves ambiguity, and returns the selected analysis
        along with status ('VALID', 'AMBIGUOUS', 'UNRESOLVED').
        """
        if self._is_punctuation(word):
            analysis = MorphAnalysis(word, word, "PUNC", None, None, None, "")
            return [analysis], "VALID"

        raw_analyses = self.analyser.analyse(word)
        if not raw_analyses:
            # For words not in FST dictionary, create a base MorphAnalysis
            synthetic = MorphAnalysis(word, word, "NOUN", "nominative", "singular", None, "")
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
        if pattern.endswith("്"):
            pattern = pattern[:-1]
        elif pattern.endswith("ൽ"):
            pattern = pattern[:-1] + "ല"
        elif pattern.endswith("ൻ"):
            pattern = pattern[:-1] + "ന"
        elif pattern.endswith("ർ"):
            pattern = pattern[:-1] + "ര"
            
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
        Preserves existing case/morphological realization of the input word.
        Does NOT synthesize Dvitva, adjective nominalizations, or participle nominalizations.

        Returns:
            copula_form (str)
            case_preserved (str): 'YES', 'NO', or 'UNKNOWN'
            copula_path (str): 'already-copular', 'case-preserving', 'normal'
        """
        raw_analysis = selected_analysis.raw_analysis if selected_analysis else ""

        # 1. Already copular check
        if "ആണ്" in word or (raw_analysis and "<aff>" in raw_analysis):
            return word, "YES", "already-copular"

        # 2. Case-preserving copula generation
        css = self.detect_css(word, selected_analysis) if selected_analysis else ""
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
                if filtered:
                    return filtered[0], "YES", "case-preserving"
                return generated_forms[0], "YES", "case-preserving"
            
            # Direct surface fallback preserving case
            surface_form = attach_aanu_surface(word)
            return surface_form, "YES", "case-preserving"

        # 3. Normal direct copula generation for non-case words
        if raw_analysis and "<" in raw_analysis:
            target_analysis = f"{raw_analysis}ആണ്<aff>"
            results = self.generator.generate(target_analysis)
            if results:
                return results[0][0], "YES", "normal"

        # Direct surface fallback
        surface_form = attach_aanu_surface(word)
        return surface_form, "YES", "normal"
