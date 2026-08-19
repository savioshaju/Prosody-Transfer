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


DEGEMINATION_RULES = [
    ("ശ്ശ", "ശ"),
    ("സ്സ", "സ"),
    ("ഷ്ഷ", "ഷ"),
]

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
        self.fallback_parser = FallbackParser(self.analyser, Generator())

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
        Runs mlmorph analysis, collects all parses, resolves ambiguity, and returns the selected analysis
        along with status ('VALID', 'AMBIGUOUS', 'UNRESOLVED').

        If the primary analysis fails, the FallbackParser is invoked to attempt a decomposition-based
        analysis, returning a synthetic MorphAnalysis if one can be confidently constructed.
        """
        if self._is_punctuation(word):
            analysis = MorphAnalysis(word, word, "PUNC", None, None, None, "")
            return [analysis], "VALID"

        raw_analyses = self.analyser.analyse(word)
        if not raw_analyses:
            # Primary analysis failed — try the fallback parser
            logger.warning("AnalysisLayer: no analysis for word '%s', trying fallback parser", word)
            fallback_result = self.fallback_parser.parse(word)
            if fallback_result is not None:
                return [fallback_result], "VALID"
            logger.warning("AnalysisLayer: fallback parser also failed for word '%s'", word)
            return [], "UNRESOLVED"

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
                
            # Reward adjectival/relative participle tags to prefer them over generic nominal interpretations
            participle_tags = ["adv-clause-rp-past", "cvb-adv-part-absolute", "cvb-adv-part-past", "rp"]
            if any(t in analysis.raw_analysis for t in participle_tags):
                score += 5
                
            # Penalize tag complexity to prefer simpler analyses for base words
            tag_count = analysis.raw_analysis.count('<')
            score -= tag_count * 0.5
            
            scored_analyses.append((score, analysis))

        # Group by unique morphological keys
        groups = {}
        for score, analysis in scored_analyses:
            k = analysis.key()
            if k not in groups or groups[k][0] < score:
                groups[k] = (score, analysis)

        # Filter by maximum score
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

        # Otherwise they are structurally equivalent, just pick the first one
        return [best_groups[0]], "VALID"


class FallbackParser:
    """
    Attempts a decomposition-based morphological analysis for words that mlmorph cannot directly parse.

    Two strategies are applied in order:

    Strategy 1 — Compound Nominalization Fallback:
        Detects words of the form <compound-adjective-participle> + ത്, where the base compound
        is recognized by mlmorph with an n-v-compound analysis, but the FST cannot continue
        to the nominalization state. The fallback constructs a synthetic MorphAnalysis that
        the CopulaLayer can then use to apply the adjective-nominalized copula path.

    Strategy 2 — Sibilant De-gemination Fallback:
        Detects words where mlmorph compound generation strictly geminated a sibilant (ശ → ശ്ശ,
        സ → സ്സ, ഷ → ഷ്ഷ) that the input word spells without gemination. The fallback
        re-analyzes the generated geminated form to obtain a valid MorphAnalysis, then patches
        the raw_analysis string to target the original spelling.
    """

    def __init__(self, analyser, generator):
        self.analyser = analyser
        self.generator = generator

    def _analyse(self, word):
        return self.analyser.analyse(word)

    def _generate(self, target):
        return self.generator.generate(target)

    def _first_analysis(self, word):
        results = self._analyse(word)
        return results[0][0] if results else None

    def _has_nv_compound_participle(self, raw_analysis):
        """Returns True if the raw analysis contains an n-v-compound with a participle tag."""
        participle_tags = ["adv-clause-rp-past", "cvb-adv-part-absolute", "cvb-adv-part-past", "rp"]
        return "<n-v-compound>" in raw_analysis and any(f"<{t}>" in raw_analysis for t in participle_tags)

    def _try_compound_nominalization(self, word):
        """
        Strategy 1: Handles compound adjectival participle + ത် nominalization.

        For words like: പ്രധാനപ്പെട്ടത്, സുന്ദരമായത്
        These fail because mlmorph cannot chain <n-v-compound> participles into <n><deriv>.

        Detection: strip ത്/ത from the end and check if the base compound analyses with
        n-v-compound + participle. If so, return a synthetic MorphAnalysis carrying the
        base compound's raw_analysis extended with <n><deriv>.
        """
        # Try stripping nominalizer suffix variants
        for suffix in ["ത്", "ത"]:
            if not word.endswith(suffix):
                continue
            base = word[:-len(suffix)]
            if not base:
                continue

            base_analyses = self._analyse(base)
            if not base_analyses:
                continue

            # Look for an n-v-compound + participle analysis in the base
            for raw, _ in base_analyses:
                if self._has_nv_compound_participle(raw):
                    # Build a synthetic raw_analysis for the nominalized form
                    synthetic_raw = raw + "<n><deriv>"
                    # Validate: extract the participle suffix and try generating the nominalized form
                    v_idx = raw.rfind("<v>")
                    if v_idx == -1:
                        continue
                    stem_start = raw.rfind(">", 0, v_idx)
                    stem_start = 0 if stem_start == -1 else stem_start + 1
                    participle_suffix = raw[stem_start:]
                    
                    # Verify the FST can generate the participle suffix nominalized
                    target = f"{participle_suffix}<n><deriv>"
                    gen_results = self._generate(target)
                    if not gen_results:
                        continue
                    
                    # Build and return a synthetic MorphAnalysis for the nominalized compound
                    import re
                    tags = re.findall(r"<([^>]+)>", raw)
                    pos_map = {
                        "n": "NOUN", "np": "NOUN", "sanskrit": "NOUN",
                        "v": "VERB", "adj": "ADJ"
                    }
                    raw_pos = tags[0] if tags else "n"
                    pos = pos_map.get(raw_pos, "NOUN")
                    stems = [s for s in re.split(r"<[^>]+>", raw) if s]
                    lemma = stems[-1] if stems else base
                    
                    logger.info("FallbackParser: compound nominalization fallback for '%s' via base '%s'", word, base)
                    return MorphAnalysis(
                        raw_analysis=synthetic_raw,
                        lemma=lemma,
                        pos=pos,
                        case="nominative",
                        number="singular",
                        person=None,
                        other_features="|".join(t for t in tags if t not in {raw_pos, "nominative", "singular"})
                    )
        return None

    def _try_degemination(self, word):
        """
        Strategy 2: Handles sibilant de-gemination mismatches in mlmorph compound generation.

        For words like: ഗവേഷണശാല (mlmorph generates ഗവേഷണശ്ശാല),
                         ഗവേഷണശാലയിൽ (mlmorph generates ഗവേഷണശ്ശാലയിൽ)

        The FST strictly applies gemination rules in sandhi. If the input word differs from the
        generated compound only in a known sibilant gemination pattern, we accept the compound
        analysis but mark the match as de-geminated.

        We detect case suffixes in the word to handle inflected forms of compound nouns.
        """
        # Try to detect a case suffix on the word
        case_suffix = None
        case_name = None
        for cname, suffixes in CASE_SUFFIXES.items():
            for suf in suffixes:
                if word.endswith(suf) and len(word) > len(suf):
                    case_suffix = suf
                    case_name = cname
                    break
            if case_suffix:
                break

        # Identify noun part (strip case suffix if present)
        noun_part = word[:-len(case_suffix)] if case_suffix else word

        # Try all possible binary splits of the noun into a left and right compound component
        # We look for right-component sub-words recognized by mlmorph as nouns
        for split_point in range(2, len(noun_part) - 1):
            left = noun_part[:split_point]
            right = noun_part[split_point:]
            if not left or not right:
                continue

            # Check if left is recognized as a noun
            left_analyses = self._analyse(left + "ം")  # try restoring the anusvara
            if not left_analyses:
                left_analyses = self._analyse(left)
            if not left_analyses:
                continue

            # Check if right is recognized as a noun  
            right_analyses = self._analyse(right)
            if not right_analyses:
                continue

            # Determine the correct FST tag for the left noun
            left_raw = left_analyses[0][0]
            import re
            left_tags = re.findall(r"<([^>]+)>", left_raw)
            left_pos_raw = left_tags[0] if left_tags else "n"

            # Try generating compound (with or without case suffix)
            if case_suffix:
                # Determine locative/genitive/etc tag from case name
                gen_target = f"{left_raw}<adj>{right}<n><{case_name}>"
                gen_results = self._generate(gen_target)
            else:
                gen_target = f"{left_raw}<adj>{right}<n>"
                gen_results = self._generate(gen_target)

            if not gen_results:
                continue

            # Check if any generated form de-geminates to match the input word
            for gen_form, _ in gen_results:
                normalized = gen_form
                for gem, degem in DEGEMINATION_RULES:
                    normalized = normalized.replace(gem, degem)
                if normalized == word:
                    # Found a matching de-geminated compound — extract analysis
                    # Use the generated geminated form to get the mlmorph analysis
                    gem_analyses = self._analyse(gen_form)
                    if gem_analyses:
                        raw = gem_analyses[0][0]
                    else:
                        raw = gen_target

                    # Build MorphAnalysis from the geminated form's analysis
                    tags = re.findall(r"<([^>]+)>", raw)
                    pos_map = {
                        "n": "NOUN", "np": "NOUN", "sanskrit": "NOUN",
                        "v": "VERB", "adj": "ADJ"
                    }
                    raw_pos = tags[0] if tags else "n"
                    pos = pos_map.get(raw_pos, "NOUN")
                    stems = [s for s in re.split(r"<[^>]+>", raw) if s]
                    lemma = stems[-1] if stems else noun_part

                    # Detect case from tags
                    case = None
                    for tag in tags:
                        if tag in CASE_SUFFIXES:
                            case = tag
                            break
                    if not case:
                        case = "nominative"

                    logger.info(
                        "FallbackParser: de-gemination fallback for '%s' (generated '%s')",
                        word, gen_form
                    )
                    return MorphAnalysis(
                        raw_analysis=raw,
                        lemma=lemma,
                        pos=pos,
                        case=case,
                        number="singular",
                        person=None,
                        other_features="|".join(t for t in tags if t not in {raw_pos, case, "singular"})
                    )
        return None

    def parse(self, word):
        """
        Runs the fallback strategies in order and returns the first successful MorphAnalysis,
        or None if all strategies fail.
        """
        # Strategy 1: compound + nominalization (e.g. പ്രധാനപ്പെട്ടത്, സുന്ദരമായത്)
        result = self._try_compound_nominalization(word)
        if result is not None:
            return result

        # Strategy 2: sibilant de-gemination mismatch (e.g. ഗവേഷണശാല, ഗവേഷണശാലയിൽ)
        result = self._try_degemination(word)
        if result is not None:
            return result

        return None


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

        # Fallback using difference between word and lemma
        lemma = selected_analysis.lemma
        if len(word) > len(lemma):
            return word[len(lemma):]
        return ""

    def transform_word(self, word, selected_analysis):
        """
        Applies morphological copula generation. Preserves case realization and checks for shifts.
        Returns:
            mlmorph_copula_form (str)
            case_preserved (str): 'YES', 'NO', or 'UNKNOWN'
            copula_path (str): 'already-copular', 'case-preserving', 'adjective-nominalized', 'normal', 'not-supported'
        """
        raw_analysis = selected_analysis.raw_analysis

        # 1. Already copular check
        if "ആണ്" in word or "<aff>" in raw_analysis:
            return "ALREADY COPULAR", "UNKNOWN", "already-copular"

        # 2. Case present check (case-preserving)
        css = self.detect_css(word, selected_analysis)
        if css:
            target_analysis = f"{raw_analysis}ആണ്<aff>"
            results = self.generator.generate(target_analysis)
            if not results:
                return "NOT SUPPORTED", "UNKNOWN", "not-supported"
            
            generated_forms = []
            for form, weight in results:
                if form not in generated_forms:
                    generated_forms.append(form)
                    
            sandhi_patterns = self.get_sandhi_css_candidates(css)
            filtered = [g for g in generated_forms if any(p in g for p in sandhi_patterns)]
            if filtered:
                return filtered[0], "YES", "case-preserving"
            else:
                return generated_forms[0], "NO", "case-preserving"

        # 3. Adjective/adjectival participle check (adjective-nominalized)
        is_participle = False
        participle_suffix = None
        v_idx = raw_analysis.rfind("<v>")
        if v_idx != -1:
            stem_start = raw_analysis.rfind(">", 0, v_idx)
            if stem_start == -1:
                stem_start = 0
            else:
                stem_start += 1
            suffix_candidate = raw_analysis[stem_start:]
            participle_tags = ["adv-clause-rp-past", "cvb-adv-part-absolute", "cvb-adv-part-past", "rp"]
            if any(f"<{t}>" in suffix_candidate for t in participle_tags):
                is_participle = True
                participle_suffix = suffix_candidate

        is_adj = (
            "<adj>" in raw_analysis or 
            selected_analysis.pos == "ADJ" or
            word.endswith("പെട്ട") or 
            word.endswith("ആയ") or 
            word.endswith("മായ")
        )
        
        is_already_nominalized = raw_analysis.endswith("അത്<prn>") or "<n><deriv>" in raw_analysis
        
        if (is_participle or is_adj) and not is_already_nominalized:
            if is_participle:
                participle_surface_forms = self.generator.generate(participle_suffix)
                if participle_surface_forms:
                    participle_surface = participle_surface_forms[0][0]
                    matched_suffix = None
                    if word.endswith(participle_surface):
                        matched_suffix = participle_surface
                    elif participle_surface == "ആയ" and word.endswith("മായ"):
                        matched_suffix = "മായ"
                        
                    if matched_suffix:
                        surface_prefix = word[:-len(matched_suffix)]
                        if matched_suffix == "മായ":
                            surface_prefix += "മ"
                            
                        target_suffix = f"{participle_suffix}<n><deriv>ആണ്<aff>"
                        copula_results = self.generator.generate(target_suffix)
                        if copula_results:
                            nominalized_copula = copula_results[0][0]
                            combined = surface_prefix + nominalized_copula
                            if surface_prefix.endswith("മ") and nominalized_copula.startswith("ആ"):
                                combined = surface_prefix[:-1] + "മാ" + nominalized_copula[1:]
                            return combined, "YES", "adjective-nominalized"
            
            # For plain adjectives or fallback, try standard FST nominalization
            target_analysis = f"{raw_analysis}<n><deriv>ആണ്<aff>"
            results = self.generator.generate(target_analysis)
            if not results:
                target_analysis = f"{raw_analysis}അത്<prn>ആണ്<aff>"
                results = self.generator.generate(target_analysis)
                
            if results:
                generated_form = results[0][0]
                # Reject ungrammatical forms like 'നല്ലയതാണ്' compared to 'നല്ലതാണ്'
                if generated_form.endswith("യതാണ്") and not word.endswith("യ"):
                    return "NOT SUPPORTED", "UNKNOWN", "not-supported"
                return generated_form, "YES", "adjective-nominalized"
                
            return "NOT SUPPORTED", "UNKNOWN", "not-supported"

        # 4. Normal copula check
        target_analysis = f"{raw_analysis}ആണ്<aff>"
        results = self.generator.generate(target_analysis)
        if not results:
            return "NOT SUPPORTED", "UNKNOWN", "not-supported"
            
        return results[0][0], "YES", "normal"
