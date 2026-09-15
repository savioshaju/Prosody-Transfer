"""
standalone_syntactic_aligner.py — Standalone Syntactic Phrase & Chinking Aligner for English-Malayalam.

Theoretical & Methodological Grounding:
  1. K. A. Jayaseelan (2001), "IP-Internal Topic and Focus Phrases", Studia Linguistica:
     - Structural head-direction asymmetry: English is head-initial (SVO: S V O PP),
       Malayalam is head-final (SOV: S PP/IO O V).
     - Phrase-level constituency: NP, PP, AdvP, VP.
     - Adposition/Case duality: English Prepositional Phrase [P + NP] maps to
       Malayalam Postpositional Phrase [NP + Postposition/Case Suffix].
  2. Rodney Moag, "Malayalam: A Course in Four Volumes":
     - Case morphology: Nominative, Accusative (-നെ/-യെ), Dative (-ക്ക്/-ന്),
       Locative (-ൽ/-യിൽ/-ത്തിൽ), Ablative (-ൽനിന്ന്), Sociative (-ഓട്/-ഉമായി),
       Instrumental (-ആൽ/-കൊണ്ട്).
  3. Syntactic Chunking & Chinking:
     - Decomposes clauses into maximal phrase projections (XP).
     - Chinking isolates content lexical heads (Nouns, Verbs, Adjectives) from
       functional markers (Prepositions, Postpositions, Determiners, Case Suffixes).
     - Bipartite phrasal matching followed by fine-grained intra-phrase token projection.
     - Operates completely standalone without pretrained neural checkpoints (SimAlign/mBERT).
"""

from __future__ import annotations

import re
import difflib
from typing import Any, Dict, List, Optional, Set, Tuple

from aligner.alignment_utils import strip_punctuation, get_word_lemmas, is_word_match
from aligner.english_phrase_chunker import EnglishPhraseChunker, EnglishChunk
from cleft.phrase_chunker import MalayalamPhraseChunker, PhraseChunk, POSTPOSITIONS, PP_SUFFIXES, CASE_SUFFIXES, TIME_WORDS


# ═══════════════════════════════════════════════════════════════════════
# 1. Closed-Class Bilingual Lexicon & Functional Category Mappings
# ═══════════════════════════════════════════════════════════════════════

PREPOSITION_TO_CASE_POSTPOS: Dict[str, Dict[str, Any]] = {
    "in": {
        "cases": {"locative"},
        "suffixes": ("ൽ", "ിൽ", "ത്തിൽ", "ത്ത്", "ലെ", "ിലെ"),
        "postpositions": {"വെച്ച്", "വച്ച്", "ഉള്ളിൽ", "അകത്ത്"},
    },
    "at": {
        "cases": {"locative", "dative"},
        "suffixes": ("ൽ", "ിൽ", "ത്തിൽ", "ത്ത്", "ക്ക്", "യ്ക്ക്", "ന്"),
        "postpositions": {"വെച്ച്", "വച്ച്"},
    },
    "on": {
        "cases": {"locative"},
        "suffixes": ("ൽ", "ിൽ", "ത്തിൽ", "ത്ത്", "ന്മേൽ"),
        "postpositions": {"മേൽ", "മുകളിൽ", "വെച്ച്"},
    },
    "to": {
        "cases": {"dative"},
        "suffixes": ("ക്ക്", "യ്ക്ക്", "്ക്ക്", "ന്", "ിന്", "ലേക്ക്", "ിലേക്ക്", "ത്തേക്ക്"),
        "postpositions": {"ലേക്ക്", "നേരെ", "വരെ"},
    },
    "from": {
        "cases": {"ablative"},
        "suffixes": ("ൽനിന്ന്", "യിൽനിന്ന്", "ിൽനിന്ന്", "നിന്ന്", "നിന്നും", "ൽനിന്നും", "യിൽനിന്നും"),
        "postpositions": {"നിന്ന്", "മുതൽ", "മുതല്‍"},
    },
    "with": {
        "cases": {"sociative", "instrumental"},
        "suffixes": ("ഉമായി", "ുമായി", "മായി", "ഓട്", "യോട്", "ആൽ", "ാൽ"),
        "postpositions": {"കൂടെ", "ഒപ്പം", "കൊണ്ട്", "ഉൾപ്പെടെ", "അടക്കം"},
    },
    "by": {
        "cases": {"instrumental"},
        "suffixes": ("ആൽ", "ാൽ", "ത്തിനാൽ"),
        "postpositions": {"കൊണ്ട്", "വഴി", "മുഖേന"},
    },
    "for": {
        "cases": {"dative"},
        "suffixes": ("ക്ക്", "യ്ക്ക്", "ന്", "ിന്", "ത്തിനായി", "നായി"),
        "postpositions": {"വേണ്ടി", "ആയി", "ആയിട്ട്"},
    },
    "about": {
        "cases": {"sociative", "accusative"},
        "suffixes": ("കുറിച്ച്", "സംബന്ധിച്ച്", "നെക്കുറിച്ച്", "യെക്കുറിച്ച്"),
        "postpositions": {"കുറിച്ച്", "സംബന്ധിച്ച്", "കുറിച്ചുള്ള"},
    },
    "through": {
        "cases": {"instrumental"},
        "suffixes": ("ലൂടെ", "ിലൂടെ", "ആലൂടെ"),
        "postpositions": {"വഴി"},
    },
    "after": {
        "cases": {"dative"},
        "suffixes": ("തിനുശേഷം", "ശേഷം"),
        "postpositions": {"ശേഷം", "പിന്നീട്"},
    },
    "before": {
        "cases": {"dative"},
        "suffixes": ("തിനുമുമ്പ്", "മുമ്പ്", "മുൻപ്"),
        "postpositions": {"മുമ്പ്", "മുൻപ്"},
    },
    "without": {
        "cases": set(),
        "suffixes": ("കൂടാതെ", "ഇല്ലാതെ"),
        "postpositions": {"കൂടാതെ", "ഇല്ലാതെ"},
    },
    "under": {
        "cases": {"locative"},
        "suffixes": ("ൽ", "ിൽ"),
        "postpositions": {"കീഴിൽ", "താഴെ", "അടിയിൽ"},
    },
    "above": {
        "cases": {"locative"},
        "suffixes": ("ൽ", "ിൽ"),
        "postpositions": {"മുകളിൽ", "മീതെ"},
    },
    "between": {
        "cases": {"dative", "locative"},
        "suffixes": ("ഇടയിൽ", "തമ്മിൽ"),
        "postpositions": {"ഇടയിൽ", "തമ്മിൽ"},
    },
    "against": {
        "cases": {"dative"},
        "suffixes": ("എതിരെ"),
        "postpositions": {"എതിരെ"},
    },
}

PRONOUN_MAP: Dict[str, Set[str]] = {
    "i": {"ഞാൻ", "ഞാന്", "എന്നെ", "എനിക്ക്", "എനിക്കു്", "എന്റെ", "എൻ്റെ"},
    "me": {"എന്നെ", "എനിക്ക്", "എനിക്കു്"},
    "my": {"എന്റെ", "എൻ്റെ", "എൻറെ"},
    "mine": {"എന്റെത്", "എൻ്റേത്"},
    "we": {"ഞങ്ങൾ", "ഞങ്ങള്", "നമ്മൾ", "നമ്മള്", "ഞങ്ങളെ", "ഞങ്ങൾക്ക്", "നമുക്ക്"},
    "us": {"ഞങ്ങളെ", "നമ്മളെ", "ഞങ്ങൾക്ക്", "നമുക്ക്"},
    "our": {"ഞങ്ങളുടെ", "നമ്മുടെ"},
    "you": {"നീ", "നിങ്ങൾ", "നിങ്ങളെ", "നിനക്ക്", "നിങ്ങൾക്ക്", "നിന്റെ", "നിങ്ങളുടെ"},
    "your": {"നിന്റെ", "നിങ്ങളുടെ", "നിൻ്റെ"},
    "he": {"അവൻ", "അവന്", "അദ്ദേഹം", "ഇദ്ദേഹം", "അയാൾ", "ഇയാൾ"},
    "him": {"അവനെ", "അദ്ദേഹത്തെ", "അയാൾക്ക്", "അവന്"},
    "his": {"അവന്റെ", "അദ്ദേഹത്തിന്റെ", "അയാളുടെ", "അവൻ്റെ"},
    "she": {"അവൾ", "അവള്", "അവർ", "ഇവർ"},
    "her": {"അവളെ", "അവൾക്ക്", "അവളുടെ", "അവർക്ക്"},
    "they": {"അവർ", "അവര്", "ഇവർ", "അവരെ", "അവർക്ക്", "അവരുടെ"},
    "them": {"അവരെ", "അവർക്ക്"},
    "their": {"അവരുടെ"},
    "it": {"അത്", "ഇത്", "അതിനെ", "ഇതിനെ", "അതിന്", "ഇതിന്"},
    "its": {"അതിന്റെ", "അതിൻ്റെ", "ഇതിന്റെ"},
    "this": {"ഇത്", "ഈ"},
    "that": {"അത്", "ആ"},
    "these": {"ഇവ", "ഇവർ", "ഈ"},
    "those": {"അവ", "അവർ", "ആ"},
}

TEMPORAL_MAP: Dict[str, Set[str]] = {
    "yesterday": {"ഇന്നലെ"},
    "today": {"ഇന്ന്", "ഇന്നു്"},
    "tomorrow": {"നാളെ"},
    "now": {"ഇപ്പോൾ", "ഇപ്പൊൾ", "ഇപ്പോഴേ"},
    "then": {"അപ്പോൾ", "അന്ന്"},
    "morning": {"രാവിലെ"},
    "evening": {"വൈകുന്നേരം", "വൈകിട്ട്"},
    "night": {"രാത്രി"},
    "day": {"പകൽ", "ദിവസം"},
    "early": {"നേരത്തെ"},
    "always": {"എപ്പോഴും"},
    "never": {"ഒരിക്കലും"},
    "often": {"പലപ്പോഴും"},
    "sometimes": {"ചിലപ്പോൾ"},
    "daily": {"ദിവസവും", "പ്രതിദിനം"},
    "yearly": {"പ്രതിവർഷം", "വർഷംതോറും"},
    "monthly": {"പ്രതിമാസം"},
}

DETERMINER_NUMERAL_MAP: Dict[str, Set[str]] = {
    "a": {"ഒരു", "ഒരൊറ്റ"},
    "an": {"ഒരു", "ഒരൊറ്റ"},
    "one": {"ഒരു", "ഒന്ന്"},
    "two": {"രണ്ട്", "രണ്ടു്"},
    "three": {"മൂന്ന്", "മൂന്നു്"},
    "four": {"നാല്", "നാലു്"},
    "five": {"അഞ്ച്", "അഞ്ചു്"},
    "six": {"ആറ്"},
    "seven": {"ഏഴ്"},
    "eight": {"എട്ട്"},
    "nine": {"ഒൻപത്", "ഒമ്പത്"},
    "ten": {"പത്ത്"},
    "first": {"ഒന്നാം", "ആദ്യ", "ആദ്യത്തെ"},
    "second": {"രണ്ടാം", "രണ്ടാമത്തെ"},
    "third": {"മൂന്നാം", "മൂന്നാമത്തെ"},
    "all": {"എല്ലാ", "മുഴുവൻ", "എല്ലാം"},
    "some": {"ചില", "കുറച്ച്", "അല്പം"},
    "many": {"പല", "ധാരാളം", "നിരവധി"},
    "much": {"കൂടുതൽ", "വളരെ"},
    "few": {"കുറച്ച്", "ചില"},
    "only": {"മാത്രം", "മാത്രമേ"},
}

CORE_VERBAL_ROOTS: Dict[str, Set[str]] = {
    "buy": {"വാങ്ങുക", "വാങ്ങി", "വാങ്ങുന്നു", "വാങ്ങും", "വാങ്ങിച്ചത്", "വാങ്ങിയത്"},
    "bought": {"വാങ്ങി", "വാങ്ങിയത്", "വാങ്ങിച്ചു"},
    "see": {"കാണുക", "കണ്ടു", "കാണുന്നു", "കാണും", "കണ്ടത്"},
    "saw": {"കണ്ടു", "കണ്ടത്", "കണ്ടുകഴിഞ്ഞു"},
    "give": {"കൊടുക്കുക", "നൽകുക", "കൊടുത്തു", "നൽകി", "കൊടുത്തത്"},
    "gave": {"കൊടുത്തു", "നൽകി", "കൊടുത്തത്"},
    "take": {"എടുക്കുക", "എടുത്തു", "എടുക്കുന്നു", "എടുത്തത്"},
    "took": {"എടുത്തു", "എടുത്തത്"},
    "go": {"പോകുക", "പോയി", "പോകുന്നു", "പോയത്"},
    "went": {"പോയി", "പോയത്"},
    "come": {"വരുക", "വന്നു", "വരുന്നു", "വന്നത്"},
    "came": {"വന്നു", "വന്നത്"},
    "say": {"പറയുക", "പറഞ്ഞു", "പറയുന്നു", "പറഞ്ഞത്"},
    "said": {"പറഞ്ഞു", "പറഞ്ഞത്"},
    "tell": {"പറയുക", "പറഞ്ഞു", "പറഞ്ഞു-കൊടുത്തു"},
    "told": {"പറഞ്ഞു", "പറഞ്ഞത്"},
    "write": {"എഴുതുക", "എഴുതി", "എഴുതുന്നു", "എഴുതിയത്"},
    "wrote": {"എഴുതി", "എഴുതിയത്"},
    "read": {"വായിക്കുക", "വായിച്ചു", "വായിക്കുന്നു", "വായിച്ചത്"},
    "make": {"ഉണ്ടാക്കുക", "സൃഷ്ടിക്കുക", "ഉണ്ടാക്കി"},
    "made": {"ഉണ്ടാക്കി", "സൃഷ്ടിച്ചു", "ഉണ്ടാക്കിയത്"},
    "do": {"ചെയ്യുക", "ചെയ്തു", "ചെയ്യുന്നു", "ചെയ്തത്"},
    "did": {"ചെയ്തു", "ചെയ്തത്"},
    "put": {"വെക്കുക", "വെച്ചു", "ഇടുക", "ഇട്ടു"},
    "eat": {"തിന്നുക", "കഴിക്കുക", "തിന്നു", "കഴിച്ചു"},
    "ate": {"തിന്നു", "കഴിച്ചു", "തിന്നത്"},
    "send": {"അയക്കുക", "അയച്ചു", "അയച്ചത്"},
    "sent": {"അയച്ചു", "അയച്ചത്"},
    "find": {"കണ്ടെത്തുക", "കണ്ടെത്തി"},
    "found": {"കണ്ടെത്തി", "കണ്ടെത്തിയത്"},
    "elect": {"തെരഞ്ഞെടുക്കുക", "തിരഞ്ഞെടുക്കുക", "തെരഞ്ഞെടുത്തു"},
    "elected": {"തെരഞ്ഞെടുത്തു", "തെരഞ്ഞെടുക്കപ്പെട്ടു"},
    "marry": {"വിവാഹം", "വിവാഹം കഴിക്കുക", "വിവാഹം കഴിച്ചു"},
    "married": {"വിവാഹം കഴിച്ചു", "വിവാഹം ചെയ്തു"},
    "happen": {"സംഭവിക്കുക", "സംഭവിച്ചു", "നടന്നു"},
    "happened": {"സംഭവിച്ചു", "നടന്നു"},
    "control": {"നിയന്ത്രിക്കുക", "നിയന്ത്രിച്ചു"},
    "controlled": {"നിയന്ത്രിച്ചു"},
    "describe": {"വിശേഷിപ്പിക്കുക", "വിവരിക്കുക", "വിശേഷിപ്പിച്ചു"},
    "described": {"വിശേഷിപ്പിച്ചു", "വിവരിച്ചു"},
    "establish": {"സ്ഥാപിക്കുക", "സ്ഥാപിച്ചു"},
    "established": {"സ്ഥാപിച്ചു"},
    "cultivate": {"കൃഷി", "കൃഷി ചെയ്യുക", "കൃഷി ചെയ്തു"},
    "cultivated": {"കൃഷി ചെയ്തു", "കൃഷി ചെയ്യപ്പെടുന്നു"},
    "kill": {"കൊല്ലുക", "കൊന്നു", "കൊന്നത്"},
    "killed": {"കൊന്നു", "കൊന്നത്"},
    "beat": {"തല്ലുക", "തല്ലി", "തല്ലിയത്"},
}

CORE_NOUN_MAP: Dict[str, Set[str]] = {
    "father": {"അച്ഛൻ", "അച്ഛന്", "പിതാവ്", "പിതാവിനെ"},
    "mother": {"അമ്മ", "മാതാവ്", "മാതാവിനെ"},
    "child": {"കുട്ടി", "കുട്ടിയെ", "കുട്ടിക്ക്"},
    "children": {"കുട്ടികൾ", "കുട്ടികളെ", "കുട്ടികൾക്ക്"},
    "book": {"പുസ്തകം", "പുസ്തകത്തെ", "പുസ്തകങ്ങൾ"},
    "books": {"പുസ്തകങ്ങൾ", "പുസ്തകങ്ങളെ"},
    "garden": {"തോട്ടം", "തോട്ടത്തിൽ", "പൂന്തോട്ടം"},
    "letter": {"കത്ത്", "കത്തുകൾ", "കത്തിനെ"},
    "elephant": {"ആന", "ആനയെ", "ആനയ്ക്ക്"},
    "water": {"വെള്ളം", "വെള്ളത്തെ", "ജലം"},
    "house": {"വീട്", "വീട്ടിൽ", "ഭവനം"},
    "home": {"വീട്", "വീട്ടിൽ"},
    "school": {"സ്കൂൾ", "സ്കൂളിൽ"},
    "college": {"കോളേജ്", "കോളേജിൽ"},
    "university": {"സർവകലാശാല", "സർവ്വകലാശാല"},
    "city": {"നഗരം", "നഗരത്തിൽ"},
    "country": {"രാജ്യം", "നാട്", "ദേശത്ത്"},
    "tree": {"മരം", "മരത്തിൽ"},
    "money": {"പണം", "പണത്തെ"},
    "friend": {"സുഹൃത്ത്", "കൂട്ടുകാരൻ"},
    "name": {"പേര്", "നാമം"},
    "teacher": {"അധ്യാപകൻ", "അധ്യാപിക", "ഗുരു"},
    "king": {"രാജാവ്", "മഹാരാജാവ്"},
    "flower": {"പൂവ്", "പുഷ്പം"},
    "cat": {"പൂച്ച", "പൂച്ചയെ"},
    "dog": {"നായ", "പട്ടി"},
    "bird": {"പക്ഷി", "പറവ"},
    "fish": {"മത്സ്യം", "മീൻ"},
    "food": {"ഭക്ഷണം", "ആഹാരം"},
    "fruit": {"പഴം", "ഫലം"},
    "report": {"റിപ്പോർട്ട്", "റിപ്പോർട്ടിൽ"},
    "question": {"ചോദ്യം", "സംശയം"},
    "service": {"സേവനം", "സേവനമായി"},
    "system": {"സംവിധാനം", "സിസ്റ്റം"},
}


# ═══════════════════════════════════════════════════════════════════════
# 2. Phonetic & Entity Similarity Matcher
# ═══════════════════════════════════════════════════════════════════════

class PhoneticEntityMatcher:
    """
    Matches Named Entities, digits, dates, and transliterated loan words across English & Malayalam.
    """

    @staticmethod
    def match_entities(en_word: str, ml_word: str) -> float:
        clean_en = strip_punctuation(en_word).lower()
        clean_ml = strip_punctuation(ml_word)

        if not clean_en or not clean_ml:
            return 0.0

        # Exact numeral / year match (e.g. "1994" <-> "1994", "1895" <-> "1895")
        if clean_en == clean_ml:
            return 1.0

        if clean_en.isdigit() and clean_ml.startswith(clean_en):
            return 0.95

        # Digits with Malayalam case suffixes (e.g. "1994" <-> "1994ൽ", "2005" <-> "2005ൽ")
        digits_en = re.findall(r"\d+", clean_en)
        digits_ml = re.findall(r"\d+", clean_ml)
        if digits_en and digits_ml and digits_en == digits_ml:
            return 0.95

        # Percentage match
        if "%" in clean_en and "%" in clean_ml:
            if digits_en == digits_ml:
                return 1.0

        # Phonetic transliteration heuristics for proper names (e.g. Mary/മേരി, John/ജോൺ, George/ജോർജ്ജ്)
        en_latin = re.sub(r"[^a-z]", "", clean_en)
        if len(en_latin) >= 3:
            # Check known entity name pairs
            known_names = {
                "mary": {"മേരി", "മേരിക്ക്", "മേരിയെ", "മേരിയുടെ"},
                "john": {"ജോൺ", "ജോണിന്", "ജോണിനെ", "ജോണിന്റെ"},
                "george": {"ജോർജ്ജ്", "ജോർജ്ജിനെ", "ജോർജ്ജിന്റെ", "ജോർജ്ജ്"},
                "william": {"വില്ല്യം", "വില്യം"},
                "morgan": {"മോർഗൻ", "മോർഗന്റെ"},
                "angul": {"ആംഗുലി", "ആംഗുൽ"},
                "odisha": {"ഒഡിഷ", "ഒഡീഷ"},
                "volleyball": {"വോളിബോൾ"},
                "calcutta": {"കൽക്കട്ട"},
                "kolkata": {"കൊൽക്കത്ത"},
                "delhi": {"ഡൽഹി", "ഡെൽഹി", "ഡൽഹിയിൽ", "ഡെൽഹിയിൽ"},
                "india": {"ഇന്ത്യ", "ഇന്ത്യൻ", "ഇന്ത്യയിൽ", "ഇന്ത്യയിലെ", "ഭാരതം"},
                "indian": {"ഇന്ത്യൻ", "ഇന്ത്യയിലെ"},
                "karate": {"കരാട്ടെ"},
                "commonwealth": {"കോമൺവെൽത്ത്"},
                "arizona": {"അരിസോണ", "അരിസോണയിൽ"},
                "canyon": {"കാൻയോൺ"},
                "america": {"അമേരിക്ക", "അമേരിക്കൻ", "അമേരിക്കയിൽ"},
                "american": {"അമേരിക്കൻ"},
                "kerala": {"കേരളം", "കേരളത്തിൽ", "കേരള"},
                "bangalore": {"ബാംഗ്ലൂർ", "ബാംഗ്ലൂരിന്റെ", "ബെംഗളൂരു"},
                "mumbai": {"മുംബൈ", "മുംബൈയിൽ"},
                "chennai": {"ചെന്നൈ", "ചെന്നൈയിൽ"},
                "italy": {"ഇറ്റലി", "ഇറ്റലിയിലേക്ക്"},
                "france": {"ഫ്രാൻസ്", "ഫ്രാൻസിലേക്ക്"},
                "spain": {"സ്പെയിൻ", "സ്പെയിനിലേക്ക്"},
                "jamaica": {"ജമൈക്ക", "ജമൈക്കയുടെ"},
                "ajanta": {"അജന്ത", "അജന്തയിൽ"},
                "krishna": {"കൃഷ്ണൻ", "കൃഷ്ണന്റെ"},
                "arjuna": {"അർജുനൻ", "അർജുനനെ", "അർജുനന്റെ"},
                "bjd": {"ബി. ജെ. ഡി", "ബി.ജെ.ഡി"},
                "mla": {"എം. എൽ. എ", "എം.എൽ.എ"},
                "ipc": {"ഐ.പി.സി", "ശിക്ഷാനിയമം"},
            }
            if en_latin in known_names and any(ml_name in clean_ml for ml_name in known_names[en_latin]):
                return 0.98

            # Generalized phonetic consonant matcher fallback
            p_score = PhoneticEntityMatcher._phonetic_consonant_score(en_latin, clean_ml)
            if p_score >= 0.50:
                return round(0.70 + p_score * 0.25, 3)

        return 0.0

    @staticmethod
    def _phonetic_consonant_score(en_clean: str, ml_word: str) -> float:
        en_to_ml_map = {
            'k': ('ക', 'ഖ'), 'c': ('ക', 'സ', 'ച'), 'g': ('ഗ', 'ഘ'),
            'ch': ('ച', 'ഛ'), 'j': ('ജ', 'ഝ'),
            't': ('ട', 'ഠ', 'ത', 'ഥ', 'റ്റ'), 'd': ('ഡ', 'ഢ', 'ദ', 'ധ'),
            'n': ('ണ', 'ന', 'ങ', 'ഞ'), 'p': ('പ', 'ഫ'), 'f': ('ഫ'),
            'b': ('ബ', 'ഭ'), 'm': ('മ'), 'y': ('യ'), 'r': ('ര', 'റ'),
            'l': ('ല', 'ള', 'ഴ'), 'v': ('വ'), 'w': ('വ'),
            's': ('ശ', 'ഷ', 'സ'), 'sh': ('ശ', 'ഷ'), 'h': ('ഹ'), 'z': ('സ', 'ഷ')
        }
        en_cons = []
        i = 0
        while i < len(en_clean):
            if i + 1 < len(en_clean) and en_clean[i:i+2] in ('ch', 'sh', 'th', 'ph'):
                en_cons.append(en_clean[i:i+2])
                i += 2
            elif en_clean[i] in 'bcdfghjklmnpqrstvwxyz':
                en_cons.append(en_clean[i])
                i += 1
            else:
                i += 1

        ml_cons = [ch for ch in ml_word if ('\u0D15' <= ch <= '\u0D3A' or ch in ('ൺ', 'ൻ', 'ർ', 'ൽ', 'ൾ', 'ൿ'))]
        if not en_cons or not ml_cons:
            return 0.0

        matches = 0
        ml_idx = 0
        for ec in en_cons:
            cands = en_to_ml_map.get(ec, ())
            for j in range(ml_idx, min(ml_idx + 3, len(ml_cons))):
                if ml_cons[j] in cands:
                    matches += 1
                    ml_idx = j + 1
                    break
        return (2.0 * matches) / (len(en_cons) + len(ml_cons))


# ═══════════════════════════════════════════════════════════════════════
# 3. English & Malayalam Phrase Chinker
# ═══════════════════════════════════════════════════════════════════════

class SyntacticChinker:
    """
    Decomposes typed syntactic phrase chunks into content heads and functional markers (chinking).
    """

    @staticmethod
    def chink_english_chunk(chunk: EnglishChunk, doc_tokens: List[str]) -> Dict[str, Any]:
        """
        Chinks an English chunk into:
          - chunk_type: NP, PP, AdvP, VP
          - content_tokens: list of content word indices
          - function_tokens: list of functional word indices (prepositions, determiners, auxiliaries)
          - head_token_idx: index of main syntactic head word
        """
        span_indices = [i for i in range(chunk.start_token, chunk.end_token + 1) if 0 <= i < len(doc_tokens)]
        c_type = chunk.chunk_type
        words = [doc_tokens[i] for i in span_indices]
        clean_words = [strip_punctuation(w).lower() for w in words]

        content_indices = []
        function_indices = []
        head_idx = span_indices[-1] if span_indices else 0

        if c_type == "PP":
            # First token is usually the preposition (functional head)
            function_indices.append(span_indices[0])
            for idx, w in zip(span_indices[1:], clean_words[1:]):
                if w in ("the", "a", "an", "this", "that", "these", "those", "of"):
                    function_indices.append(idx)
                else:
                    content_indices.append(idx)
            # Head noun is typically the rightmost noun in the complement NP
            head_idx = content_indices[-1] if content_indices else span_indices[-1]

        elif c_type == "NP":
            for idx, w in zip(span_indices, clean_words):
                if w in ("the", "a", "an", "this", "that", "these", "those"):
                    function_indices.append(idx)
                else:
                    content_indices.append(idx)
            head_idx = content_indices[-1] if content_indices else span_indices[-1]

        elif c_type == "VP":
            for idx, w in zip(span_indices, clean_words):
                if w in ("is", "was", "are", "were", "has", "have", "had", "did", "do", "does", "will", "would", "shall", "should", "may", "might", "can", "could"):
                    function_indices.append(idx)
                else:
                    content_indices.append(idx)
            head_idx = content_indices[-1] if content_indices else span_indices[-1]

        else:  # AdvP or other
            content_indices = list(span_indices)
            head_idx = span_indices[-1]

        return {
            "chunk_type": c_type,
            "span_indices": span_indices,
            "text": chunk.text,
            "content_indices": content_indices,
            "function_indices": function_indices,
            "head_token_idx": head_idx,
            "clause_id": getattr(chunk, "clause_id", 0),
        }

    @staticmethod
    def chink_malayalam_chunk(chunk: PhraseChunk, tokens: List[str]) -> Dict[str, Any]:
        """
        Chinks a Malayalam phrase chunk into:
          - chunk_type: NP, PP, AdvP, VP
          - content_tokens: list of content word indices
          - function_tokens: list of functional word indices (postpositions, clitics)
          - head_token_idx: index of main syntactic head word
        """
        span_indices = list(range(chunk.start_idx, chunk.end_idx + 1))
        c_type = chunk.chunk_type
        words = [tokens[i] for i in span_indices]
        clean_words = [strip_punctuation(w) for w in words]

        content_indices = []
        function_indices = []
        head_idx = span_indices[-1] if span_indices else 0

        if c_type == "PP":
            # Rightmost token is often a standalone postposition (e.g. തോട്ടത്തിൽ വെച്ച്)
            for idx, w in zip(span_indices, clean_words):
                if w in POSTPOSITIONS:
                    function_indices.append(idx)
                else:
                    content_indices.append(idx)
            head_idx = content_indices[0] if content_indices else span_indices[0]

        elif c_type == "NP":
            for idx, w in zip(span_indices, clean_words):
                if w in ("ഒരു", "ഈ", "ആ", "ഏത്"):
                    function_indices.append(idx)
                else:
                    content_indices.append(idx)
            # In Malayalam head-final NP, head noun is at the right end
            head_idx = content_indices[-1] if content_indices else span_indices[-1]

        elif c_type == "VP":
            content_indices = list(span_indices)
            head_idx = span_indices[-1]

        else:
            content_indices = list(span_indices)
            head_idx = span_indices[-1]

        return {
            "chunk_type": c_type,
            "span_indices": span_indices,
            "text": chunk.text,
            "content_indices": content_indices,
            "function_indices": function_indices,
            "head_token_idx": head_idx,
            "clause_id": getattr(chunk, "clause_id", 0),
        }


# ═══════════════════════════════════════════════════════════════════════
# 4. Standalone Syntactic Phrase & Chinking Aligner
# ═══════════════════════════════════════════════════════════════════════

class StandaloneSyntacticAligner:
    """
    Standalone Syntactic Phrase & Chinking Alignment Pipeline.
    Aligns English and Malayalam words using:
      1. Syntactic Phrase & Clause Chunking (NP, PP, AdvP, VP)
      2. Functional Chinking (Prepositions <-> Postpositions / Case Markers)
      3. Cross-lingual Case/Adposition Matrix
      4. Bilingual Lexicon & Morphological Lemma Matching (mlmorph)
      5. Phonetic Transliteration Entity Matching
      6. Intra-Phrase Fine-Grained Token Projection
    """

    def __init__(self):
        self.en_chunker = EnglishPhraseChunker()
        self.ml_chunker = MalayalamPhraseChunker()
        self.chinker = SyntacticChinker()

    def _tokenize(self, text: str) -> List[str]:
        """Universal punctuation-preserving tokenizer."""
        if not text:
            return []
        pattern = r"[\w\u0D00-\u0D7F\u200C\u200D]+(?:[.-][\w\u0D00-\u0D7F\u200C\u200D]+)*|[.,!?;:()\[\]\"'“”‘’]"
        tokens = re.findall(pattern, text)
        return [t for t in tokens if t.strip()]

    def _compute_lexical_similarity(self, en_word: str, ml_word: str) -> float:
        """
        Compute direct lexical similarity using bilingual lexicon, mlmorph lemmas,
        and phonetic entity matching.
        """
        c_en = strip_punctuation(en_word).lower()
        c_ml = strip_punctuation(ml_word)

        if not c_en or not c_ml:
            return 0.0

        # 1. Phonetic / Entity matching (highest priority for names & numbers)
        pe_score = PhoneticEntityMatcher.match_entities(c_en, c_ml)
        if pe_score > 0.8:
            return pe_score

        # 2. Closed-class pronoun matching
        if c_en in PRONOUN_MAP and any(ml_p in c_ml for ml_p in PRONOUN_MAP[c_en]):
            return 0.95

        # 3. Temporal word matching
        if c_en in TEMPORAL_MAP and any(ml_t in c_ml for ml_t in TEMPORAL_MAP[c_en]):
            return 0.95

        # 4. Determiner / Numeral matching
        if c_en in DETERMINER_NUMERAL_MAP and any(ml_d in c_ml for ml_d in DETERMINER_NUMERAL_MAP[c_en]):
            return 0.95

        # 5. Core noun dictionary
        if c_en in CORE_NOUN_MAP and any(ml_n in c_ml for ml_n in CORE_NOUN_MAP[c_en]):
            return 0.95

        # 6. Core verbal root dictionary
        if c_en in CORE_VERBAL_ROOTS and any(ml_v in c_ml for ml_v in CORE_VERBAL_ROOTS[c_en]):
            return 0.95

        # 7. Morphological lemma overlap (via mlmorph)
        ml_lemmas = get_word_lemmas(c_ml)
        for ml_lem in ml_lemmas:
            if c_en in CORE_NOUN_MAP and ml_lem in CORE_NOUN_MAP[c_en]:
                return 0.90
            if c_en in CORE_VERBAL_ROOTS and ml_lem in CORE_VERBAL_ROOTS[c_en]:
                return 0.90

        # 8. English word in Malayalam text (code-mixing / loanwords)
        if c_en in c_ml.lower():
            return 0.85

        return 0.0

    def _compute_phrase_compatibility(
        self,
        en_chink: Dict[str, Any],
        ml_chink: Dict[str, Any],
        en_tokens: List[str],
        ml_tokens: List[str],
    ) -> float:
        """
        Compute syntactic and semantic compatibility score between an English chunk and Malayalam chunk.
        """
        score = 0.0
        en_type = en_chink["chunk_type"]
        ml_type = ml_chink["chunk_type"]

        # Category compatibility
        if en_type == ml_type:
            score += 0.35
        elif (en_type == "PP" and ml_type == "NP") or (en_type == "NP" and ml_type == "PP"):
            # Often English PP maps to Malayalam case-marked NP (e.g. "to John" -> "ജോണിന്")
            score += 0.25
        elif (en_type == "AdvP" and ml_type == "PP") or (en_type == "PP" and ml_type == "AdvP"):
            score += 0.25

        # Case / Adposition correspondence check
        en_func_words = [strip_punctuation(en_tokens[i]).lower() for i in en_chink["function_indices"]]
        ml_func_words = [strip_punctuation(ml_tokens[i]) for i in ml_chink["function_indices"]]
        ml_chunk_text = " ".join(ml_tokens[i] for i in ml_chink["span_indices"])

        has_prep_match = False
        for prep in en_func_words:
            if prep in PREPOSITION_TO_CASE_POSTPOS:
                prep_info = PREPOSITION_TO_CASE_POSTPOS[prep]
                # Check postposition
                if any(post in ml_func_words for post in prep_info["postpositions"]):
                    has_prep_match = True
                    score += 0.40
                    break
                # Check case suffix on Malayalam words within this chunk
                ml_chunk_words = [strip_punctuation(ml_tokens[idx]) for idx in ml_chink["span_indices"]]
                if any(any(w.endswith(sfx) for sfx in prep_info["suffixes"]) for w in ml_chunk_words):
                    has_prep_match = True
                    score += 0.35
                    break

        # Lexical content head similarity
        en_content = [en_tokens[i] for i in en_chink["content_indices"]]
        ml_content = [ml_tokens[i] for i in ml_chink["content_indices"]]

        lex_max = 0.0
        for ew in en_content:
            for mw in ml_content:
                sim = self._compute_lexical_similarity(ew, mw)
                if sim > lex_max:
                    lex_max = sim

        score += lex_max * 0.50

        return score

    def align(self, src_sentence: str, tgt_sentence: str) -> Dict[str, Any]:
        """
        Execute standalone syntactic phrase and token alignment for English -> Malayalam.
        Returns dictionary in exact format expected by downstream cleft pipelines:
          - source_tokens, target_tokens
          - aligned_pairs: List[Tuple[str, str]]
          - reconciled_pairs: List[Dict[str, Any]]
        """
        en_tokens = self._tokenize(src_sentence)
        ml_tokens = self._tokenize(tgt_sentence)

        if not en_tokens or not ml_tokens:
            return {
                "source_tokens": en_tokens,
                "target_tokens": ml_tokens,
                "aligned_pairs": [],
                "reconciled_pairs": [],
                "phrase_alignments": [],
            }

        # 1. Parse into English syntactic phrase chunks
        en_parse = self.en_chunker.parse(src_sentence)
        en_chunks = en_parse.get("chunks", [])
        en_tokens = en_parse.get("tokens") or self._tokenize(src_sentence)
        en_chinks = [self.chinker.chink_english_chunk(c, en_tokens) for c in en_chunks]

        # 2. Parse into Malayalam syntactic phrase chunks
        ml_chunks = self.ml_chunker.chunk_sentence(ml_tokens, sentence_text=tgt_sentence)
        ml_chinks = [self.chinker.chink_malayalam_chunk(c, ml_tokens) for c in ml_chunks]

        # 3. Bipartite Phrasal Alignment
        phrase_matches: List[Tuple[int, int, float]] = []
        used_en = set()
        used_ml = set()

        # Compute pairwise phrasal compatibility matrix
        compat_scores = []
        for i, ec in enumerate(en_chinks):
            for j, mc in enumerate(ml_chinks):
                s = self._compute_phrase_compatibility(ec, mc, en_tokens, ml_tokens)
                compat_scores.append((s, i, j))

        # Greedy maximum weight bipartite matching
        compat_scores.sort(key=lambda x: x[0], reverse=True)
        for s, i, j in compat_scores:
            if s >= 0.30 and i not in used_en and j not in used_ml:
                phrase_matches.append((i, j, s))
                used_en.add(i)
                used_ml.add(j)

        # 4. Fine-Grained Intra-Phrase Token Alignment
        token_pairs: List[Tuple[int, int, str]] = []
        used_en_tokens = set()
        used_ml_tokens = set()

        for i_en, j_ml, p_score in phrase_matches:
            ec = en_chinks[i_en]
            mc = ml_chinks[j_ml]

            # 4a. Content word alignment (bilingual lexicon + phonetic similarity)
            for en_idx in ec["content_indices"]:
                best_s = 0.0
                best_tgt = -1
                for ml_idx in mc["content_indices"]:
                    sim = self._compute_lexical_similarity(en_tokens[en_idx], ml_tokens[ml_idx])
                    if sim > best_s:
                        best_s = sim
                        best_tgt = ml_idx

                if best_tgt != -1 and best_s >= 0.5:
                    token_pairs.append((en_idx, best_tgt, "SYNTACTIC_PHRASE_CONTENT"))
                    used_en_tokens.add(en_idx)
                    used_ml_tokens.add(best_tgt)
                elif ec["head_token_idx"] == en_idx and mc["head_token_idx"] not in used_ml_tokens:
                    # Head-to-head alignment fallback within phrase
                    token_pairs.append((en_idx, mc["head_token_idx"], "SYNTACTIC_HEAD_PROJECTION"))
                    used_en_tokens.add(en_idx)
                    used_ml_tokens.add(mc["head_token_idx"])

            # 4b. Functional word alignment (Preposition <-> Postposition)
            for en_fn in ec["function_indices"]:
                fn_word = strip_punctuation(en_tokens[en_fn]).lower()
                for ml_fn in mc["function_indices"]:
                    ml_word = strip_punctuation(ml_tokens[ml_fn])
                    if fn_word in PREPOSITION_TO_CASE_POSTPOS:
                        if ml_word in PREPOSITION_TO_CASE_POSTPOS[fn_word]["postpositions"]:
                            token_pairs.append((en_fn, ml_fn, "ADPOSITION_ALIGNMENT"))
                            used_en_tokens.add(en_fn)
                            used_ml_tokens.add(ml_fn)
                            break
                    elif fn_word in DETERMINER_NUMERAL_MAP and ml_word in DETERMINER_NUMERAL_MAP[fn_word]:
                        token_pairs.append((en_fn, ml_fn, "DETERMINER_ALIGNMENT"))
                        used_en_tokens.add(en_fn)
                        used_ml_tokens.add(ml_fn)
                        break

        # 5. Global Lexical Matching for remaining unaligned tokens
        for en_idx, en_tok in enumerate(en_tokens):
            if en_idx in used_en_tokens:
                continue
            best_s = 0.0
            best_tgt = -1
            for ml_idx, ml_tok in enumerate(ml_tokens):
                if ml_idx in used_ml_tokens:
                    continue
                sim = self._compute_lexical_similarity(en_tok, ml_tok)
                if sim > best_s:
                    best_s = sim
                    best_tgt = ml_idx

            if best_tgt != -1 and best_s >= 0.5:
                token_pairs.append((en_idx, best_tgt, "GLOBAL_LEXICAL_ANCHOR"))
                used_en_tokens.add(en_idx)
                used_ml_tokens.add(best_tgt)

        # 6. Align matching punctuation (e.g. final periods)
        if en_tokens and ml_tokens:
            if en_tokens[-1] in (".", "!", "?") and ml_tokens[-1] in (".", "!", "?"):
                token_pairs.append((len(en_tokens) - 1, len(ml_tokens) - 1, "STRUCTURAL_PUNCT"))

        # Sort alignments
        token_pairs.sort(key=lambda x: (x[0], x[1]))

        reconciled_pairs = []
        aligned_pairs = []
        for s_i, t_j, conf in token_pairs:
            src_w = en_tokens[s_i]
            tgt_w = ml_tokens[t_j]
            is_punct = (not strip_punctuation(src_w)) and (not strip_punctuation(tgt_w))
            reconciled_pairs.append({
                "src_index": s_i,
                "src_word": src_w,
                "tgt_index": t_j,
                "tgt_word": tgt_w,
                "is_lexical": not is_punct,
                "is_bidirectional": True,
                "is_punctuation_only": is_punct,
                "confidence": conf,
            })
            aligned_pairs.append((src_w, tgt_w))

        phrase_align_records = []
        for i_en, j_ml, s in phrase_matches:
            phrase_align_records.append({
                "en_chunk": en_chinks[i_en]["text"],
                "en_type": en_chinks[i_en]["chunk_type"],
                "ml_chunk": ml_chinks[j_ml]["text"],
                "ml_type": ml_chinks[j_ml]["chunk_type"],
                "score": round(s, 3),
            })

        # Calculate lexical coverage
        src_lexical = sum(1 for w in en_tokens if strip_punctuation(w).strip())
        aligned_src_lexical = sum(1 for p in reconciled_pairs if p["is_lexical"])
        coverage = aligned_src_lexical / max(src_lexical, 1)

        return {
            "source_tokens": en_tokens,
            "target_tokens": ml_tokens,
            "src_tokens": en_tokens,
            "tgt_tokens": ml_tokens,
            "aligned_pairs": aligned_pairs,
            "reconciled_pairs": reconciled_pairs,
            "phrase_alignments": phrase_align_records,
            "coverage": coverage,
            "aligner_name": "StandaloneSyntacticAligner",
        }

    def align_focus(
        self,
        en_sentence: str,
        ml_sentence: str,
        en_focus: str,
    ) -> Dict[str, Any]:
        """
        Syntactic focus constituent projection:
        Projects an English focus word or phrase to its corresponding maximal
        Malayalam constituent (XP) adhering to Dravidian island constraints (Jayaseelan 2001).
        """
        align_res = self.align(en_sentence, ml_sentence)
        en_tokens = align_res["source_tokens"]
        ml_tokens = align_res["target_tokens"]
        reconciled = align_res["reconciled_pairs"]

        clean_focus_words = [strip_punctuation(w).lower() for w in en_focus.split() if strip_punctuation(w)]
        en_clean_tokens = [strip_punctuation(w).lower() for w in en_tokens]

        # 1. Locate English focus token indices
        focus_src_indices = []
        flen = len(clean_focus_words)
        if flen > 0:
            for i in range(len(en_clean_tokens) - flen + 1):
                if en_clean_tokens[i : i + flen] == clean_focus_words:
                    focus_src_indices = list(range(i, i + flen))
                    break
            if not focus_src_indices:
                for i, w in enumerate(en_clean_tokens):
                    if any(fw in w or w in fw for fw in clean_focus_words if fw):
                        focus_src_indices.append(i)

        # 2. Extract aligned Malayalam target indices
        matched_tgt_indices = [
            p["tgt_index"]
            for p in reconciled
            if p["src_index"] in focus_src_indices and p.get("is_lexical", True)
        ]

        if not matched_tgt_indices:
            # Fallback to phrase-level overlap
            for pa in align_res.get("phrase_alignments", []):
                en_c = pa["en_chunk"].lower()
                if any(fw in en_c for fw in clean_focus_words):
                    ml_c_text = pa["ml_chunk"]
                    ml_words = ml_c_text.split()
                    span_indices = []
                    for mw in ml_words:
                        clean_mw = strip_punctuation(mw)
                        for ti, tok in enumerate(ml_tokens):
                            if clean_mw and clean_mw == strip_punctuation(tok):
                                span_indices.append(ti)
                                break
                    if span_indices:
                        matched_tgt_indices = span_indices
                        break

        if not matched_tgt_indices:
            # First word fallback
            matched_tgt_indices = [0] if ml_tokens else []

        # 3. Enforce Dravidian Island Constraints (Jayaseelan 2001):
        # Expand focus span to enclosing maximal phrase (e.g. PP postposition or NP case head)
        min_t = min(matched_tgt_indices)
        max_t = max(matched_tgt_indices)

        # P-stranding ban: if next token is postposition (e.g. വെച്ച്), absorb it
        if max_t + 1 < len(ml_tokens):
            next_clean = strip_punctuation(ml_tokens[max_t + 1]).strip()
            if next_clean in POSTPOSITIONS:
                max_t += 1

        span_indices = list(range(min_t, max_t + 1))
        selected_text = " ".join(ml_tokens[i] for i in span_indices if strip_punctuation(ml_tokens[i]).strip())

        return {
            "selected_constituent": selected_text,
            "focus_span_indices": span_indices,
            "span_range": f"[{min_t}..{max_t}]",
            "alignment_result": align_res,
        }
