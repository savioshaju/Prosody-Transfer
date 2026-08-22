import sys
import os
import json
import re
from mlmorph import Analyser, Generator

# Ensure UTF-8 console output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ssf_pipeline.verb_normalizer import (
    VerbNormalizer, VerbAnalysis,
    CLASS_PERMISSIVE, CLASS_HABITUAL_POS, CLASS_HABITUAL_NEG,
    _is_valid_nominalized_verb
)

# ---------------------------------------------------------------------------
# Corpus of 30+ sentence-level test cases per category
# ---------------------------------------------------------------------------

corpus = {
    "permissive": [
        ("ഞങ്ങൾക്ക് നാളെ അങ്ങോട്ട് വരാം.", "വരാം", "വരാവുന്നത്"),
        ("നിങ്ങൾക്ക് ഇപ്പോൾ വീട്ടിലേക്ക് പോകാം.", "പോകാം", "പോകാവുന്നത്"),
        ("അവൻ ഈ വലിയ ജോലി ചെയ്യാം എന്ന് പറഞ്ഞു.", "ചെയ്യാം", "ചെയ്യാവുന്നത്"),
        ("നമുക്ക് ആ കാഴ്ച അവിടെനിന്ന് കാണാം.", "കാണാം", "കാണാവുന്നത്"),
        ("അവർക്ക് സത്യം ധൈര്യത്തോടെ പറയാം.", "പറയാം", "പറയാവുന്നത്"),
        ("കുട്ടികൾക്ക് പുതിയ പാഠം എഴുതാം.", "എഴുതാം", "എഴുതാവുന്നത്"),
        ("അവൾക്ക് നല്ല പാട്ടുകൾ പാടാം.", "പാടാം", "പാടാവുന്നത്"),
        ("ദാഹിക്കുമ്പോൾ ശുദ്ധജലം കുടിക്കാം.", "കുടിക്കാം", "കുടിക്കാവുന്നത്"),
        ("വിശക്കുമ്പോൾ ഭക്ഷണം നന്നായി തിന്നാം.", "തിന്നാം", "തിന്നാവുന്നത്"),
        ("അവിടെ കുറച്ചുനേരം സുഖമായി ഇരിക്കാം.", "ഇരിക്കാം", "ഇരിക്കാവുന്നത്"),
        ("നമുക്ക് വേഗത്തിൽ ഗ്രൗണ്ടിലൂടെ ഓടാം.", "ഓടാം", "ഓടാവുന്നത്"),
        ("വിപണിയിൽനിന്ന് ആവശ്യമുള്ള സാധനങ്ങൾ വാങ്ങാം.", "വാങ്ങാം", "വാങ്ങാവുന്നത്"),
        ("അവശർക്ക് സഹായധനം കൊടുക്കാം.", "കൊടുക്കാം", "കൊടുക്കാവുന്നത്"),
        ("യാത്രക്കാർക്ക് സ്റ്റേഷനിൽ കുറച്ചുനേരം നിൽക്കാം.", "നിൽക്കാം", "നിൽക്കാവുന്നത്"),
        ("കുട്ടിക്ക് തറയിൽ ശാന്തമായി ഉറങ്ങാം.", "ഉറങ്ങാം", "ഉറങ്ങാവുന്നത്"),
        ("രാവിലെ തോട്ടത്തിൽ പൂക്കൾ നുള്ളാം.", "നുള്ളാം", "നുള്ളാവുന്നത്"),
        ("അധ്യാപകനോട് സംശയങ്ങൾ ചോദിക്കാം.", "ചോദിക്കാം", "ചോദിക്കാവുന്നത്"),
        ("പുതിയ പുസ്തകം ലൈബ്രറിയിൽനിന്ന് എടുക്കാം.", "എടുക്കാം", "എടുക്കാവുന്നത്"),
        ("പഴയ പെട്ടി തറയിൽ വെക്കാം.", "വെക്കാം", "വെക്കാവുന്നത്"),
        ("കതക് വേഗത്തിൽ തുറക്കാം.", "തുറക്കാം", "തുറക്കാവുന്നത്"),
        ("ജനൽ വൈകുന്നേരം അടക്കാം.", "അടക്കാം", "അടക്കാവുന്നത്"),
        ("തോട്ടത്തിലെ പുല്ലുകൾ ചെത്താം.", "ചെത്താം", "ചെത്താവുന്നത്"),
        ("ഭാരം കുറഞ്ഞ പെട്ടി ഉയർത്താം.", "ഉയർത്താം", "ഉയർത്താവുന്നത്"),
        ("പാഠപുസ്തകങ്ങൾ ശ്രദ്ധയോടെ വായിക്കാം.", "വായിക്കാം", "വായിക്കാവുന്നത്"),
        ("അവിടെനിന്ന് ദൂരദൃശ്യങ്ങൾ നോക്കാം.", "നോക്കാം", "നോക്കാവുന്നത്"),
        ("വഴിയരികിൽ പുതിയ ചെടികൾ നടാം.", "നടാം", "നടാവുന്നത്"),
        ("നദി നീന്തി കടക്കാം.", "കടക്കാം", "കടക്കാവുന്നത്"),
        ("പഴയ വസ്ത്രങ്ങൾ അലക്കാം.", "അലക്കാം", "അലക്കാവുന്നത്"),
        ("കുട്ടികൾക്ക് സന്തോഷത്തോടെ ചിരിക്കാം.", "ചിരിക്കാം", "ചിരിക്കാവുന്നത്"),
        ("നല്ലൊരു സുഹൃത്തിനെ സ്നേഹിക്കാം.", "സ്നേഹിക്കാം", "സ്നേഹിക്കാവുന്നത്"),
        ("കഥകൾ രസകരമായി കേൾക്കാം.", "കേൾക്കാം", "കേൾക്കാവുന്നത്"),
        ("അവൻ വഴിയിൽ വീഴാം.", "വീഴാം", "വീഴാവുന്നത്")
    ],
    "habitual_pos": [
        ("അവൻ എന്നും രാവിലെ ഇവിടെ വരാറുണ്ട്.", "വരാറുണ്ട്", "വരാറുള്ളത്"),
        ("ഞങ്ങൾ വൈകുന്നേരങ്ങളിൽ പാർക്കിൽ പോകാറുണ്ട്.", "പോകാറുണ്ട്", "പോകാറുള്ളത്"),
        ("അമ്മ വിശേഷദിവസങ്ങളിൽ പായസം ചെയ്യാറുണ്ട്.", "ചെയ്യാറുണ്ട്", "ചെയ്യാറുള്ളത്"),
        ("അവർ ആകാശത്ത് നക്ഷത്രങ്ങളെ കാണാറുണ്ട്.", "കാണാറുണ്ട്", "കാണാറുള്ളത്"),
        ("അച്ഛൻ കുട്ടിക്കാലത്തെ കഥകൾ പറയാറുണ്ട്.", "പറയാറുണ്ട്", "പറയാറുള്ളത്"),
        ("അധ്യാപകൻ നല്ല കവിതകൾ എഴുതാറുണ്ട്.", "എഴുതാറുണ്ട്", "എഴുതാറുള്ളത്"),
        ("ലീല സന്ധ്യയ്ക്ക് മനോഹരമായി പാടാറുണ്ട്.", "പാടാറുണ്ട്", "പാടാറുള്ളത്"),
        ("കുട്ടി രാവിലെ ചൂടുപാൽ കുടിക്കാറുണ്ട്.", "കുടിക്കാറുണ്ട്", "കുടിക്കാറുള്ളത്"),
        ("പക്ഷികൾ ധാന്യമണികൾ തിന്നാറുണ്ട്.", "തിന്നാറുണ്ട്", "തിന്നാറുള്ളത്"),
        ("മുത്തശ്ശി ഉമ്മറത്ത് വന്നിരിക്കാറുണ്ട്.", "ഇരിക്കാറുണ്ട്", "ഇരിക്കാറുള്ളത്"),
        ("യുവാക്കൾ മൈതാനത്ത് വേഗത്തിൽ ഓടാറുണ്ട്.", "ഓടാറുണ്ട്", "ഓടാറുള്ളത്"),
        ("അവൻ കടയിൽനിന്ന് പുസ്തകങ്ങൾ വാങ്ങാറുണ്ട്.", "വാങ്ങാറുണ്ട്", "വാങ്ങാറുള്ളത്"),
        ("അവർ പാവപ്പെട്ടവർക്ക് വസ്ത്രങ്ങൾ കൊടുക്കാറുണ്ട്.", "കൊടുക്കാറുണ്ട്", "കൊടുക്കാറുള്ളത്"),
        ("യാത്രക്കാർ ബസ് കാത്ത് നിൽക്കാറുണ്ട്.", "നിൽക്കാറുണ്ട്", "നിൽക്കാറുള്ളത്"),
        ("കുഞ്ഞ് തൊട്ടിലിൽ സുഖമായി ഉറങ്ങാറുണ്ട്.", "ഉറങ്ങാറുണ്ട്", "ഉറങ്ങാറുള്ളത്"),
        ("അമ്മ തോട്ടത്തിലെ മുല്ലപ്പൂക്കൾ നുള്ളാറുണ്ട്.", "നുള്ളാറുണ്ട്", "നുള്ളാറുള്ളത്"),
        ("വിദ്യാർത്ഥികൾ പ്രധാന ചോദ്യങ്ങൾ ചോദിക്കാറുണ്ട്.", "ചോദിക്കാറുണ്ട്", "ചോദിക്കാറുള്ളത്"),
        ("അവൻ അലമാരയിൽനിന്ന് തുണികൾ എടുക്കാറുണ്ട്.", "എടുക്കാറുണ്ട്", "എടുക്കാറുള്ളത്"),
        ("അച്ഛൻ മേശപ്പുറത്ത് താക്കോൽ വെക്കാറുണ്ട്.", "വെക്കാറുണ്ട്", "വെക്കാറുള്ളത്"),
        ("അവർ അതിരാവിലെ പ്രധാന വാതിൽ തുറക്കാറുണ്ട്.", "തുറക്കാറുണ്ട്", "തുറക്കാറുള്ളത്"),
        ("സന്ധ്യയാകുമ്പോൾ ജനലുകൾ അടക്കാറുണ്ട്.", "അടക്കാറുണ്ട്", "അടക്കാറുള്ളത്"),
        ("വേലക്കാരൻ തോട്ടത്തിലെ പുല്ല് ചെത്താറുണ്ട്.", "ചെത്താറുണ്ട്", "ചെത്താറുള്ളത്"),
        ("അവൻ കല്ലുകൾ ദൂരേക്ക് എറിയാറുണ്ട്.", "എറിയാറുണ്ട്", "എറിയാറുള്ളത്"),
        ("അവൾ ദിവസവും പത്രങ്ങൾ വായിക്കാറുണ്ട്.", "വായിക്കാറുണ്ട്", "വായിക്കാറുള്ളത്"),
        ("യാത്രക്കാർ ജാലകത്തിലൂടെ പുറത്തേക്ക് നോക്കാറുണ്ട്.", "നോക്കാറുണ്ട്", "നോക്കാറുള്ളത്"),
        ("കർഷകൻ പാടത്ത് വിത്തുകൾ വിതയ്ക്കാറുണ്ട്.", "വിതയ്ക്കാറുണ്ട്", "വിതയ്ക്കാറുള്ളത്"),
        ("അവർ തോണിയിൽ പുഴ കടക്കാറുണ്ട്.", "കടക്കാറുണ്ട്", "കടക്കാറുള്ളത്"),
        ("അമ്മ തുണികൾ വൃത്തിയായി അലക്കാറുണ്ട്.", "അലക്കാറുണ്ട്", "അലക്കാറുള്ളത്"),
        ("കുട്ടികൾ തമാശകൾ കേട്ട് ചിരിക്കാറുണ്ട്.", "ചിരിക്കാറുണ്ട്", "ചിരിക്കാറുള്ളത്"),
        ("അവർ സുഹൃത്തുക്കളെ മനസ്സാ സ്നേഹിക്കാറുണ്ട്.", "സ്നേഹിക്കാറുണ്ട്", "സ്നേഹിക്കാറുള്ളത്"),
        ("ഞങ്ങൾ ആകാശത്ത് പറവകളെ അറിയാറുണ്ട്.", "അറിയാറുണ്ട്", "അറിയാറുള്ളത്"),
        ("കുട്ടികൾ മുതിർന്നവരുടെ ഉപദേശങ്ങൾ കേൾക്കാറുണ്ട്.", "കേൾക്കാറുണ്ട്", "കേൾക്കാറുള്ളത്")
    ],
    "habitual_neg": [
        ("അവൻ രാത്രികളിൽ ഇവിടെ വരാറില്ല.", "വരാറില്ല", "വരാറില്ലാത്തത്"),
        ("ഞങ്ങൾ ആ തിരക്കുള്ള വഴിയിൽ പോകാറില്ല.", "പോകാറില്ല", "പോകാറില്ലാത്തത്"),
        ("അവർ അനാവശ്യ കാര്യങ്ങൾ ചെയ്യാറില്ല.", "ചെയ്യാറില്ല", "ചെയ്യാറില്ലാത്തത്"),
        ("പകൽസമയത്ത് അവൻ പുറത്തിറങ്ങി കാണാറില്ല.", "കാണാറില്ല", "കാണാറില്ലാത്തത്"),
        ("അവൾ ആരെക്കുറിച്ചും മോശം പറയാറില്ല.", "പറയാറില്ല", "പറയാറില്ലാത്തത്"),
        ("അവൻ കത്തുകൾ സ്വന്തമായി എഴുതാറില്ല.", "എഴുതാറില്ല", "എഴുതാറില്ലാത്തത്"),
        ("ലീല സദസ്സിൽ പാട്ടുകൾ പാടാറില്ല.", "പാടാറില്ല", "പാടാറില്ലാത്തത്"),
        ("കുട്ടി കട്ടൻചായ കുടിക്കാറില്ല.", "കുടിക്കാറില്ല", "കുടിക്കാറില്ലാത്തത്"),
        ("അവർ പഴയ ആഹാരം തിന്നാറില്ല.", "തിന്നാറില്ല", "തിന്നാറില്ലാത്തത്"),
        ("മുത്തച്ഛൻ വെയിലത്ത് ഇരിക്കാറില്ല.", "ഇരിക്കാറില്ല", "ഇരിക്കാറില്ലാത്തത്"),
        ("രോഗികൾ കഠിനമായി ഓടാറില്ല.", "ഓടാറില്ല", "ഓടാറില്ലാത്തത്"),
        ("അവൻ അനാവശ്യ സാധനങ്ങൾ വാങ്ങാറില്ല.", "വാങ്ങാറില്ല", "വാങ്ങാറില്ലാത്തത്"),
        ("അവർ ശത്രുക്കൾക്ക് രഹസ്യങ്ങൾ കൊടുക്കാറില്ല.", "കൊടുക്കാറില്ല", "കൊടുക്കാറില്ലാത്തത്"),
        ("ആരും വെയിലത്ത് അധികനേരം നിൽക്കാറില്ല.", "നിൽക്കാറില്ല", "നിൽക്കാറില്ലാത്തത്"),
        ("ശബ്ദമുള്ളപ്പോൾ കുഞ്ഞ് ഉറങ്ങാറില്ല.", "ഉറങ്ങാറില്ല", "ഉറങ്ങാറില്ലാത്തത്"),
        ("അവർ പൂന്തോട്ടത്തിലെ പൂക്കൾ നുള്ളാറില്ല.", "നുള്ളാറില്ല", "നുള്ളാറില്ലാത്തത്"),
        ("ശിശുക്കൾ കാര്യമായ ചോദ്യങ്ങൾ ചോദിക്കാറില്ല.", "ചോദിക്കാറില്ല", "ചോദിക്കാറില്ലാത്തത്"),
        ("അനുവാദമില്ലാതെ അവൻ പണം എടുക്കാറില്ല.", "എടുക്കാറില്ല", "എടുക്കാറില്ലാത്തത്"),
        ("അവൻ സാധനങ്ങൾ നിലത്ത് വെക്കാറില്ല.", "വെക്കാറില്ല", "വെക്കാറില്ലാത്തത്"),
        ("രാത്രിയിൽ ആരും ആ വാതിൽ തുറക്കാറില്ല.", "തുറക്കാറില്ല", "തുറക്കാറില്ലാത്തത്"),
        ("പകൽസമയത്ത് ജനലുകൾ അടക്കാറില്ല.", "അടക്കാറില്ല", "അടക്കാറില്ലാത്തത്"),
        ("അവൻ ആവശ്യമില്ലാതെ കാടുകൾ ചെത്താറില്ല.", "ചെത്താറില്ല", "ചെത്താറില്ലാത്തത്"),
        ("കുട്ടികൾ കിണറ്റിലേക്ക് കല്ലുകൾ എറിയാറില്ല.", "എറിയാറില്ല", "എറിയാറില്ലാത്തത്"),
        ("അവൾ പുസ്തകങ്ങൾ അപൂർവ്വമായല്ലാതെ വായിക്കാറില്ല.", "വായിക്കാറില്ല", "വായിക്കാറില്ലാത്തത്"),
        ("അവർ അന്യരുടെ കാര്യങ്ങളിലേക്ക് നോക്കാറില്ല.", "നോക്കാറില്ല", "നോക്കാറില്ലാത്തത്"),
        ("മഴയില്ലാത്തപ്പോൾ വിത്ത് വിതയ്ക്കാറില്ല.", "വിതയ്ക്കാറില്ല", "വിതയ്ക്കാറില്ലാത്തത്"),
        ("രാത്രിയിൽ ആരും ആ പുഴ കടക്കാറില്ല.", "കടക്കാറില്ല", "കടക്കാറില്ലാത്തത്"),
        ("അവൻ വസ്ത്രങ്ങൾ സ്വയം അലക്കാറില്ല.", "അലക്കാറില്ല", "അലക്കാറില്ലാത്തത്"),
        ("വിഷമഘട്ടങ്ങളിൽ ആരും ചിരിക്കാറില്ല.", "ചിരിക്കാറില്ല", "ചിരിക്കാറില്ലാത്തത്"),
        ("അവർ അഹങ്കാരികളെ സ്നേഹിക്കാറില്ല.", "സ്നേഹിക്കാറില്ല", "സ്നേഹിക്കാറില്ലാത്തത്"),
        ("അവൻ ആ രഹസ്യം അറിയാറില്ല.", "അറിയാറില്ല", "അറിയാറില്ലാത്തത്"),
        ("അവർ അന്യരുടെ വാക്കുകൾ കേൾക്കാറില്ല.", "കേൾക്കാറില്ല", "കേൾക്കാറില്ലാത്തത്")
    ]
}

def run_empirical_audit():
    analyser = Analyser()
    generator = Generator()
    
    audit_results = {}
    
    for category, items in corpus.items():
        cat_stats = {
            "total": len(items),
            "mlmorph_only_success": 0,
            "mlmorph_only_fail": 0,
            "fallback_success_after_fail": 0,
            "fallback_incorrect": 0,
            "fallback_never_triggered": 0,
            "direct_validation_success": 0,
            "hook_validation_success": 0,
            "rows": []
        }
        
        for sentence, verb, expected in items:
            # 1. Analyse verb using scored analysis
            raw_analyses = analyser.analyse(verb)
            def _score(entry):
                raw, weight = entry
                return weight + (-200 if "<v>" in raw else 0)
            sorted_raw = sorted(raw_analyses, key=_score)
            best_raw = sorted_raw[0][0] if sorted_raw else "NO_ANALYSIS"
            va = VerbAnalysis(best_raw)
            lemma = va.lemma
            
            # --- EVALUATE PRIMARY MLMORPH PATH ---
            mlmorph_primary_res = None
            primary_target = ""
            
            if category == "permissive":
                primary_target = f"{lemma}<v><permissive-mood>"
                gen_res = generator.generate(primary_target)
                if gen_res:
                    for form, _ in gen_res:
                        if form.endswith("ാവുന്നത്"):
                            mlmorph_primary_res = form
                            break
                        elif form.endswith("ാവുന്നതാണ്") or form.endswith("ാവുന്നതാണു്"):
                            mlmorph_primary_res = re.sub(r"താണ്$|താണു്$", "ത്", form)
                            break
                            
            elif category == "habitual_pos":
                primary_target = f"{lemma}<v><habitual-aspect>"
                gen_res = generator.generate(primary_target)
                if gen_res:
                    stem = gen_res[0][0]
                    if stem.endswith("്"):
                        mlmorph_primary_res = stem[:-1] + "ുള്ളത്"
                    else:
                        mlmorph_primary_res = stem + "ുള്ളത്"
                        
            elif category == "habitual_neg":
                primary_target = f"{lemma}<v><habitual-aspect>"
                gen_res = generator.generate(primary_target)
                if gen_res:
                    stem = gen_res[0][0]
                    if stem.endswith("്"):
                        mlmorph_primary_res = stem[:-1] + "ില്ലാത്തത്"
                    else:
                        mlmorph_primary_res = stem + "ില്ലാത്തത്"

            mlmorph_only_ok = (mlmorph_primary_res == expected)
            
            # --- EVALUATE FALLBACK PATH ---
            fallback_triggered = (mlmorph_primary_res is None)
            fallback_res = None
            if fallback_triggered:
                if category == "permissive":
                    fallback_res = verb[:-2] + "ാവുന്നത്" if verb.endswith("ാം") else verb + "ാവുന്നത്"
                elif category == "habitual_pos":
                    fallback_res = verb[:-6] + "ാറുള്ളത്" if verb.endswith("ാറുണ്ട്") else verb + "ഉള്ളത്"
                elif category == "habitual_neg":
                    fallback_res = verb[:-6] + "ാറില്ലാത്തത്" if verb.endswith("ാറില്ല") else verb + "ഇല്ലാത്തത്"

            final_form = mlmorph_primary_res if mlmorph_primary_res else fallback_res
            
            # --- VALIDATION EVALUATION ---
            direct_reanalysis = analyser.analyse(final_form)
            direct_valid = False
            for raw, _ in direct_reanalysis:
                if "<n><deriv>" in raw and ("<adv-clause-rp-" in raw or "<cvb-adv-part-" in raw):
                    direct_valid = True
                    break
            
            hook_valid, hook_raw = _is_valid_nominalized_verb(final_form, analyser)

            if mlmorph_only_ok:
                cat_stats["mlmorph_only_success"] += 1
                cat_stats["fallback_never_triggered"] += 1
            else:
                cat_stats["mlmorph_only_fail"] += 1
                if fallback_res == expected:
                    cat_stats["fallback_success_after_fail"] += 1
                else:
                    cat_stats["fallback_incorrect"] += 1
                    
            if direct_valid:
                cat_stats["direct_validation_success"] += 1
            if hook_valid:
                cat_stats["hook_validation_success"] += 1
                
            cat_stats["rows"].append({
                "sentence": sentence,
                "input_verb": verb,
                "expected": expected,
                "mlmorph_analysis": best_raw,
                "lemma": lemma,
                "primary_target": primary_target,
                "primary_result": mlmorph_primary_res,
                "mlmorph_only_success": mlmorph_only_ok,
                "fallback_triggered": fallback_triggered,
                "fallback_result": fallback_res,
                "final_form": final_form,
                "direct_validation_valid": direct_valid,
                "hook_validation_valid": hook_valid,
                "hook_analysis": hook_raw,
                "is_correct": (final_form == expected)
            })
            
        audit_results[category] = cat_stats
        
    out_file = os.path.join("Datasets", "verb_normalizer_empirical_fallback_audit.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, ensure_ascii=False, indent=2)
        
    print(f"Empirical audit completed. Results saved to {out_file}\n")
    
    # Print summary decision table
    print("=" * 110)
    print("EMPIRICAL FALLBACK AUDIT RESULTS (32 SENTENCES PER CATEGORY)")
    print("=" * 110)
    print(f"{'Category':<24} | {'Cases':<6} | {'MLMorph Alone':<14} | {'Fallback Trig':<14} | {'Direct Valid':<14} | {'Hook Valid':<12}")
    print("-" * 110)
    for cat, stats in audit_results.items():
        print(f"{cat:<24} | {stats['total']:<6} | {stats['mlmorph_only_success']}/{stats['total']:<12} | {stats['total'] - stats['fallback_never_triggered']}/{stats['total']:<12} | {stats['direct_validation_success']}/{stats['total']:<12} | {stats['hook_validation_success']}/{stats['total']:<10}")
    print("=" * 110)

if __name__ == "__main__":
    run_empirical_audit()
