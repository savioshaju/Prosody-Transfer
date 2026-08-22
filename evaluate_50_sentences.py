import sys
import os
import json
import re

# Ensure UTF-8 output encoding for Malayalam characters on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ssf_pipeline.cleft_pipeline import CleftPipeline
from mlmorph import Analyser

test_corpus = [
    "അവൻ <FF>ഇന്നലെ</FF> ഇവിടെ വന്നു.",
    "രാമൻ <FF>പുസ്തകം</FF> മേശപ്പുറത്ത് വെച്ചു.",
    "ഞാൻ ആ സമ്മാനം <FF>അമ്മയ്ക്ക്</FF> കൊടുത്തു.",
    "കാട്ടിൽ വെച്ച് വേട്ടക്കാരൻ <FF>ആനയെ</FF> കണ്ടു.",
    "പൂച്ച <FF>ഫ്രിഡ്ജിൽ</FF> കയറി ഇരുന്നു.",
    "ഞങ്ങൾ <FF>നാളെ</FF> എറണാകുളത്തേക്ക് പോകും.",
    "അച്ഛൻ തോട്ടത്തിൽ നിന്ന് <FF>തേൻ</FF> ശേഖരിച്ചു.",
    "കുട്ടി കളിപ്പാട്ടം തറയിൽ <FF>ഇട്ടു</FF> പൊട്ടിച്ചു.",
    "അവൻ മരം വെട്ടിയത് <FF>കോടാലികൊണ്ട്</FF> ആണ്.",
    "ഈ പുതിയ പേന <FF>എന്റെ</FF> ആണ്.",
    "ലീല ജനക്കൂട്ടത്തോട് <FF>ധൈര്യത്തോടെ</FF> സംസാരിച്ചു.",
    "പൂച്ചകൾ എലികളെ പിടിക്കുന്നത് <FF>പെട്ടെന്ന്</FF> ആണ്.",
    "വഴിയിൽ വെച്ച് ഞാൻ <FF>രാഹുലിനെ</FF> കണ്ടുമുട്ടി.",
    "ആ പെൺകുട്ടി പാട്ട് <FF>മനോഹരമായി</FF> പാടി.",
    "ഞാൻ ആ കത്ത് <FF>അമ്മയോട്</FF> ചോദിച്ച് വാങ്ങി.",
    "അവർ കളിസ്ഥലത്തുനിന്ന് <FF>വീട്ടിലേക്ക്</FF> ഓടിവന്നു.",
    "അവൻ കടയിൽനിന്നും <FF>നെൽ</FF> വാങ്ങി വന്നു.",
    "രാഘവൻ കിണറ്റിലേക്ക് <FF>കൽ</FF> എറിഞ്ഞു.",
    "അവൾ ആഭരണങ്ങൾ ഉണ്ടാക്കാൻ <FF>പൊൻ</FF> എടുത്തു.",
    "ആട്ടിടയൻ മലമുകളിൽ നല്ല <FF>പുൽ</FF> കണ്ടു.",
    "അധ്യാപകൻ <FF>വിദ്യാർത്ഥിയെ</FF> ക്ലാസിൽ വെച്ച് ശകാരിച്ചു.",
    "കുട്ടികൾ മുറ്റത്ത് ഒരു <FF>ചുവന്നപ്പൂവ്</FF> കണ്ടു.",
    "പൂച്ച മേശപ്പുറത്തിരുന്ന <FF>പാൽ</FF> കുടിച്ചു തീർത്തു.",
    "അവൻ നടക്കുമ്പോൾ അവന്റെ <FF>കാൽ</FF> തട്ടി വീണു.",
    "കപ്പൽ വലിയ <FF>കടലിൽ</FF> വച്ച് തകർന്നുപോയി.",
    "പക്ഷി തൻ്റെ മനോഹരമായ <FF>തൂവൽ</FF> കൊഴിച്ചിട്ടു.",
    "രാമൻ പുതിയ വണ്ടി വാങ്ങിയത് <FF>അടുത്തവർഷം</FF> ആണ്.",
    "കാർ വേഗത്തിൽ ഓടിച്ചത് <FF>അവൻ</FF> ആണ്.",
    "പോലീസ് കള്ളനെ പിടിച്ചത് <FF>ഇവിടെ</FF> വെച്ചാണ്.",
    "അമ്മ കുട്ടിയെ പാട്ടുപാടി <FF>ഉറക്കി</FF> കിടത്തി.",
    "അവൻ തൻ്റെ കൃഷിയിടത്തിൽ ആടുകളെ <FF>വളർത്തി</FF>.",
    "ഞങ്ങൾ താക്കോലുകൾ തിരഞ്ഞത് <FF>മേശപ്പുറത്ത്</FF> ആണ്.",
    "കാട്ടിൽ വച്ച് കടുവ <FF>ആടിനെ</FF> പിടിച്ചു കൊന്നു.",
    "കുട്ടികൾ തങ്ങളുടെ പ്രിയപ്പെട്ട <FF>ഗുരുവിനെ</FF> വണങ്ങി.",
    "രാമൻ്റെ വീട് വളരെ <FF>വലുത്</FF> ആണ്.",
    "അവൻ കടയിൽനിന്ന് കുറെ <FF>പച്ചക്കായ</FF> വാങ്ങി.",
    "കാക്ക ദാഹിച്ചപ്പോൾ കുറേ <FF>വെള്ളം</FF> കുടിച്ചു.",
    "അമ്മ മകന് നല്ല <FF>ഭക്ഷണം</FF> പാകം ചെയ്തു.",
    "പെൺകുട്ടി സദസ്സിൽ വച്ച് മനോഹരമായി <FF>നൃത്തം</FF> ചെയ്തു.",
    "ലീലയ്ക്ക് പുതിയ പരീക്ഷയിൽ നല്ല <FF>വിജയം</FF> ലഭിച്ചു.",
    "അവൻ ഒരു പുതിയ കട തുടങ്ങി കുറേ <FF>പണം</FF> ഉണ്ടാക്കി.",
    "അമ്മ കുട്ടിയോട് അവിടെ <FF>ഇരിക്കാൻ</FF> പറഞ്ഞു.",
    "അവൻ ആ ജോലി <FF>ചെയ്യാം</FF> എന്ന് സമ്മതിച്ചു.",
    "കുട്ടി സ്കൂളിൽ പോകാൻ മടിച്ച് <FF>കരഞ്ഞു</FF>.",
    "പ്രളയം കാരണം എൻ്റെ കപ്പൽ യാത്ര <FF>വൈകി</FF>.",
    "കുഞ്ഞ് മുറിയിൽ കിടന്ന് സുഖമായി <FF>ഉറങ്ങി</FF>.",
    "മഴ പെയ്തപ്പോൾ തോട്ടത്തിലെ പൂക്കൾ <FF>വിരിഞ്ഞു</FF>.",
    "രാമു തൻ്റെ വലിയ പെട്ടിയിൽ പണം <FF>ഒളിപ്പിച്ചു</FF>.",
    "വിമാനം നാലുമണിക്ക് നെടുമ്പാശ്ശേരിയിൽ <FF>എത്തി</FF>.",
    "അവർ ഇന്നലെ കായലിൽ ഒരു പുതിയ <FF>തോണി</FF> കണ്ടു."
]

def run_evaluation():
    pipeline = CleftPipeline()
    raw_analyser = Analyser()
    
    results = []
    
    for idx, marked_sent in enumerate(test_corpus, 1):
        clean_sent = re.sub(r"</?FF>", "", marked_sent).strip()
        focus_match = re.search(r"<FF>(.*?)</FF>", marked_sent)
        focus_word = focus_match.group(1).strip() if focus_match else ""
        
        # Run raw mlmorph analysis on the focus word
        mlmorph_analyses = raw_analyser.analyse(focus_word)
        mlmorph_analyses_str = [a[0] for a in mlmorph_analyses] if mlmorph_analyses else []
        
        # Run cleft pipeline on marked sentence
        cleft_res = pipeline.process(marked_sent)
        
        # Determine status
        is_success = cleft_res.status in ("VALID", "PE_ROUTED")
        failure_reason = cleft_res.error if not is_success else ""
        
        # Categorize host word ending & properties
        properties = []
        if focus_word.endswith("ം"):
            properties.append("ം-final")
        elif focus_word.endswith(("ൻ", "ൺ")):
            properties.append("ൻ/ൺ-final")
        elif focus_word.endswith(("ൽ", "ൾ")):
            properties.append("ൽ/ൾ-final")
        elif focus_word.endswith("ർ"):
            properties.append("ർ-final")
        elif any(focus_word.endswith(v) for v in ("ി", "ീ", "െ", "േ", "ൈ", "്യ")):
            properties.append("palatal-vowel-final")
        elif any(focus_word.endswith(v) for v in ("ു", "ൂ", "ൊ", "ോ", "ൌ", "വ്")):
            properties.append("rounded-vowel-final")
            
        if any(focus_word.endswith(sfx) for sfx in ("െ", "നെ", "യെ", "ക്ക്", "ന്", "ൽ", "ിൽ", "ത്ത്", "റ്റിൽ", "കൊണ്ട്", "യോട്", "ഓട്", "ന്റെ", "്റെ", "ുടെ", "നിന്ന്")):
            properties.append("case-marked")
        if focus_word in ("അവൻ", "അവൾ", "അവർ", "ഞാൻ", "നീ", "അത്", "ഇത്", "എന്റെ", "തന്റെ"):
            properties.append("pronoun")
        if focus_word in ("ഇന്നലെ", "നാളെ", "ഇവിടെ", "പെട്ടെന്ന്", "മനോഹരമായി", "ധൈര്യത്തോടെ"):
            properties.append("adverb/temporal/manner")
        if any(focus_word.endswith(v) for v in ("ഉക", "ുക", "ി", "ു", "ാൻ", "ാം")):
            properties.append("verb/participle")
            
        record = {
            "id": idx,
            "original_sentence": clean_sent,
            "marked_sentence": marked_sent,
            "focus_word": focus_word,
            "properties": properties,
            "mlmorph_has_analysis": bool(mlmorph_analyses),
            "mlmorph_analyses": mlmorph_analyses_str,
            "copula_form": cleft_res.copula_form,
            "copula_path": cleft_res.copula_path,
            "constituent_type": cleft_res.constituent_type,
            "strategy_route": cleft_res.route,
            "main_verb": cleft_res.main_verb,
            "normalized_verb": cleft_res.normalized_verb,
            "final_cleft_sentence": cleft_res.cleft_sentence,
            "status": cleft_res.status,
            "is_success": is_success,
            "failure_reason": failure_reason
        }
        results.append(record)

    # Save to file
    out_file = os.path.join("Datasets", "cleft_50_diagnostic_evaluation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Evaluation complete for 50 sentences. Results saved to {out_file}")
    
    # Print summary statistics
    total = len(results)
    success = sum(1 for r in results if r['is_success'])
    failure = sum(1 for r in results if not r['is_success'])
    mlmorph_fail = sum(1 for r in results if not r['mlmorph_has_analysis'])
    
    print("\n" + "="*80)
    print("50-SENTENCE CLEFTING DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"Total tests       : {total}")
    print(f"SUCCESS           : {success}")
    print(f"FAILURE           : {failure}")
    print(f"MLMorph NO ANALYS : {mlmorph_fail}")
    print("="*80)
    
    print("\nFAILURES BREAKDOWN:")
    for r in results:
        if not r['is_success']:
            print(f"[ID {r['id']:02d}] Focus: '{r['focus_word']}' | Status: {r['status']} | Reason: {r['failure_reason']}")

if __name__ == "__main__":
    run_evaluation()
