import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure UTF-8 output encoding for Malayalam characters on Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from cleft.copula_pipeline import AnalysisLayer, CopulaLayer

def print_formatted_output(word, selected, status, mlmorph_copula, copula_path):
    if status == "UNRESOLVED":
        lemma = "None"
        pos = "None"
        case = "None"
        features = "None"
        generated_form = "NOT SUPPORTED"
    else:
        lemma = selected.lemma
        pos = selected.pos
        case = selected.case if selected.case else "None"
        
        feat_list = []
        if selected.number:
            feat_list.append(selected.number)
        if hasattr(selected, 'person') and selected.person:
            feat_list.append(selected.person)
        if selected.other_features:
            feat_list.append(selected.other_features)
        features = "|".join(feat_list) if feat_list else "None"
        
        if copula_path == "already-copular":
            generated_form = word
        elif copula_path == "not-supported":
            generated_form = "NOT SUPPORTED"
        else:
            generated_form = mlmorph_copula

    print("--------------------------------------------------")
    print(f"Original               : {word}")
    print(f"Lemma                  : {lemma}")
    print(f"POS                    : {pos}")
    print(f"Case                   : {case}")
    print(f"Morphological features : {features}")
    print(f"Selected Copula Path   : {copula_path}")
    print(f"Generated Form         : {generated_form}")
    print(f"Status                 : {status}")
    print("--------------------------------------------------")

def run_pipeline_for_word(word, analysis_layer, copula_layer):
    # Stage 1: Analysis Layer
    analyses, status = analysis_layer.analyze_word(word)
    
    if status == "UNRESOLVED":
        print_formatted_output(word, None, status, "NOT SUPPORTED", "not-supported")
        return
        
    if status == "AMBIGUOUS":
        print(f"!!! AMBIGUOUS ANALYSES FOR '{word}':")
        for a in analyses:
            print(f"  - {a.raw_analysis}")
        selected = analyses[0]
    else:
        selected = analyses[0]

    # Stage 2: Copula Layer
    mlmorph_copula, preserved, copula_path = copula_layer.transform_word(word, selected)
    
    if preserved == "NO":
        print(f"WARNING: CASE REALIZATION CHANGED")
        print(f"Original: {word}")
        print(f"mlmorph : {mlmorph_copula}")
        
    print_formatted_output(word, selected, status, mlmorph_copula, copula_path)

def main():
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) > 1:
        word = " ".join(sys.argv[1:]).strip()
    else:
        print("Usage: python test_word_copula.py \"<Malayalam Word>\"")
        sys.exit(1)

    print("\n==============================================")
    print("--- Malayalam Word Copula Pipeline ---")
    print("==============================================")
    analysis_layer = AnalysisLayer()
    copula_layer = CopulaLayer()
    run_pipeline_for_word(word, analysis_layer, copula_layer)


if __name__ == "__main__":
    main()

