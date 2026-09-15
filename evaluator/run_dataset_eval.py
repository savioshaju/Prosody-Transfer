import re
import json
import time
import os
import sys
import pandas as pd
eval_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(eval_dir)
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from cleft.awesome_cleft_pipeline import AwesomeCleftPipeline
try:
    from evaluator.export_mismatches import clean_no_punct
except ImportError:
    from export_mismatches import clean_no_punct


def normalize_text(text):
    return clean_no_punct(text)

def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    eval_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(eval_dir)
    if workspace_dir not in sys.path:
        sys.path.insert(0, workspace_dir)

    csv_path = os.path.join(workspace_dir, "Synthetic_data - Sheet1.csv")
    df = pd.read_csv(csv_path)

    print(f"Starting evaluation on {len(df)} rows from {csv_path} with SimAlign (itermax)...", flush=True)
    pipeline = AwesomeCleftPipeline(aligner_type="simalign", aligner_method="itermax")

    results = []
    t0 = time.time()

    for idx, row in df.iterrows():
        eng = str(row['eng']) if pd.notna(row['eng']) else ''
        mal = str(row['mal']) if pd.notna(row['mal']) else ''
        focus = str(row['Source_Prosody_word']) if pd.notna(row['Source_Prosody_word']) else ''
        target = str(row['Target']) if pd.notna(row['Target']) else ''

        try:
            res = pipeline.process(
                english_sentence=eng,
                malayalam_sentence=mal,
                english_focus=focus,
            )

            cleft_dict = res.get('cleft_pipeline_output', {}) or {}
            cleft_out = res.get('emphasized_malayalam_sentence', '') or ''

            prev_reord = res.get('preverbal_focus_reordering', {}) or {}
            prev_app = bool(prev_reord.get('applicable', False))
            prev_sent = prev_reord.get('reordered_sentence', '') if prev_app else None

            # Strict Cleft completion: must have aanu attached AND verb normalised
            status = cleft_dict.get('status', '')
            copula_form = cleft_dict.get('copula_form', '') or cleft_dict.get('aanu_attachment', {}).get('attached_form', '')
            norm_verb = cleft_dict.get('nominalized_verb', '')
            main_verb = cleft_dict.get('main_verb', '')
            has_copula = any(cop in copula_form for cop in ("ാണ്", "ആണ്", "യാണ്", "മായാണ്", "ാൺ", "വാൺ", "ആയിട്ടാണ്")) or any(cop in cleft_out for cop in ("ാണ്", "ആണ്", "യാണ്", "മായാണ്", "ാൺ", "വാൺ", "ആയിട്ടാണ്"))
            has_norm_verb = bool(
                norm_verb and norm_verb != main_verb and any(norm_verb.endswith(sfx) for sfx in ("ത്", "തു്", "ച്ചത്", "ട്ടത്", "ന്നത്", "ത്തത്", "ുന്നത്"))
            ) or (not main_verb and has_copula) or bool(norm_verb and any(norm_verb.endswith(sfx) for sfx in ("ത്", "തു്", "ച്ചത്", "ട്ടത്", "ന്നത്", "ത്തത്", "ുന്നത്")))

            cleft_done = bool(status == 'VALID' and cleft_out and cleft_out != mal and '<PE>' not in cleft_out and has_copula and has_norm_verb)
            
            # Preverbal reordering check
            prev_reord_done = prev_app and bool(prev_sent)

            candidate_outputs = {}
            if cleft_done:
                candidate_outputs['cleft_emphasized'] = cleft_out
            if prev_sent:
                candidate_outputs['preverbal_reordered'] = prev_sent

            # Target matching
            norm_target = normalize_text(target)

            exact_match_found = False
            exact_matched_types = []
            norm_match_found = False
            norm_matched_types = []

            for out_name, out_val in candidate_outputs.items():
                if out_val == target:
                    exact_match_found = True
                    exact_matched_types.append(out_name)
                if normalize_text(out_val) == norm_target:
                    norm_match_found = True
                    norm_matched_types.append(out_name)

            results.append({
                'index': idx,
                'eng': eng,
                'mal': mal,
                'focus': focus,
                'target': target,
                'cleft_done': cleft_done,
                'prev_reord_done': prev_reord_done,
                'cleft_reord_done': False,
                'both_cleft_and_prev_reord': cleft_done and prev_reord_done,
                'both_cleft_and_cleft_reord': False,
                'either_cleft_or_prev_reord': cleft_done or prev_reord_done,
                'either_cleft_or_any_reord': cleft_done or prev_reord_done,
                'exact_match': exact_match_found,
                'exact_matched_types': exact_matched_types,
                'norm_match': norm_match_found,
                'norm_matched_types': norm_matched_types,
                'candidate_outputs': candidate_outputs,
                'proj_focus': res.get('focused_malayalam_constituent', ''),
            })
        except Exception as e:
            results.append({
                'index': idx,
                'eng': eng,
                'mal': mal,
                'focus': focus,
                'target': target,
                'error': str(e),
                'cleft_done': False,
                'const_reord_done': False,
                'cleft_reord_done': False,
                'both_cleft_and_const_reord': False,
                'both_cleft_and_cleft_reord': False,
                'either_cleft_or_const_reord': False,
                'either_cleft_or_any_reord': False,
                'exact_match': False,
                'exact_matched_types': [],
                'norm_match': False,
                'norm_matched_types': [],
                'candidate_outputs': {},
                'proj_focus': '',
            })

        eval_json_path = os.path.join(workspace_dir, 'eval_results.json')
        with open(eval_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        match_tag = "MATCH!" if norm_match_found else "MISMATCH"
        print(f"[{idx + 1}/{len(df)}] Row {idx + 1}: {match_tag} ({time.time() - t0:.1f}s) - Proj: '{res.get('focused_malayalam_constituent', '')}'", flush=True)

    # Save to JSON and CSV
    eval_csv_path = os.path.join(workspace_dir, 'eval_results.csv')

    flat_records = []
    for r in results:
        flat_records.append({
            'index': r['index'],
            'eng': r['eng'],
            'mal': r['mal'],
            'focus': r['focus'],
            'proj_focus': r.get('proj_focus', ''),
            'target': r['target'],
            'cleft_done': r['cleft_done'],
            'prev_reord_done': r['prev_reord_done'],
            'exact_match': r['exact_match'],
            'exact_matched_types': ', '.join(r['exact_matched_types']),
            'norm_match': r['norm_match'],
            'norm_matched_types': ', '.join(r['norm_matched_types']),
            'cleft_emphasized': r['candidate_outputs'].get('cleft_emphasized', ''),
            'preverbal_reordered': r['candidate_outputs'].get('preverbal_reordered', ''),
        })
    pd.DataFrame(flat_records).to_csv(eval_csv_path, index=False, encoding='utf-8-sig')

    total = len(results)
    exact_matches = sum(1 for r in results if r['exact_match'])
    norm_matches = sum(1 for r in results if r['norm_match'])
    cleft_cnt = sum(1 for r in results if r['cleft_done'])
    prev_reord_cnt = sum(1 for r in results if r['prev_reord_done'])
    both_cleft_prev = sum(1 for r in results if r.get('both_cleft_and_prev_reord', False))
    either_cleft_prev = sum(1 for r in results if r.get('either_cleft_or_prev_reord', False))

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Sentences Evaluated               : {total}")
    print(f"Can do Clefting                        : {cleft_cnt} ({cleft_cnt / total * 100:.2f}%)")
    print(f"Can do Preverbal Focus Reordering      : {prev_reord_cnt} ({prev_reord_cnt / total * 100:.2f}%)")
    print(f"Can do Both (Clefting + Preverbal)     : {both_cleft_prev} ({both_cleft_prev / total * 100:.2f}%)")
    print(f"Can do Either (Clefting OR Preverbal)  : {either_cleft_prev} ({either_cleft_prev / total * 100:.2f}%)")
    print("-" * 60)
    print(f"Strict Exact Match to Target           : {exact_matches} ({exact_matches / total * 100:.2f}%)")
    print(f"Normalized Match (ignoring punct/space): {norm_matches} ({norm_matches / total * 100:.2f}%)")
    print("=" * 60)

if __name__ == '__main__':
    main()
