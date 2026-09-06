import re
import json
import time
import os
import sys
import pandas as pd
from awesome_cleft_pipeline import AwesomeCleftPipeline

def normalize_text(text):
    if not text or not isinstance(text, str):
        return ""
    # Remove zero-width characters, extra spaces, trailing/leading punctuation
    t = text.strip()
    t = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', t)
    t = re.sub(r'\s+', ' ', t)
    # Remove common sentence-ending punctuation for soft match comparison
    t_clean = re.sub(r'[.,;!?\"\'“”‘’]+$', '', t).strip()
    t_clean = re.sub(r'^[.,;!?\"\'“”‘’]+', '', t_clean).strip()
    return t_clean

def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Dataset-gen - Sheet1.csv")
    df = pd.read_csv(csv_path)

    print(f"Starting evaluation on {len(df)} rows from {csv_path} with SimAlign (itermax)...")
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
            cleft_reord = res.get('cleft_reorderings', {}) or {}
            prev_f = cleft_reord.get('preverbal_focus', '') or ''
            postv_f = cleft_reord.get('postverbal_focus', '') or ''
            clause_init_f = cleft_reord.get('clause_initial_focus', '') or ''

            const_reord = res.get('constituency_reordering', {}) or {}
            const_app = bool(const_reord.get('applicable', False))
            const_sent = const_reord.get('reordered_sentence', '') if const_app else None

            # Cleft done check
            cleft_done = bool(cleft_dict.get('route') == 'CLEFT' or (cleft_out and cleft_out != mal and '<PE>' not in cleft_out))
            
            # Reordering checks
            const_reord_done = const_app and bool(const_sent)
            cleft_reord_done = bool(prev_f or postv_f or clause_init_f)
            reord_any_done = const_reord_done or cleft_reord_done

            candidate_outputs = {}
            if cleft_out:
                candidate_outputs['cleft_emphasized'] = cleft_out
            if prev_f:
                candidate_outputs['cleft_preverbal'] = prev_f
            if postv_f:
                candidate_outputs['cleft_postverbal'] = postv_f
            if clause_init_f:
                candidate_outputs['cleft_clause_initial'] = clause_init_f
            if const_sent:
                candidate_outputs['constituency_reordered'] = const_sent

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
                'const_reord_done': const_reord_done,
                'cleft_reord_done': cleft_reord_done,
                'both_cleft_and_const_reord': cleft_done and const_reord_done,
                'both_cleft_and_cleft_reord': cleft_done and cleft_reord_done,
                'either_cleft_or_const_reord': cleft_done or const_reord_done,
                'either_cleft_or_any_reord': cleft_done or reord_any_done,
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

        if (idx + 1) % 10 == 0 or idx == len(df) - 1:
            print(f"Processed {idx + 1}/{len(df)} rows in {time.time() - t0:.1f}s")

    # Save to JSON and CSV
    eval_json_path = os.path.join(base_dir, 'eval_results.json')
    eval_csv_path = os.path.join(base_dir, 'eval_results.csv')
    with open(eval_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

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
            'const_reord_done': r['const_reord_done'],
            'cleft_reord_done': r['cleft_reord_done'],
            'exact_match': r['exact_match'],
            'exact_matched_types': ', '.join(r['exact_matched_types']),
            'norm_match': r['norm_match'],
            'norm_matched_types': ', '.join(r['norm_matched_types']),
            'cleft_emphasized': r['candidate_outputs'].get('cleft_emphasized', ''),
            'cleft_preverbal': r['candidate_outputs'].get('cleft_preverbal', ''),
            'cleft_postverbal': r['candidate_outputs'].get('cleft_postverbal', ''),
            'cleft_clause_initial': r['candidate_outputs'].get('cleft_clause_initial', ''),
            'constituency_reordered': r['candidate_outputs'].get('constituency_reordered', ''),
        })
    pd.DataFrame(flat_records).to_csv(eval_csv_path, index=False, encoding='utf-8-sig')

    total = len(results)
    exact_matches = sum(1 for r in results if r['exact_match'])
    norm_matches = sum(1 for r in results if r['norm_match'])
    cleft_cnt = sum(1 for r in results if r['cleft_done'])
    const_reord_cnt = sum(1 for r in results if r['const_reord_done'])
    cleft_reord_cnt = sum(1 for r in results if r['cleft_reord_done'])
    reord_any_cnt = sum(1 for r in results if r['const_reord_done'] or r['cleft_reord_done'])
    both_cleft_const = sum(1 for r in results if r['both_cleft_and_const_reord'])
    both_cleft_cleft_reord = sum(1 for r in results if r['both_cleft_and_cleft_reord'])
    either_cleft_const = sum(1 for r in results if r['either_cleft_or_const_reord'])
    either_cleft_any_reord = sum(1 for r in results if r['either_cleft_or_any_reord'])

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Sentences Evaluated               : {total}")
    print(f"Can do Clefting                        : {cleft_cnt} ({cleft_cnt / total * 100:.2f}%)")
    print(f"Can do Constituency Reordering         : {const_reord_cnt} ({const_reord_cnt / total * 100:.2f}%)")
    print(f"Can do Positional Cleft Reordering     : {cleft_reord_cnt} ({cleft_reord_cnt / total * 100:.2f}%)")
    print(f"Can do Any Reordering (Const or Cleft) : {reord_any_cnt} ({reord_any_cnt / total * 100:.2f}%)")
    print(f"Can do Both (Clefting + Const Reorder) : {both_cleft_const} ({both_cleft_const / total * 100:.2f}%)")
    print(f"Can do Both (Clefting + Cleft Reorder) : {both_cleft_cleft_reord} ({both_cleft_cleft_reord / total * 100:.2f}%)")
    print(f"Can do Either (Clefting OR Const Reord): {either_cleft_const} ({either_cleft_const / total * 100:.2f}%)")
    print(f"Can do Either (Clefting OR Any Reord)  : {either_cleft_any_reord} ({either_cleft_any_reord / total * 100:.2f}%)")
    print("-" * 60)
    print(f"Strict Exact Match to Target           : {exact_matches} ({exact_matches / total * 100:.2f}%)")
    print(f"Normalized Match (ignoring punct/space): {norm_matches} ({norm_matches / total * 100:.2f}%)")
    print("=" * 60)

if __name__ == '__main__':
    main()
