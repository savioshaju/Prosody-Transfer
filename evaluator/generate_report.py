import json
import re
import unicodedata

def clean_no_punct(text):
    if not text or not isinstance(text, str):
        return ""
    t = text
    # 1. Normalize Malayalam chillu variants (both ZWJ-based and raw virama at word end/before space/punct)
    # ZWJ chillus
    t = t.replace('\u0d23\u0d4d\u200d', '\u0d7a')  # ൺ
    t = t.replace('\u0d28\u0d4d\u200d', '\u0d7b')  # ൻ
    t = t.replace('\u0d30\u0d4d\u200d', '\u0d7c')  # ർ
    t = t.replace('\u0d32\u0d4d\u200d', '\u0d7d')  # ൽ
    t = t.replace('\u0d33\u0d4d\u200d', '\u0d7e')  # ൾ
    t = t.replace('\u0d15\u0d4d\u200d', '\u0d7f')  # ൿ

    # Raw virama chillus at word boundary or before whitespace
    t = re.sub(r'\u0d33\u0d4d(?=[\s\b.,;!?\"\'“”‘’\-]|$)', '\u0d7e', t)  # ള് -> ൾ
    t = re.sub(r'\u0d32\u0d4d(?=[\s\b.,;!?\"\'“”‘’\-]|$)', '\u0d7d', t)  # ല് -> ൽ
    t = re.sub(r'\u0d28\u0d4d(?=[\s\b.,;!?\"\'“”‘’\-]|$)', '\u0d7b', t)  # ന് -> ൻ
    t = re.sub(r'\u0d30\u0d4d(?=[\s\b.,;!?\"\'“”‘’\-]|$)', '\u0d7c', t)  # ര് -> ർ
    t = re.sub(r'\u0d23\u0d4d(?=[\s\b.,;!?\"\'“”‘’\-]|$)', '\u0d7a', t)  # ണ് -> ൺ

    # Standardize 'ന്റെ' vs 'ന്‍റെ'
    t = t.replace('\u0d28\u0d4d\u200d\u0d31', '\u0d28\u0d4d\u0d31')
    t = t.replace('\u0d7b\u0d31', '\u0d28\u0d4d\u0d31')
    
    # 2. Remove zero-width characters (ZWNJ, ZWJ, ZWSP)
    t = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', t)
    
    # 3. Strip all punctuation marks everywhere
    chars = []
    for c in t:
        cat = unicodedata.category(c)
        if not cat.startswith('P') and c not in ('"', "'", '`', '´', '’', '‘', '“', '”', '—', '–', '-', '…'):
            chars.append(c)
        else:
            chars.append(' ')
    return re.sub(r'\s+', ' ', "".join(chars)).strip()

import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    eval_json_path = os.path.join(base_dir, 'eval_results.json')
    with open(eval_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    cleft_cnt = sum(1 for d in data if d['cleft_done'])
    const_reord_cnt = sum(1 for d in data if d['const_reord_done'])
    cleft_reord_cnt = sum(1 for d in data if d['cleft_reord_done'])
    reord_any_cnt = sum(1 for d in data if d['const_reord_done'] or d['cleft_reord_done'])

    both_cleft_const = sum(1 for d in data if d['cleft_done'] and d['const_reord_done'])
    both_cleft_cleft_reord = sum(1 for d in data if d['cleft_done'] and d['cleft_reord_done'])
    either_cleft_const = sum(1 for d in data if d['cleft_done'] or d['const_reord_done'])
    either_cleft_any_reord = sum(1 for d in data if d['cleft_done'] or reord_any_cnt)

    matches_no_punct = 0
    matched_indices = []
    type_counts = {}
    
    match_details = []

    for d in data:
        target_clean = clean_no_punct(d['target'])
        m_types = []
        for out_name, out_val in d['candidate_outputs'].items():
            if clean_no_punct(out_val) == target_clean:
                m_types.append(out_name)
                type_counts[out_name] = type_counts.get(out_name, 0) + 1
        
        if m_types:
            matches_no_punct += 1
            matched_indices.append(d['index'])
            match_details.append({
                'index': d['index'],
                'eng': d['eng'],
                'focus': d['focus'],
                'proj_focus': d.get('proj_focus', ''),
                'target': d['target'],
                'matched_types': m_types,
                'matched_outputs': {t: d['candidate_outputs'][t] for t in m_types},
            })

    report = {
        'total_sentences': total,
        'can_cleft': cleft_cnt,
        'can_cleft_pct': round(cleft_cnt / total * 100, 2),
        'can_const_reorder': const_reord_cnt,
        'can_const_reorder_pct': round(const_reord_cnt / total * 100, 2),
        'can_cleft_reorder': cleft_reord_cnt,
        'can_cleft_reorder_pct': round(cleft_reord_cnt / total * 100, 2),
        'can_any_reorder': reord_any_cnt,
        'can_any_reorder_pct': round(reord_any_cnt / total * 100, 2),
        'both_cleft_and_const_reorder': both_cleft_const,
        'both_cleft_and_const_reorder_pct': round(both_cleft_const / total * 100, 2),
        'both_cleft_and_cleft_reorder': both_cleft_cleft_reord,
        'both_cleft_and_cleft_reorder_pct': round(both_cleft_cleft_reord / total * 100, 2),
        'either_cleft_or_const_reorder': either_cleft_const,
        'either_cleft_or_const_reorder_pct': round(either_cleft_const / total * 100, 2),
        'either_cleft_or_any_reorder': either_cleft_any_reord,
        'either_cleft_or_any_reorder_pct': round(either_cleft_any_reord / total * 100, 2),
        'matches_ignoring_punctuation': matches_no_punct,
        'matches_ignoring_punctuation_pct': round(matches_no_punct / total * 100, 2),
        'matched_output_types': type_counts,
        'matched_indices': matched_indices,
        'match_details': match_details,
    }

    final_json_path = os.path.join(base_dir, 'final_report.json')
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("SUCCESS: Generated final_report.json")
    print(f"Total: {total}")
    print(f"Matches (No Punctuation): {matches_no_punct} ({matches_no_punct/total*100:.2f}%)")
    print(f"Can Cleft: {cleft_cnt} ({cleft_cnt/total*100:.2f}%)")
    print(f"Can Const Reorder: {const_reord_cnt} ({const_reord_cnt/total*100:.2f}%)")
    print(f"Can Cleft Reorder: {cleft_reord_cnt} ({cleft_reord_cnt/total*100:.2f}%)")
    print(f"Can Both (Cleft + Const): {both_cleft_const} ({both_cleft_const/total*100:.2f}%)")
    print(f"Can Either (Cleft or Const): {either_cleft_const} ({either_cleft_const/total*100:.2f}%)")

if __name__ == '__main__':
    main()
