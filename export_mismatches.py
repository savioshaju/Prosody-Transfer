import json
import os
import re
import unicodedata
import pandas as pd

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

def diagnose_difference(target, candidates, mal, proj_focus):
    target_clean = clean_no_punct(target)
    cleft = clean_no_punct(candidates.get('cleft_emphasized', ''))
    
    # Check if target contains copula
    tgt_has_aanu = any(cop in target for cop in ("ാണ്", "ആണ്", "യാണ", "യാണെ"))
    cleft_has_aanu = any(cop in cleft for cop in ("ാണ്", "ആണ്", "യാണ", "യാണെ"))
    
    if not cleft:
        return "Clefting not produced (PE / Ineligible constituent)"
    
    # Check if differences are in verb ending
    tgt_words = target_clean.split()
    cleft_words = cleft.split()
    
    if tgt_words and cleft_words:
        if tgt_words[-1] != cleft_words[-1] and tgt_words[:-1] == cleft_words[:-1]:
            return f"Verb lexical/derivation difference ('{cleft_words[-1]}' vs '{tgt_words[-1]}')"
        
        # Check focus attachment
        if proj_focus and clean_no_punct(proj_focus) not in target_clean:
            return f"Focus projection span boundary difference (Pipeline projected '{proj_focus}')"
        
    return "Word order / clausal attachment variation"

def find_first_mismatch(target_str, candidate_str):
    target_clean = clean_no_punct(target_str)
    cand_clean = clean_no_punct(candidate_str)
    
    tgt_words = target_clean.split()
    cand_words = cand_clean.split()
    
    word_idx = None
    target_word = ''
    output_word = ''
    
    min_len = min(len(tgt_words), len(cand_words))
    for i in range(min_len):
        if tgt_words[i] != cand_words[i]:
            word_idx = i
            target_word = tgt_words[i]
            output_word = cand_words[i]
            break
            
    if word_idx is None:
        if len(tgt_words) != len(cand_words):
            word_idx = min_len
            target_word = tgt_words[min_len] if min_len < len(tgt_words) else '<END>'
            output_word = cand_words[min_len] if min_len < len(cand_words) else '<END>'
        else:
            word_idx = -1
            target_word = ''
            output_word = ''

    char_idx = None
    min_char_len = min(len(target_clean), len(cand_clean))
    for j in range(min_char_len):
        if target_clean[j] != cand_clean[j]:
            char_idx = j
            break
    if char_idx is None:
        if len(target_clean) != len(cand_clean):
            char_idx = min_char_len
        else:
            char_idx = -1

    return {
        'first_mismatch_word_index_0based': word_idx,
        'first_mismatch_word_index_1based': word_idx + 1 if word_idx >= 0 else -1,
        'first_mismatch_char_index': char_idx,
        'first_mismatch_target_word': target_word,
        'first_mismatch_output_word': output_word
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    eval_json_path = os.path.join(base_dir, 'eval_results.json')
    
    with open(eval_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    matched_list = []
    mismatched_list = []

    for d in data:
        target_clean = clean_no_punct(d['target'])
        matched_types = []
        
        for out_name, out_val in d['candidate_outputs'].items():
            if clean_no_punct(out_val) == target_clean:
                matched_types.append(out_name)

        is_match = len(matched_types) > 0
        diag = "" if is_match else diagnose_difference(d['target'], d['candidate_outputs'], d['mal'], d.get('proj_focus', ''))
        cleft_output = d['candidate_outputs'].get('cleft_emphasized', '')
        
        mismatch_info = find_first_mismatch(d['target'], cleft_output) if not is_match else {
            'first_mismatch_word_index_0based': -1,
            'first_mismatch_word_index_1based': -1,
            'first_mismatch_char_index': -1,
            'first_mismatch_target_word': '',
            'first_mismatch_output_word': ''
        }

        record = {
            'row_index': d['index'] + 1,  # 1-indexed for easy spreadsheet reference
            'dataset_index_0based': d['index'],
            'english_sentence': d['eng'],
            'english_focus_word': d['focus'],
            'projected_malayalam_focus': d.get('proj_focus', ''),
            'original_malayalam_sentence': d['mal'],
            'ground_truth_target': d['target'],
            'is_matched_any_output': is_match,
            'matched_output_types': ", ".join(matched_types) if is_match else "NONE",
            'difference_diagnosis': diag,
            'first_mismatch_word_index_0based': mismatch_info['first_mismatch_word_index_0based'],
            'first_mismatch_word_index_1based': mismatch_info['first_mismatch_word_index_1based'],
            'first_mismatch_char_index': mismatch_info['first_mismatch_char_index'],
            'first_mismatch_target_word': mismatch_info['first_mismatch_target_word'],
            'first_mismatch_output_word': mismatch_info['first_mismatch_output_word'],
            'output_cleft_emphasized': cleft_output,
            'output_constituency_reordered': d['candidate_outputs'].get('constituency_reordered', ''),
        }

        if is_match:
            matched_list.append(record)
        else:
            mismatched_list.append(record)

    total = len(data)
    mismatched_df = pd.DataFrame(mismatched_list)
    all_df = pd.DataFrame(matched_list + mismatched_list).sort_values('dataset_index_0based')

    mismatched_csv_path = os.path.join(base_dir, 'mismatched_sentences.csv')
    mismatched_json_path = os.path.join(base_dir, 'mismatched_sentences.json')
    
    mismatched_df.to_csv(mismatched_csv_path, index=False, encoding='utf-8-sig')
    with open(mismatched_json_path, 'w', encoding='utf-8') as f:
        json.dump(mismatched_list, f, ensure_ascii=False, indent=2)

    print(f"Total Sentences Tested           : {total}")
    print(f"Matched Sentences (No Punctuation): {len(matched_list)} ({len(matched_list)/total*100:.2f}%)")
    print(f"Mismatched Sentences             : {len(mismatched_list)} ({len(mismatched_list)/total*100:.2f}%)")
    print(f"Saved {len(mismatched_list)} mismatched sentences to '{mismatched_csv_path}' and '{mismatched_json_path}'")

if __name__ == '__main__':
    main()
