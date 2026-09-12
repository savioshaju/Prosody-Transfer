import json
import os
import re
import unicodedata
import pandas as pd

def clean_no_punct(text):
    if not text or not isinstance(text, str):
        return ""
    t = text
    # 0. Normalize hyphens between digits/dates and suffixes (e.g. 2015-ൽ -> 2015ൽ, 2015-ലാണ് -> 2015ലാണ്)
    t = re.sub(r'(\d+)\s*[-–—]\s*', r'\1', t)

    # 1. Normalize Malayalam chillu variants (both ZWJ-based and raw virama)
    # Standardize 'ന്റെ' vs 'ന്‍റെ' first before chillu conversion
    t = t.replace('\u0d28\u0d4d\u200d\u0d31', '\u0d28\u0d4d\u0d31')
    t = t.replace('\u0d7b\u0d31', '\u0d28\u0d4d\u0d31')
    t = t.replace('ൻറ', 'ന്റെ').replace('ൻറെ', 'ന്റെ')

    # Atomic chillu normalization (both with and without ZWJ)
    t = re.sub(r'ല\u0d4d\u200d?|ല്\u200d?', 'ൽ', t)
    t = re.sub(r'ള\u0d4d\u200d?|ള്\u200d?', 'ൾ', t)
    t = re.sub(r'ര\u0d4d\u200d?|ര്\u200d?', 'ർ', t)
    t = re.sub(r'ണ\u0d4d\u200d?|ണ്\u200d?', 'ൺ', t)
    t = re.sub(r'ക\u0d4d\u200d|ക്\u200d', 'ൿ', t)
    
    # Chillu N: ZWJ chillu or virama n not followed by rra
    t = t.replace('\u0d28\u0d4d\u200d', '\u0d7b')
    t = re.sub(r'ന്(?![\u0d31റ])', 'ൻ', t)
    t = re.sub(r'\u0d28\u0d4d(?![\u0d31])', '\u0d7b', t)
    
    # 2. Remove zero-width characters (ZWNJ, ZWJ, ZWSP)
    t = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', t)

    # 3. Orthographic free variations in Malayalam:
    # (a) Adverbial copula -മായാണ് / -മായാൺ vs -മായിയാണ് / -മായിയാൺ
    t = re.sub(r'മായാ[ണ്ൺ]\b', 'മായിയാൺ', t)
    t = re.sub(r'മായിയാ[ണ്ൺ]\b', 'മായിയാൺ', t)
    # (b) Detached copula after alveolar/dental stops (e.g. ഇൻകോർപറേറ്റഡ് ആണ് vs ഇൻകോർപറേറ്റഡാണ്)
    t = re.sub(r'(\w)്\s+ആ[ണ്ൺ]\b', r'\1ാൺ', t)
    # (b2) Detached copula after front vowels with glide y (e.g. റേ ആണ് vs റേയാണ്, ഡൽഹി ആണ് vs ഡൽഹിയാണ്)
    t = re.sub(r'([േൈീി])\s+ആ[ണ്ൺ]\b', r'\1യാൺ', t)
    t = re.sub(r'([േൈീി])യാ[ണ്ൺ]\b', r'\1യാൺ', t)
    # (b3) Locative adnominal vs argument copular allomorphs (-ിലെയാണ് vs -ിലാണ്, -ലെയാണ് vs -ലാണ്)
    t = re.sub(r'ത്തിലെയാ[ണ്ൺ]\b', 'ത്തിലാൺ', t)
    t = re.sub(r'ിലെയാ[ണ്ൺ]\b', 'ിലാൺ', t)
    t = re.sub(r'ത്തെയാ[ണ്ൺ]\b', 'ത്താൺ', t)
    t = re.sub(r'ലെയാ[ണ്ൺ]\b', 'ലാൺ', t)
    t = re.sub(r'(\d+)\s+ലാ[ണ്ൺ]\b', r'\1ലാൺ', t)
    # (c) Sandhi fusion of demonstrative pronoun + initial vowel (e.g. അത് ഉൾക്കൊള്ളുന്നത് vs അതുൾക്കൊള്ളുന്നത്)
    t = re.sub(r'\bഅത്\s+ഉൾ', 'അതുൾ', t)
    
    # 4. Strip all punctuation marks everywhere
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
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(eval_dir)
    eval_json_path = os.path.join(workspace_dir, 'eval_results.json')
    
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

    mismatched_csv_path = os.path.join(workspace_dir, 'mismatched_sentences.csv')
    mismatched_json_path = os.path.join(workspace_dir, 'mismatched_sentences.json')
    
    mismatched_df.to_csv(mismatched_csv_path, index=False, encoding='utf-8-sig')
    with open(mismatched_json_path, 'w', encoding='utf-8') as f:
        json.dump(mismatched_list, f, ensure_ascii=False, indent=2)

    print(f"Total Sentences Tested           : {total}")
    print(f"Matched Sentences (No Punctuation): {len(matched_list)} ({len(matched_list)/total*100:.2f}%)")
    print(f"Mismatched Sentences             : {len(mismatched_list)} ({len(mismatched_list)/total*100:.2f}%)")
    print(f"Saved {len(mismatched_list)} mismatched sentences to '{mismatched_csv_path}' and '{mismatched_json_path}'")

if __name__ == '__main__':
    main()
