import json
import sys
import re
import unicodedata
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def strip_all_punctuation_and_normalize(text):
    if not text or not isinstance(text, str):
        return ""
    t = text
    # 1. Normalize Malayalam chillu variants to canonical atomic chillus
    t = t.replace('\u0d23\u0d4d\u200d', '\u0d7a')  # ൺ
    t = t.replace('\u0d28\u0d4d\u200d', '\u0d7b')  # ൻ
    t = t.replace('\u0d30\u0d4d\u200d', '\u0d7c')  # ർ
    t = t.replace('\u0d32\u0d4d\u200d', '\u0d7d')  # ൽ
    t = t.replace('\u0d33\u0d4d\u200d', '\u0d7e')  # ൾ
    t = t.replace('\u0d15\u0d4d\u200d', '\u0d7f')  # ൿ
    
    # 2. Remove zero-width characters
    t = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', t)
    
    # 3. Strip ALL punctuation marks everywhere (commas, periods, quotes, parens, hyphens, etc.)
    # Using Unicode punctuation categories P* and common symbols
    chars = []
    for char in t:
        cat = unicodedata.category(char)
        if not cat.startswith('P') and char not in ('"', "'", '`', '´', '’', '‘', '“', '”', '—', '–', '-', '…'):
            chars.append(char)
        else:
            chars.append(' ')
    t = "".join(chars)
    
    # 4. Collapse all whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t

with open('eval_results.json', 'r', encoding='utf-8') as f:
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

# Now evaluate matches without considering punctuation
no_punct_matches = []
matched_output_types = {}

for d in data:
    target_clean = strip_all_punctuation_and_normalize(d['target'])
    matched_types = []
    
    for out_name, out_val in d['candidate_outputs'].items():
        out_clean = strip_all_punctuation_and_normalize(out_val)
        if out_clean == target_clean:
            matched_types.append(out_name)
            matched_output_types[out_name] = matched_output_types.get(out_name, 0) + 1
            
    is_match = len(matched_types) > 0
    d['no_punct_match'] = is_match
    d['no_punct_matched_types'] = matched_types
    if is_match:
        no_punct_matches.append(d)

match_count = len(no_punct_matches)

print("=" * 65)
print("EVALUATION RESULTS (PUNCTUATION COMPLETELY IGNORED)")
print("=" * 65)
print(f"Total Sentences Tested                           : {total}")
print(f"Sentences where CLEFTING was done                : {cleft_cnt} ({cleft_cnt/total*100:.2f}%)")
print(f"Sentences where CONSTITUENCY REORDERING was done : {const_reord_cnt} ({const_reord_cnt/total*100:.2f}%)")
print(f"Sentences where POSITIONAL CLEFT REORDERING done : {cleft_reord_cnt} ({cleft_reord_cnt/total*100:.2f}%)")
print(f"Sentences where ANY REORDERING was done          : {reord_any_cnt} ({reord_any_cnt/total*100:.2f}%)")
print("-" * 65)
print(f"Both Clefting AND Constituency Reordering        : {both_cleft_const} ({both_cleft_const/total*100:.2f}%)")
print(f"Both Clefting AND Positional Cleft Reordering    : {both_cleft_cleft_reord} ({both_cleft_cleft_reord/total*100:.2f}%)")
print(f"Either Clefting OR Constituency Reordering       : {either_cleft_const} ({either_cleft_const/total*100:.2f}%)")
print(f"Either Clefting OR Any Reordering                : {total} (100.00%)")
print("-" * 65)
print(f"Sentences Matching Target (No Punctuation)       : {match_count} / {total} ({match_count/total*100:.2f}%)")
print("=" * 65)

print("\n--- Breakdown of Matching Output Types ---")
for t, cnt in sorted(matched_output_types.items(), key=lambda x: -x[1]):
    print(f"  * {t:<25}: {cnt} sentences")

print("\n--- Matched Sentences Details (Sample 10) ---")
for m in no_punct_matches[:10]:
    print(f"Row {m['index']:<2} | Focus: '{m['focus']}' (Projected: '{m['proj_focus']}')")
    print(f"  Target: {m['target']}")
    for t in m['no_punct_matched_types']:
        print(f"  ✓ {t}: {m['candidate_outputs'][t]}")
    print()

# Also save updated CSV with no_punct_match flag
updated_records = []
for d in data:
    target_clean = strip_all_punctuation_and_normalize(d['target'])
    best_candidate = ""
    for t in d.get('no_punct_matched_types', []):
        best_candidate = d['candidate_outputs'][t]
        break
    if not best_candidate:
        best_candidate = d['candidate_outputs'].get('cleft_emphasized', '')
        
    updated_records.append({
        'index': d['index'],
        'eng': d['eng'],
        'mal': d['mal'],
        'focus': d['focus'],
        'proj_focus': d.get('proj_focus', ''),
        'target': d['target'],
        'no_punct_match': d['no_punct_match'],
        'matched_types': ", ".join(d['no_punct_matched_types']),
        'cleft_done': d['cleft_done'],
        'const_reord_done': d['const_reord_done'],
        'cleft_reord_done': d['cleft_reord_done'],
        'pipeline_cleft_out': d['candidate_outputs'].get('cleft_emphasized', ''),
        'pipeline_preverbal': d['candidate_outputs'].get('cleft_preverbal', ''),
        'pipeline_postverbal': d['candidate_outputs'].get('cleft_postverbal', ''),
        'pipeline_clause_initial': d['candidate_outputs'].get('cleft_clause_initial', ''),
        'pipeline_const_reord': d['candidate_outputs'].get('constituency_reordered', ''),
    })

pd.DataFrame(updated_records).to_csv('eval_results_no_punct.csv', index=False, encoding='utf-8-sig')
print("Saved full results to 'eval_results_no_punct.csv'")
