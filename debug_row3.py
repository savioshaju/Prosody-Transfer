import sys
import os

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from awesome_cleft_pipeline import AwesomeCleftPipeline

pipeline = AwesomeCleftPipeline(aligner_type="simalign", aligner_method="itermax")

eng = "The 2015 Commonwealth Karate Games were held in Delhi, India."
mal = "2015 ലെ കോമൺവെൽത്ത് കരാട്ടെ ഗെയിംസ് ഇന്ത്യയിലെ ഡൽഹിയിൽ നടന്നു."
focus = "The 2015"

res = pipeline.process(english_sentence=eng, malayalam_sentence=mal, english_focus=focus)

print("Pre-alignment:")
for item in res['pre_cleft_en_to_ml_alignment']['src_to_tgt_alignments']:
    print(f"  [{item['src_index']}] {item['src_word']} -> [{item['tgt_index']}] {item['tgt_word']} (bidi={item['is_bidirectional']})")

print("\nFocus candidates:")
for c in res['focus_candidates']:
    print(f"  Rank {c['rank']}: {c['span_range']} -> '{c['span_text']}'")

print("\nProjected focus:", res['focused_malayalam_constituent'])
print("Cleft output:", res['emphasized_malayalam_sentence'])
