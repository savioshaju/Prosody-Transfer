import sys
import os
import json

# Ensure UTF-8 output encoding for Malayalam characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ssf_pipeline.verb_normalizer import VerbNormalizer

test_suite = [
    # 1. Past -ന്നു → -ന്നത്
    {"category": "Past -ന്നു → -ന്നത്", "input": "വന്നു", "expected": "വന്നത്"},
    {"category": "Past -ന്നു → -ന്നത്", "input": "കണ്ടു", "expected": "കണ്ടത്"},
    {"category": "Past -ന്നു → -ന്നത്", "input": "തിന്നു", "expected": "തിന്നത്"},
    {"category": "Past -ന്നു → -ന്നത്", "input": "നിന്നു", "expected": "നിന്നത്"},

    # 2. Past -ി → -ിയത്
    {"category": "Past -ി → -ിയത്", "input": "വാങ്ങി", "expected": "വാങ്ങിയത്"},
    {"category": "Past -ി → -ിയത്", "input": "പാടി", "expected": "പാടിയത്"},
    {"category": "Past -ി → -ിയത്", "input": "എഴുതി", "expected": "എഴുതിയത്"},
    {"category": "Past -ി → -ിയത്", "input": "ഓടി", "expected": "ഓടിയത്"},

    # 3. Present -ുന്നു → -ുന്നത്
    {"category": "Present -ുന്നു → -ുന്നത്", "input": "വരുന്നു", "expected": "വരുന്നത്"},
    {"category": "Present -ുന്നു → -ുന്നത്", "input": "കാണുന്നു", "expected": "കാണുന്നത്"},
    {"category": "Present -ുന്നു → -ുന്നത്", "input": "എഴുതുന്നു", "expected": "എഴുതുന്നത്"},
    {"category": "Present -ുന്നു → -ുന്നത്", "input": "ഓടുന്നു", "expected": "ഓടുന്നത്"},

    # 4. Present -യുന്നു → -യുന്നത്
    {"category": "Present -യുന്നു → -യുന്നത്", "input": "ചെയ്യുന്നു", "expected": "ചെയ്യുന്നത്"},
    {"category": "Present -യുന്നു → -യുന്നത്", "input": "പറയുന്നു", "expected": "പറയുന്നത്"},
    {"category": "Present -യുന്നു → -യുന്നത്", "input": "അറിയുന്നു", "expected": "അറിയുന്നത്"},
    {"category": "Present -യുന്നു → -യുന്നത്", "input": "കരയുന്നു", "expected": "കരയുന്നത്"},

    # 5. Future -ും → -ുന്നത്
    {"category": "Future -ും → -ുന്നത്", "input": "വരും", "expected": "വരുന്നത്"},
    {"category": "Future -ും → -ുന്നത്", "input": "പോകും", "expected": "പോകുന്നത്"},
    {"category": "Future -ും → -ുന്നത്", "input": "ചെയ്യും", "expected": "ചെയ്യുന്നത്"},
    {"category": "Future -ും → -ുന്നത്", "input": "കാണും", "expected": "കാണുന്നത്"},
    {"category": "Future -ും → -ുന്നത്", "input": "പറയും", "expected": "പറയുന്നത്"},
    {"category": "Future -ും → -ുന്നത്", "input": "എഴുതും", "expected": "എഴുതുന്നത്"},

    # 6. Negative -ഇല്ല → -ാത്തത്
    {"category": "Negative -ഇല്ല → -ാത്തത്", "input": "വരുന്നില്ല", "expected": "വരാത്തത്"},
    {"category": "Negative -ഇല്ല → -ാത്തത്", "input": "കാണുന്നില്ല", "expected": "കാണാത്തത്"},
    {"category": "Negative -ഇല്ല → -ാത്തത്", "input": "ചെയ്യുന്നില്ല", "expected": "ചെയ്യാത്തത്"},
    {"category": "Negative -ഇല്ല → -ാത്തത്", "input": "വന്നില്ല", "expected": "വരാത്തത്"},
    {"category": "Negative -ഇല്ല → -ാത്തത്", "input": "കണ്ടില്ല", "expected": "കാണാത്തത്"},
    {"category": "Negative -ഇല്ല → -ാത്തത്", "input": "പോയില്ല", "expected": "പോകാത്തത്"},

    # 7. Habitual affirmative -ാറുണ്ട് → -ാറുള്ളത്
    {"category": "Habitual affirmative -ാറുണ്ട് → -ാറുള്ളത്", "input": "വരാറുണ്ട്", "expected": "വരാറുള്ളത്"},
    {"category": "Habitual affirmative -ാറുണ്ട് → -ാറുള്ളത്", "input": "കാണാറുണ്ട്", "expected": "കാണാറുള്ളത്"},
    {"category": "Habitual affirmative -ാറുണ്ട് → -ാറുള്ളത്", "input": "ചെയ്യാറുണ്ട്", "expected": "ചെയ്യാറുള്ളത്"},
    {"category": "Habitual affirmative -ാറുണ്ട് → -ാറുള്ളത്", "input": "പോകാറുണ്ട്", "expected": "പോകാറുള്ളത്"},
    {"category": "Habitual affirmative -ാറുണ്ട് → -ാറുള്ളത്", "input": "പറയാറുണ്ട്", "expected": "പറയാറുള്ളത്"},
    {"category": "Habitual affirmative -ാറുണ്ട് → -ാറുള്ളത്", "input": "എഴുതാറുണ്ട്", "expected": "എഴുതാറുള്ളത്"},

    # 8. Habitual negative -ാറില്ല → -ാറില്ലാത്തത്
    {"category": "Habitual negative -ാറില്ല → -ാറില്ലാത്തത്", "input": "വരാറില്ല", "expected": "വരാറില്ലാത്തത്"},
    {"category": "Habitual negative -ാറില്ല → -ാറില്ലാത്തത്", "input": "കാണാറില്ല", "expected": "കാണാറില്ലാത്തത്"},
    {"category": "Habitual negative -ാറില്ല → -ാറില്ലാത്തത്", "input": "ചെയ്യാറില്ല", "expected": "ചെയ്യാറില്ലാത്തത്"},
    {"category": "Habitual negative -ാറില്ല → -ാറില്ലാത്തത്", "input": "പോകാറില്ല", "expected": "പോകാറില്ലാത്തത്"},
    {"category": "Habitual negative -ാറില്ല → -ാറില്ലാത്തത്", "input": "പറയാറില്ല", "expected": "പറയാറില്ലാത്തത്"},
    {"category": "Habitual negative -ാറില്ല → -ാറില്ലാത്തത്", "input": "എഴുതാറില്ല", "expected": "എഴുതാറില്ലാത്തത്"},

    # 9. Obligative -ണം → -േണ്ടത്
    {"category": "Obligative -ണം → -േണ്ടത്", "input": "ചെയ്യണം", "expected": "ചെയ്യേണ്ടത്"},
    {"category": "Obligative -ണം → -േണ്ടത്", "input": "വരണം", "expected": "വരേണ്ടത്"},
    {"category": "Obligative -ണം → -േണ്ടത്", "input": "പോകണം", "expected": "പോകേണ്ടത്"},
    {"category": "Obligative -ണം → -േണ്ടത്", "input": "എഴുതണം", "expected": "എഴുതേണ്ടത്"},
    {"category": "Obligative -ണം → -േണ്ടത്", "input": "കാണണം", "expected": "കാണേണ്ടത്"},
    {"category": "Obligative -ണം → -േണ്ടത്", "input": "പറയണം", "expected": "പറയേണ്ടത്"},

    # 10. Permissive -ാം → -ാവുന്നത്
    {"category": "Permissive -ാം → -ാവുന്നത്", "input": "വരാം", "expected": "വരാവുന്നത്"},
    {"category": "Permissive -ാം → -ാവുന്നത്", "input": "പോകാം", "expected": "പോകാവുന്നത്"},
    {"category": "Permissive -ാം → -ാവുന്നത്", "input": "ചെയ്യാം", "expected": "ചെയ്യാവുന്നത്"},
    {"category": "Permissive -ാം → -ാവുന്നത്", "input": "കാണാം", "expected": "കാണാവുന്നത്"},
    {"category": "Permissive -ാം → -ാവുന്നത്", "input": "പറയാം", "expected": "പറയാവുന്നത്"},
    {"category": "Permissive -ാം → -ാവുന്നത്", "input": "എഴുതാം", "expected": "എഴുതാവുന്നത്"},

    # 11. Already nominalized -ത് → unchanged
    {"category": "Already nominalized -ത് → unchanged", "input": "വന്നത്", "expected": "വന്നത്"},
    {"category": "Already nominalized -ത് → unchanged", "input": "വാങ്ങിയത്", "expected": "വാങ്ങിയത്"},
    {"category": "Already nominalized -ത് → unchanged", "input": "വരുന്നത്", "expected": "വരുന്നത്"},
    {"category": "Already nominalized -ത് → unchanged", "input": "ചെയ്യുന്നത്", "expected": "ചെയ്യുന്നത്"},
    {"category": "Already nominalized -ത് → unchanged", "input": "പോയത്", "expected": "പോയത്"},

    # 12. Negative Controls (Unsupported / Unresolved)
    {"category": "Unsupported Controls", "input": "വന്നാൽ", "expected": None},
    {"category": "Unsupported Controls", "input": "വരട്ടെ", "expected": None},
    {"category": "Unsupported Controls", "input": "മേശ", "expected": None}
]

def run_tests():
    normalizer = VerbNormalizer()
    results = []
    
    total = len(test_suite)
    passed = 0
    failed = 0
    
    print("=" * 100)
    print(f"{'CATEGORY':<36} | {'INPUT':<14} | {'EXPECTED':<16} | {'NORMALIZED':<16} | {'STATUS':<7} | {'MLMORPH RE-ANALYSIS'}")
    print("=" * 100)
    
    for item in test_suite:
        cat = item["category"]
        verb = item["input"]
        expected = item["expected"]
        
        res = normalizer.normalize(verb)
        norm_output = res["normalized"]
        status = res["status"]
        reanalysis = res["normalized_analysis"]
        
        if expected is None:
            # Expected to be UNRESOLVED
            is_pass = (status == "UNRESOLVED")
        else:
            is_pass = (status == "VALID" and norm_output == expected and bool(reanalysis))
            
        if is_pass:
            passed += 1
            verdict = "PASS"
        else:
            failed += 1
            verdict = "FAIL"
            
        print(f"{cat:<36} | {verb:<14} | {str(expected):<16} | {str(norm_output):<16} | {verdict:<7} | {reanalysis}")
        
        results.append({
            "category": cat,
            "input": verb,
            "expected": expected,
            "normalized": norm_output,
            "status": status,
            "verdict": verdict,
            "reanalysis": reanalysis,
            "failure_reason": res["failure_reason"]
        })
        
    print("=" * 100)
    print(f"Total Tests: {total} | Passed: {passed} | Failed: {failed} | Success Rate: {(passed/total)*100:.1f}%")
    print("=" * 100)

if __name__ == "__main__":
    run_tests()
