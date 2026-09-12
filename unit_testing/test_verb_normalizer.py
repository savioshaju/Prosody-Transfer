import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verb_norm.verb_normalizer import VerbNormalizer, print_result


def main():
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) > 1:
        verb = " ".join(sys.argv[1:]).strip()
    else:
        print("Usage: python test_verb_normalizer.py \"<Malayalam Verb>\"")
        sys.exit(1)

    print("\n==============================================")
    print("--- Malayalam Verb Normalizer ---")
    print("==============================================")
    normalizer = VerbNormalizer()
    result = normalizer.normalize(verb)
    print_result(result)


if __name__ == "__main__":
    main()