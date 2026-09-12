import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cleft.cleft_pipeline import CleftPipeline


def main():
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:]).strip()
    else:
        print("Usage: python test_cleft.py \"<Malayalam Sentence with <FF>focus<FF>>\"")
        sys.exit(1)

    pipeline = CleftPipeline()
    result = pipeline.process(sentence)
    print("\n" + str(result))


if __name__ == "__main__":
    main()

