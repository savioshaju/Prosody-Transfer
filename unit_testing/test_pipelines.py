import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cleft.malayalam_pipeline import MalayalamPipeline


def main():
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) > 1:
        malayalam_text = " ".join(sys.argv[1:]).strip()
    else:
        print("Usage: python test_pipelines.py \"<Malayalam Sentence>\"")
        sys.exit(1)

    print("\n==============================================")
    print("--- Malayalam Pipeline ---")
    print("==============================================")
    mal_pipeline = MalayalamPipeline()
    mal_sentence = mal_pipeline.process(malayalam_text)
    print("\n--- JSON IR Output ---")
    print(mal_sentence.to_json())


if __name__ == "__main__":
    main()
    