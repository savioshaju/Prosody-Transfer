"""
Root entry point / proxy for awesome_cleft_pipeline.
The main implementation lives in cleft/awesome_cleft_pipeline.py.
"""
import os
import sys

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

from cleft.awesome_cleft_pipeline import (
    AwesomeCleftPipeline,
    AwesomeAlignerWrapper,
    strip_punctuation,
    extract_alignment_and_prosody,
    clean_no_punct,
    main,
)

if __name__ == "__main__":
    main()
