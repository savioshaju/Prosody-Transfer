# Cross-Lingual Prosody & Emphasis Transfer (English → Malayalam)

An end-to-end system that takes an English sentence with an emphasized word (focus), its parallel Malayalam translation, and automatically converts the Malayalam sentence into a natural, grammatically correct **cleft sentence** carrying the exact same emphasis.

> **Important Note on Input:** This project works on **parallel sentence pairs** (an English sentence + its corresponding Malayalam translation).

---

## Quick Example of What This Project Does

* **English Input:** `"Raman gave Krishnan two books yesterday."`
* **English Word with Emphasis (Focus):** `"two books"`
* **Malayalam Parallel Input:** `"രാമൻ ഇന്നലെ കൃഷ്ണന് രണ്ട് പുസ്തകം കൊടുത്തു."`
* **System Output (Malayalam Cleft Sentence):**  
  `"രാമൻ ഇന്നലെ കൃഷ്ണന് രണ്ട് പുസ്തകമാണ് കൊടുത്തത്."`

### Why is this needed?
In English, people usually emphasize words using their voice pitch and tone (*"I gave him **two books**"*), or sometimes with an it-cleft (*"It was two books that I gave him"*).  
In Malayalam, voice tone alone is often not enough to show what was emphasized. Instead, Malayalam speakers naturally use **cleft sentences**:
1. Attach the focus copula **\ml{ആണ്}** (*āṇŭ*) directly to the emphasized word (\ml{രണ്ട് പുസ്തകം} $\to$ \ml{രണ്ട് പുസ്തകമാണ്}).
2. Turn the main verb into a nominalized form ending in **\ml{-ത്}** (*-atŭ*) (\ml{കൊടുത്തു} $\to$ \ml{കൊടുത്തത്}).

This project automates that complete transformation using neural alignment and rule-based morphology.

---

## Table of Contents
1. [Implementation-Level Architecture (The 6 Core Blocks)](#1-implementation-level-architecture-the-6-core-blocks)
2. [Why Each External Tool is Used](#2-why-each-external-tool-is-used)
3. [File-by-File Codebase Walkthrough](#3-file-by-file-codebase-walkthrough)
4. [Prerequisites & Installation](#4-prerequisites--installation)
5. [How to Run (CLI & Python API)](#5-how-to-run-cli--python-api)
6. [Input & Output Specifications](#6-input--output-specifications)
7. [Running Tests & Benchmarks](#7-running-tests--benchmarks)

---

## 1. Implementation-Level Architecture (The 6 Core Blocks)

The pipeline is split into six clear blocks. Each block solves one specific linguistic problem. Here is how data moves through them:

```
[Inputs: English Sentence + Malayalam Translation + English Focus Word]
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ BLOCK 1: Pre-Alignment Morphological Normalisation                     │
│ File: aligner/morpho_simalign.py                                       │
│ • Splits fused compound words into separate root stems                 │
│ • Strips case suffixes (e.g. locative, dative) so roots align cleanly │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ BLOCK 2: Neural Word Alignment & Focus Projection                      │
│ Files: aligner/simalign_wrapper.py, aligner/alignment_utils.py         │
│ • Runs mBERT (Layer 8) to get word embeddings                          │
│ • Uses IterMax algorithm to map English focus to Malayalam words       │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ BLOCK 3: Syntactic Phrase Chunking (Boundary Expansion)               │
│ File: cleft/phrase_chunker.py                                          │
│ • Checks neighboring tokens using mlmorph tags                         │
│ • Expands single words into full noun phrases (NP) or postpositional  │
│   phrases (PP) so modifiers (adjectives/demonstratives) stay together  │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ BLOCK 4: Cleft Host Formation & Morphophonemic Sandhi                  │
│ Files: cleft/copula_pipeline.py, cleft/cleft_pipeline.py               │
│ • Preserves case markers (e.g. dative ജോണിന് → ജോണിനാണ്, not ജോണാണ്)   │
│ • Applies sound-change rules (anusvāra, glides, virāma elision)        │
│ • Attaches ആണ് (āṇŭ) to create the cleft host                          │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ BLOCK 5: Clausal Verb Normalisation (-ത് Nominalisation)               │
│ File: verb_norm/verb_normalizer.py                                     │
│ • Identifies the main sentence verb                                    │
│ • Extracts tense, aspect, mood, and polarity via mlmorph               │
│ • Generates the relative participial form ending in -ത്                │
│ • Guards against double nominalisation                                 │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ BLOCK 6: Quality Assurance & Roundtrip Validation                      │
│ File: cleft/cleft_pipeline.py                                          │
│ • Feeds generated words back into mlmorph.Analyser                     │
│ • Rejects any non-existent or corrupted forms to stop hallucinations   │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
[Final Output: Emphasized Malayalam Cleft Sentence]
```

---

### Detailed Breakdown of Each Block

#### Block 1: Pre-Alignment Morphological Normalisation
* **Implemented in:** `aligner/morpho_simalign.py` (`MorphoSimAligner`)
* **What problem it solves:** In Malayalam, words frequently join together without spaces into compound words (*samāsam*), and nouns attach case suffixes (*vibhakti*). If you pass these directly to a neural word aligner, it gets confused because two English words (like *"Penal Code"*) correspond to a single Malayalam compound token (\ml{ശിക്ഷാനിയമം}), or because a case suffix distorts the word's embedding.
* **How it works step-by-step:**
  1. Tokenizes the Malayalam sentence.
  2. Passes each token into `mlmorph.Analyser`.
  3. If a word is a compound, it splits it into distinct words (e.g., \ml{ശിക്ഷാനിയമം} $\to$ \ml{ശിക്ഷ} + \ml{നിയമം}).
  4. If a word has a case suffix (like locative \ml{-യിൽ} or dative \ml{-ിന്}), it separates the case marker from the base root (e.g., \ml{ഇന്ത്യയിൽ} $\to$ root \ml{ഇന്ത്യ} + `<locative>`).
  5. The clean, bare root words are sent to the aligner. Once aligned, the indices are mapped back to the original sentence tokens.
* **Concrete Example:**  
  *Input word:* \ml{ശിക്ഷാനിയമം}  
  *Block 1 output:* \ml{ശിക്ഷ} + \ml{നിയമം} (allowing *"Penal"* to align with \ml{ശിക്ഷ} and *"Code"* with \ml{നിയമം}).

---

#### Block 2: Neural Word Alignment & Focus Projection
* **Implemented in:** `aligner/simalign_wrapper.py` (`SimAlignerWrapper`) and `aligner/alignment_utils.py`
* **What problem it solves:** Given that the user emphasized a word in English (e.g., `"two books"`), we need to know exactly which Malayalam word(s) in the parallel sentence correspond to that English word.
* **How it works step-by-step:**
  1. Loads multilingual BERT (`bert-base-multilingual-cased`).
  2. Extracts word representations from **Layer 8** (which captures the best balance of semantic and grammatical information).
  3. Calculates pairwise cosine similarity between all English and Malayalam words.
  4. Decodes the best alignment links using the **IterMax** algorithm:
     - It first pairs up the strongest bidirectional matches.
     - Then it discounts those matched words and iteratively extracts secondary matches. This prevents words from being dropped and cleanly handles cases where multiple English words map to one Malayalam word.
  5. Locates the token positions of the English focus word and projects them to the aligned Malayalam word(s).
* **Concrete Example:**  
  *English focus:* `"two books"` (tokens at positions 3 and 4)  
  *Alignment matrix:* `(3 -> 3)`, `(4 -> 4)`  
  *Block 2 output:* Malayalam focus = \ml{രണ്ട് പുസ്തകം} (tokens at positions 3 and 4).

---

#### Block 3: Syntactic Phrase Chunking (Boundary Expansion)
* **Implemented in:** `cleft/phrase_chunker.py` (`MalayalamPhraseChunker`)
* **What problem it solves:** Word alignment often only matches the central head noun (e.g., matching English *"man"* to Malayalam \ml{മനുഷ്യൻ}). But in a cleft sentence, you cannot emphasize just the noun and leave its adjective or demonstrative dangling; you must emphasize the **whole noun phrase** (*"That elderly man"* $\to$ \ml{ആ വൃദ്ധനായ മനുഷ്യൻ}).
* **How it works step-by-step:**
  1. Takes the projected Malayalam word from Block 2.
  2. Inspects preceding tokens using `mlmorph.Analyser` tags:
     - If the preceding word is a demonstrative (tag `<dem>`, e.g., \ml{ആ}, \ml{ഈ}), it absorbs it into the focus span.
     - If the preceding word is an adjective or relative participle (tag `<adj>`, e.g., \ml{നല്ല}, \ml{വൃദ്ധനായ}), it absorbs it into the focus span.
  3. Inspects following tokens:
     - If followed by a bound postposition (e.g., \ml{പുറത്ത്} in \ml{മേശപ്പുറത്ത്}), it keeps the whole postpositional phrase intact.
* **Concrete Example:**  
  *English focus:* `"That elderly man"`  
  *Raw alignment output:* \ml{മനുഷ്യൻ} (just *"man"*)  
  *Block 3 output:* Expanded focus = \ml{ആ വൃദ്ധനായ മനുഷ്യൻ} (*"That elderly man"*).

---

#### Block 4: Cleft Host Formation & Morphophonemic Sandhi
* **Implemented in:** `cleft/copula_pipeline.py` (`CopulaPipeline`) and `cleft/cleft_pipeline.py`
* **What problem it solves:** The copular focus marker \ml{ആണ്} (*āṇŭ*) must be attached to the focus word. But in Malayalam, you cannot simply glue \ml{ആണ്} with a space; it must fuse naturally according to classical sound-combination rules (*sandhi*), and you must never strip grammatical case endings.
* **How it works step-by-step:**
  1. **Case Marker Preservation:** Checks if the focused word carries an oblique case ending (like dative \ml{-ന്}, locative \ml{-ൽ}, instrumental \ml{-ആൽ}). If so, it keeps the case marker and attaches \ml{ആണ്} outside of it.
     - *Example:* Dative \ml{ജോണിന്} (to John) $+$ \ml{ആണ്} $\to$ \ml{ജോണിനാണ്} (It is to John). If you stripped the case to *\ml{ജോണാണ്}, the sentence would mean "It is John who...", changing the recipient meaning completely!
  2. **Sandhi Engine:** Applies the appropriate phonological rule based on how the host word ends:
     - **Anusvāra Resolution (\ml{ം} $\to$ \ml{മാ}):** If ending in anusvāra (\ml{ം}), it resolves to \ml{മ} + \ml{ആ} $\to$ \ml{മാ}.  
       *Example:* \ml{പുസ്തകം} $+$ \ml{ആണ്} $\to$ \ml{പുസ്തകമാണ്}
     - **Front Vowel $y$-Glide (\ml{യ}-insertion):** If ending in vowels like \ml{അ, ഇ, എ}, inserts \ml{യ}.  
       *Example:* \ml{കസേര} $+$ \ml{ആണ്} $\to$ \ml{കസേരയാണ്}, \ml{കുട്ടി} $+$ \ml{ആണ്} $\to$ \ml{കുട്ടിയാണ്}
     - **Back Vowel $v$-Glide (\ml{വ}-insertion):** If ending in vowels like \ml{ഉ, ഒ}, inserts \ml{വ}.  
       *Example:* \ml{ഗുരു} $+$ \ml{ആണ്} $\to$ \ml{ഗുരുവാണ്}
     - **Virāma Elision (\ml{്} $\to$ \ml{ാ}):** If ending in a consonant with a virāma (\ml{്}), drops the virāma and adds the long vowel sign \ml{ാ}.  
       *Example:* \ml{നിലത്ത്} $+$ \ml{ആണ്} $\to$ \ml{നിലത്താണ്}
     - **Chillu Consonant Fusion:** If ending in a chillu letter (\ml{ൻ, ർ, ൽ}), fuses directly.  
       *Example:* \ml{അവൻ} $+$ \ml{ആണ്} $\to$ \ml{അവനാണ്}, \ml{വീട്ടിൽ} $+$ \ml{ആണ്} $\to$ \ml{വീട്ടിലാണ്}
* **Block 4 output:** The fully formed, grammatically inflected cleft host (e.g., \ml{രണ്ട് പുസ്തകമാണ്}).

---

#### Block 5: Clausal Verb Normalisation (\ml{-ത്} Nominalisation)
* **Implemented in:** `verb_norm/verb_normalizer.py` (`VerbNormalizer`)
* **What problem it solves:** Once you place emphasis on a word using \ml{ആണ്}, the original main verb of the sentence can no longer remain a normal finite verb. It must be turned into a relative participial noun ending in \ml{-ത്} (*-atŭ*), which acts as the presupposition of the cleft.
* **How it works step-by-step:**
  1. Finds the main matrix verb of the clause (usually at the end of the sentence).
  2. Calls `mlmorph.Analyser` to detect the verb's lemma, tense, aspect, mood, and polarity.
  3. **Double Nominalisation Guard:** Checks if the verb already ends in `<deriv>` or \ml{-ത്}. If it is already nominalized, it skips it to prevent ungrammatical doubling like *\ml{വന്നതത്}.
  4. Builds the target grammatical tag string and passes it to `mlmorph.Generator` to produce the surface word.
* **Concrete Examples across Verb Types:**
  * **Past Tense:** \ml{കൊടുത്തു} (gave) $\to$ \ml{കൊടുത്തത്} (that which was given)
  * **Past Tense:** \ml{വാങ്ങി} (bought) $\to$ \ml{വാങ്ങിയത്} (that which was bought)
  * **Present Tense:** \ml{വരുന്നു} (comes) $\to$ \ml{വരുന്നത്} (that which comes)
  * **Negative Finite:** \ml{പോയില്ല} (did not go) $\to$ \ml{പോകാത്തത്} (that which did not go)
  * **Obligative Modal:** \ml{പോകണം} (must go) $\to$ \ml{പോകേണ്ടത്} (that which must be gone)
  * **Habitual Aspect:** \ml{എഴുതാറുണ്ട്} (usually writes) $\to$ \ml{എഴുതാറുള്ളത്}

---

#### Block 6: Quality Assurance & Roundtrip Validation
* **Implemented in:** `cleft/cleft_pipeline.py`
* **What problem it solves:** Pure string operations or machine-generated text can accidentally create broken words or phonological gibberish.
* **How it works step-by-step:**
  1. Takes the newly generated cleft host and the nominalized verb.
  2. Feeds them back into `mlmorph.Analyser`.
  3. If `mlmorph.Analyser` recognizes the words and returns valid morphological parse trees, the output is accepted.
  4. If `mlmorph.Analyser` returns empty ($\emptyset$), it means the word does not exist in Malayalam grammar. The system detects the failure and safely falls back to a deterministic canonical form.
* **Block 6 output:** Final verification check guaranteeing zero morphological hallucination.

---

## 2. Why Each External Tool is Used

We specifically picked four external libraries to implement this pipeline. Here is why each one is used:

### 1. `mlmorph` (by Santhosh Thottingal)
* **What it is:** A finite-state morphological processor specifically built for Malayalam using the Helsinki Finite-State Transducer (HFST) library.
* **Why we use it:**
  * Standard machine learning tokenizers (like WordPiece or Byte-Pair Encoding) break Malayalam words into meaningless character slices based only on frequency.
  * `mlmorph` understands authentic Malayalam grammar. It can bidirectionally **analyze** a word into its exact linguistic parts (stem, case, tense, mood) and **generate** a surface word from those parts.
  * We use it to decompose compound words, strip case markers before alignment, identify grammatical parts of speech for chunking, and synthesize the nominalized verb ending in \ml{-ത്}.

### 2. `SimAlign` (by Jalili Sabet et al.)
* **What it is:** A tool that extracts word alignments from multilingual transformer models without needing any human-annotated alignment datasets.
* **Why we use it:**
  * English and Malayalam have very different word orders (English is SVO; Malayalam is SOV). There is virtually no large, high-quality parallel word-alignment training data available.
  * SimAlign works **zero-shot** using pre-trained multilingual embeddings.
  * Its **IterMax** algorithm doesn't drop words. It matches words iteratively, which is essential because English often uses multiple separate auxiliary words (*"has written"*) for what Malayalam says in a single inflected word (\ml{എഴുതിയിട്ടുണ്ട്}).

### 3. `Awesome-Align` (by Dou & Neubig)
* **What it is:** A neural aligner that fine-tunes embeddings using probability thresholding on cross-lingual attention.
* **Why it is in the repository:**
  * It was implemented and tested as a comparative baseline.
  * In our evaluations, Awesome-Align's strict probability thresholding frequently dropped verbs, auxiliary tokens, and modifiers on English–Malayalam sentences. Testing it proved that SimAlign's IterMax is superior for this specific language pair.

### 4. `PyTorch` & `transformers` (`bert-base-multilingual-cased`)
* **What they are:** Standard deep-learning frameworks used to load and run multilingual BERT (mBERT).
* **Why we use them:**
  * mBERT gives us shared numerical representations for English and Malayalam words. We extract Layer 8 vectors to compute the word similarities needed for alignment.

---

## 3. File-by-File Codebase Walkthrough

Here is what every file in this repository does at the implementation level:

| Directory | File | Implementation-Level Description |
|:---|:---|:---|
| **Root** | `awesome_cleft_pipeline.py` | Top-level CLI entry point. Parses command-line arguments, instantiates `AwesomeCleftPipeline`, executes the full workflow, and prints the report. |
| **Root** | `requirements.txt` | Defines all required Python dependencies (PyTorch, transformers, mlmorph, simalign, pandas). |
| **Root** | `main.tex` | Master LaTeX file for compiling the academic project report and thesis document. |
| **Root** | `references.bib` | BibTeX citations for `mlmorph`, `SimAlign`, `Awesome-Align`, and linguistic reference grammars. |
| **`aligner/`** | `simalign_wrapper.py` | Wraps `simalign.SentenceAligner`. Extracts mBERT Layer 8 representations and runs `itermax`, `argmax`, or `match` decoding to return alignment index pairs `(src_idx, tgt_idx)`. |
| **`aligner/`** | `morpho_simalign.py` | Implementation of `MorphoSimAligner`. Uses `mlmorph` to decompose Malayalam compounds and strip case clitics before passing clean roots to SimAlign, then remaps indices to surface tokens. |
| **`aligner/`** | `alignment_utils.py` | Helper routines: punctuation stripping, index normalization, token matching, and extraction of alignment coordinates. |
| **`aligner/`** | `tokenizer.py` | Malayalam Unicode tokenizer that handles chillu letters, viramas, and zero-width joiners properly. |
| **`aligner/`** | `english_phrase_chunker.py` | Uses syntactic rules to chunk English multi-word units (NP/VP/PP) so compound focus words (like *"two books"*) are aligned as units. |
| **`aligner/`** | `syntax_guided_aligner.py` | Experimental aligner that uses part-of-speech tags to constrain allowed word alignment matches. |
| **`aligner/`** | `bert_pos_utils.py` | Helper functions for mapping BERT subwords back to full words and extracting token features. |
| **`cleft/`** | `awesome_cleft_pipeline.py` | The main controller class (`AwesomeCleftPipeline`). Chains Block 1 through Block 6 together: alignment $\to$ focus projection $\to$ chunking $\to$ copula attachment $\to$ verb nominalization $\to$ validation. |
| **`cleft/`** | `cleft_pipeline.py` | The core clefting engine (`CleftPipeline`). Contains the sandhi transformation rules (`apply_sandhi`), verb replacement logic, and sentence reconstruction routines. |
| **`cleft/`** | `copula_pipeline.py` | Manages host constituent analysis (`CopulaPipeline`). Checks if the focus word carries dative, locative, or accusative markers and binds \ml{ആണ്} without stripping the case. |
| **`cleft/`** | `phrase_chunker.py` | Analyzes words adjacent to the focus using `mlmorph` POS tags. Merges demonstratives, adjectives, and postpositions to form complete phrase spans. |
| **`cleft/`** | `preverbal_focus_reorderer.py` | Alternative/baseline module. Implements preverbal constituent scrambling (moving the focus word immediately before the verb instead of clefting). |
| **`cleft/`** | `bhashik_focus_reorderer.py` | Focus reordering implementation designed for compatibility with Bhashik/SSF dependency formats. |
| **`cleft/`** | `multi_target_emphasis_generator.py` | Tool that takes a sentence, identifies all valid constituent words, and generates all possible cleft variations for each candidate word. |
| **`cleft/`** | `ir.py` | Defines Intermediate Representation (IR) data classes (`CleftState`, `TokenInfo`) to pass state between pipeline blocks cleanly. |
| **`cleft/`** | `malayalam_pipeline.py` | Lightweight standalone script to run clefting directly when the Malayalam focus word is already specified. |
| **`verb_norm/`**| `verb_normalizer.py` | The matrix verb normalizer (`VerbNormalizer`). Takes finite verbs, uses `mlmorph.Analyser` to extract TAM features, and uses `mlmorph.Generator` to output nominalized \ml{-ത്} forms across 7 verb classes. Includes the double-nominalization guard. |
| **`verb_norm/`**| `mlmorph_utils.py` | Helper wrappers for interacting with `mlmorph` transducers, extracting feature dictionaries from tag strings, and formatting tag queries. |
| **`evaluator/`**| `benchmark_aligners_cleft_comprehensive.py`| Comprehensive evaluation benchmark comparing Raw SimAlign, Morpho-SimAlign, and Syntactic Aligner across test datasets. |
| **`evaluator/`**| `eval_morpho_focused.py` | Evaluates focus projection accuracy specifically on sentences with complex morphology (compounds, case clitics). |
| **`evaluator/`**| `benchmark_alignment_modalities.py` | Evaluates the differences between SimAlign matching methods (`itermax` vs. `argmax` vs. `match`). |
| **`evaluator/`**| `run_dataset_eval.py` | Evaluates a dataset CSV and records focus accuracy, cleft success rate, and exact matches. |
| **`evaluator/`**| `export_mismatches.py` | Filters predictions that differ from gold-standard references and saves them for linguistic error analysis. |
| **`evaluator/`**| `recalculate_metrics.py` | Recalculates precision, recall, exact matches, and normalized matches from evaluation logs. |
| **`unit_testing/`**| `test_multiword_align.py` | Automated tests verifying multi-word alignment, equative sentences, sandhi rules, and verb normalizations. |

---

## 4. Prerequisites & Installation

### 1. Requirements
* **Operating System:** Windows 10/11 or Linux (Ubuntu 20.04+).
* **Python:** 3.10 or 3.11.
* **C++ Compiler:** Required to build the HFST finite-state transducer bindings used by `mlmorph`.
  * *On Windows:* Install Visual Studio Community or Visual Studio Build Tools with the **"Desktop development with C++"** workload selected.
  * *On Linux:* `sudo apt-get install build-essential python3-dev`

### 2. Setup Steps

```bash
# 1. Clone this repository
git clone https://github.com/savioshaju/Prosody-Transfer.git
cd Prosody-Transfer

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 4. Install all dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 5. Verify the installation
python -c "import mlmorph, simalign, torch; print('Environment is ready!')"
```

---

## 5. How to Run (CLI & Python API)

### Option A: Command-Line Interface (CLI)

Run `awesome_cleft_pipeline.py` from the root of the project:

```bash
python awesome_cleft_pipeline.py `
  --en "Raman gave Krishnan two books yesterday." `
  --ml "രാമൻ ഇന്നലെ കൃഷ്ണന് രണ്ട് പുസ്തകം കൊടുത്തു." `
  --en_focus "two books" `
  --aligner simalign
```

#### CLI Options:
* `--en` (`-e`): The source English sentence.
* `--ml` (`-m`): The corresponding parallel Malayalam translation.
* `--en_focus` (`-ef`): The specific word or phrase in the English sentence that bears emphasis.
* `--aligner` (`-a`): Which aligner to use (`simalign` or `morpho_simalign`). Default: `simalign`.
* `--aligner_method` (`-am`): Decoding method for SimAlign (`itermax`, `argmax`, or `match`). Default: `itermax`.
* `--output` (`-out`): (Optional) File path to export results as a JSON or CSV file (e.g. `--output results.json`).

---

### Option B: Programmatic Python API

You can easily import and call the pipeline directly from Python:

```python
from cleft.awesome_cleft_pipeline import AwesomeCleftPipeline

# 1. Initialize the pipeline
pipeline = AwesomeCleftPipeline(
    aligner_type="simalign",
    aligner_method="itermax"
)

# 2. Process a parallel sentence pair
result = pipeline.process(
    english_sentence="Raman gave Krishnan two books yesterday.",
    malayalam_sentence="രാമൻ ഇന്നലെ കൃഷ്ണന് രണ്ട് പുസ്തകം കൊടുത്തു.",
    english_focus="two books",
    operation="cleft"
)

# 3. Inspect the outputs
print("Projected Malayalam Focus :", result["malayalam_focus"])
print("Synthesized Cleft Host    :", result["cleft_host"])
print("Nominalized Matrix Verb   :", result["nominalized_verb"])
print("Final Cleft Sentence      :", result["cleft_sentence"])
print("Pipeline Status           :", result["status"])
```

---

## 6. Input & Output Specifications

### Input Format
The system takes three parameters:
1. **English Sentence:** Plain English text (e.g., `"The doctor examined the patient carefully."`).
2. **Malayalam Sentence:** Parallel Malayalam translation (e.g., `"ഡോക്ടർ രോഗിയെ ശ്രദ്ധാപൂർവ്വം പരിശോധിച്ചു."`).
3. **English Focus:** The exact substring in the English sentence bearing emphasis (e.g., `"The doctor"`).

### Output Format

#### 1. Visual Console Report
When running from the command line, the system prints a formatted execution summary:

```
================================================================================
           CROSS-LINGUAL FOCUS TRANSFER PIPELINE REPORT
================================================================================
English Sentence   : Raman gave Krishnan two books yesterday.
English Focus      : two books
Target Malayalam   : രാമൻ ഇന്നലെ കൃഷ്ണന് രണ്ട് പുസ്തകം കൊടുത്തു.
Aligner Used       : simalign (itermax)
Projected Focus    : രണ്ട് പുസ്തകം
Syntactic Chunk    : രണ്ട് പുസ്തകം (NP)
Cleft Host         : രണ്ട് പുസ്തകമാണ് (Sandhi: ം + ആണ് → മാണ്)
Matrix Verb        : കൊടുത്തു → കൊടുത്തത് (Past Participial Nominalization)
--------------------------------------------------------------------------------
Output Sentence    : രാമൻ ഇന്നലെ കൃഷ്ണന് രണ്ട് പുസ്തകമാണ് കൊടുത്തത്.
Status             : SUCCESS
================================================================================
```

#### 2. JSON Dictionary Format
When called programmatically or saved to a JSON file:

```json
{
  "english_sentence": "Raman gave Krishnan two books yesterday.",
  "english_focus": "two books",
  "malayalam_sentence": "രാമൻ ഇന്നലെ കൃഷ്ണന് രണ്ട് പുസ്തകം കൊടുത്തു.",
  "malayalam_focus": "രണ്ട് പുസ്തകം",
  "cleft_sentence": "രാമൻ ഇന്നലെ കൃഷ്ണന് രണ്ട് പുസ്തകമാണ് കൊടുത്തത്.",
  "cleft_host": "രണ്ട് പുസ്തകമാണ്",
  "nominalized_verb": "കൊടുത്തത്",
  "alignment_pairs": [[3, 3], [4, 4]],
  "sandhi_rule": "anusvara_resolution",
  "status": "VALID"
}
```

---

## 7. Running Tests & Benchmarks

To verify the system or run batch evaluations:

```bash
# Run unit tests (tests sandhi rules, chunking, and verb nominalization)
python -m unittest discover -s unit_testing

# Run benchmark across alignment models on the evaluation dataset
python evaluator/benchmark_aligners_cleft_comprehensive.py

# Evaluate focus projection accuracy on complex morphological forms
python evaluator/eval_morpho_focused.py
```

---

## Academic Citation

If you use this work, please cite the underlying foundational toolkits:

```bibtex
@inproceedings{thottingal2019finite,
  title     = {Finite State Transducer based Morphology analysis for {M}alayalam Language},
  author    = {Thottingal, Santhosh},
  booktitle = {Proceedings of the 2nd Workshop on Technologies for MT of Low Resource Languages},
  pages     = {1--5},
  year      = {2019}
}

@inproceedings{sabet2020simalign,
  title     = {{SimAlign}: High Quality Word Alignments without Parallel Training by Exploiting Multilingual {BERT}},
  author    = {Jalili Sabet, Masoud and Dufter, Philipp and Yvon, Fran{\c{c}}ois and Sch{\"u}tze, Hinrich},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  pages     = {1627--1643},
  year      = {2020}
}
```
