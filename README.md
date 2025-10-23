# Natural Food Corpus — Vertical Search Engine

A focused vertical search engine built for the Natural Food Corpus. This project provides a simple Streamlit web UI for document search backed by a TF-IDF based retrieval pipeline, tokenization and normalization via spaCy, and optional spell correction using a transformers model.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-orange.svg)](#)

---

## Key Features

- Build an inverted index from a JSONL corpus file.
- TF-IDF based retrieval that combines title and body signals.
- Streamlit web interface for interactive search and result inspection (`app/streamlit_app.py`).
- Text processing pipeline using spaCy (tokenization, stopword removal, lemmatization).
- Optional spelling correction using Hugging Face `transformers` models.
- Evaluation utilities for mPrecision@k and mAP (`src/evaluator.py`).

---

## Expected Output

- `inverted_index.json` generated in the repository root after running the index builder.
- Streamlit UI available at http://localhost:8501 (default) after launching the app.

---

## System Requirements

- Windows 10/11 (guide below uses `cmd.exe`)
- Python 3.10 or 3.11 recommended
- At least 8GB RAM; more required when loading larger transformer models

---

## Quick Start (Windows)

This repository includes an enhanced setup script `set_up.bat` with a beautiful terminal UI that automates the entire workflow:

### **Features:**
- 🎨 **Colorful terminal interface** with progress indicators
- 🔧 **Virtual environment** creation and activation
- 📦 **Dependency installation** from `requirements.txt` (falls back to `requirements_optimized.txt`)
- 🧠 **spaCy model** installation (`en_core_web_sm`) with automatic fallback
- ⚙️ **ENV.json configuration** (preserves existing or creates from `ENV.json.exp`)
- 🏗️ **Inverted index building** with progress feedback
- 📊 **Optional evaluation** with custom query count input
- 🚀 **Streamlit launch** with browser integration

### **Usage Options:**

**Full workflow (recommended for first-time setup):**
```cmd
cd /d d:\Project\Code\nf_search_engine
set_up.bat
```

**Setup and build only (no Streamlit launch):**
```cmd
set_up.bat --no-run
```

**Skip evaluation step:**
```cmd
set_up.bat --skip-eval
```

**Minimal setup (no evaluation, no Streamlit):**
```cmd
set_up.bat --no-run --skip-eval
```

### **Interactive Features:**
- **ENV.json Configuration:** The script will prompt you to edit `ENV.json` via Notepad if needed
- **Evaluation Options:** Choose whether to run evaluation and specify number of queries (e.g., 10, 50, 100)
- **Progress Feedback:** Color-coded status messages ([INFO], [SUCCESS], [WARN], [ERROR])
- **Error Handling:** Detailed error messages with helpful troubleshooting tips

---

## Environment Configuration

The project uses an `ENV.json` configuration file (example provided as `ENV.json.exp`). Typical fields:

- `CORPUS_PATH` — path to the JSONL corpus file
- `INVERTED_INDEX_PATH` — path to save/load the inverted index (default: `inverted_index.json`)
- `QUERIES_PATH` — path to queries JSONL
- `QRELS_PATH` — path to qrels CSV
- `MODEL_PATH` — Hugging Face model ID or local path for spelling correction
- `EVALUATION_RESULT_PATH` — path to save evaluation plots

Example `ENV.json` snippet:

```json
{
  "CORPUS_PATH": "data/nfcorpus/corpus.jsonl",
  "INVERTED_INDEX_PATH": "inverted_index.json",
  "QUERIES_PATH": "data/nfcorpus/queries.jsonl",
  "QRELS_PATH": "data/nfcorpus/qrels/dev.csv",
  "MODEL_PATH": "oliverguhr/spelling-correction-english-base",
  "EVALUATION_RESULT_PATH": "results/mPrecision_at_k_and_mAP.png"
}
```

---

## Manual Steps (for debugging)

1. Create and activate the virtual environment:

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

2. Install dependencies:

```cmd
pip install -r requirements.txt
```

3. Install the spaCy model (if not already installed):

```cmd
python -m spacy download en_core_web_sm
```

Fallback (wheel):

```cmd
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

4. Build the inverted index:

```cmd
python src\build_inverted_index.py
```

5. Run the Streamlit app:

```cmd
streamlit run app\streamlit_app.py
```

---

## Troubleshooting

- ImportError for `en_core_web_sm`: ensure `spacy` is installed in the active venv and `en_core_web_sm` is installed with a compatible version.
- Transformers model fails to download: set `MODEL_PATH` in `ENV.json` to a local model directory or ensure your environment has internet access.
- Missing packages after `pip install`: verify you installed into the activated virtualenv (`pip show <package>`).
- If `set_up.bat` opens Notepad for `ENV.json`, edit and save the file before continuing the script.

---

## Project Layout

- `app/streamlit_app.py` — Streamlit UI
- `src/build_inverted_index.py` — inverted index builder
- `src/searcher.py` — retrieval logic & snippet generation
- `src/evaluator.py` — evaluation & plotting utilities
- `utils/` — helper modules: loader, processor (spaCy & transformers), creator, caculator
- `data/` — corpus, queries, and qrels
- `results/` — generated plots and outputs

---

## Contributing & License

- License: MIT (badge at top)
- Contributions: fork the repository, create a feature branch and submit a pull request. Please include a short description of your changes and relevant tests or examples.

---

If you want, I can also:
- Embed the header image you provided and add a polished cover section to the README
- Add a `CONTRIBUTING.md` with contribution guidelines
- Add a small CI workflow (GitHub Actions) for linting and running unit tests

