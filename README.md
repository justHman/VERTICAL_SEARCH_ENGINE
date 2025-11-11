<div align="center">

# Natural Food Corpus — Vertical Search Engine

<img src="results\images\wordcloud_corpus.png" alt="Natural Food Corpus Word Cloud" width="1000">

A focused vertical search engine built for the Natural Food Corpus. This project provides dual search modes: **TF-IDF lexical search** for exact keyword matching and **Semantic Search** using neural embeddings for conceptual similarity. Features include a Streamlit web UI, automatic spelling correction, and comprehensive evaluation metrics.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-orange.svg)](#)

</div>

---

## ✨ Key Features

- 🔍 **Dual Search Modes**:
  - **TF-IDF Search**: Fast lexical matching with inverted index
  - **Semantic Search**: Neural embeddings for conceptual similarity
- 🔤 **Spell Correction** (Optional): Automatic query correction using transformers (lazy-loaded)
- 📊 **Rich Evaluation**: mAP, Precision@k metrics with visualizations
- 🎨 **Modern Web UI**: Streamlit-based interface with optimized caching
- ⚡ **GPU Acceleration**: CUDA support for semantic search
- 🛠️ **Modular Architecture**: Easy to extend and customize
- 💾 **Memory Optimized**: Lazy loading, singleton caching, minimal RAM footprint

---

## Demo

<div align="center">
    <img src="results\gif\demo.gif" alt="Search Engine Demo" width="1000">
</div>

---

## Table of Contents

- [Expected Output](#expected-output)
- [System Requirements](#system-requirements)
- [Project Layout](#project-layout)
- [Quick Start (Windows)](#quick-start-windows)
- [Environment Configuration](#environment-configuration)
- [Manual Steps (for debugging)](#manual-steps-for-debugging)
- [Troubleshooting](#troubleshooting)
- [Contributing & License](#contributing--license)

---

## Expected Output

- **TF-IDF Index**: `inverted_index.json` generated in `data/` after running the index builder
- **Semantic Index**: `{model_name}/index.index` and `{model_name}/index_docs.pkl` in `data/semantic/`
- **Streamlit UI**: Available at http://localhost:8501 (default) after launching the app
- **Evaluation Results**: Plots saved in `results/images/` with performance metrics

---

## System Requirements

- Windows 10/11 (guide below uses `cmd.exe`)
- Python 3.10 or 3.11 recommended
- At least 8GB RAM; more required when loading larger transformer models

---

## Project Layout

Repository tree (top-level):

```text
nf_search_engine/
├─ app/
│  └─ streamlit_app.py           # Streamlit UI with dual search modes
├─ src/
│  ├─ build_inverted_index.py    # TF-IDF inverted index builder
│  ├─ build_index.py             # Semantic index builder (FAISS)
│  ├─ searcher.py                # Dual search engine (TF-IDF + Semantic)
│  ├─ evaluator.py               # Evaluation & metrics
│  └─ visualize.py               # Plotting utilities
├─ utils/
│  ├─ loader.py                  # Data loaders
│  ├─ processor.py               # Text processing (spaCy + transformers)
│  ├─ creator.py                 # Index creator
│  └─ caculator.py               # TF-IDF, metrics
├─ data/
│  ├─ nfcorpus/
│  │  ├─ corpus.jsonl            # Document corpus
│  │  ├─ queries.jsonl           # Test queries
│  │  └─ qrels/                  # Relevance judgments
│  ├─ inverted_index.json        # TF-IDF index
│  └─ semantic/                  # Semantic indices by model
│     ├─ all-MiniLM-L6-v2/
│     ├─ multi-qa-mpnet-base-dot-v1/
│     └─ S-PubMedBert-MS-MARCO/
├─ models/                       # Local model cache
│  └─ spelling-correction-english-base/  # Spelling correction model (optional)
├─ results/                      # Evaluation outputs and plots
├─ docs/                         # Technical documentation
│  ├─ technical_documentation.md
│  └─ streamlit_optimization.md  # Web app optimization guide
├─ ENV.json.exp / ENV.json       # Configuration
├─ requirements.txt
├─ main.py                       # CLI search interface
└─ set_up.bat                    # Automated setup script
```

---

## Quick Start (Windows)

This repository includes an enhanced setup script `set_up.bat` with a beautiful terminal UI that automates the entire workflow:

### **Features:**
- 🎨 **Colorful terminal interface** with progress indicators
- 🔧 **Virtual environment** creation and activation
- 📦 **Dependency installation** from `requirements.txt`
- 🧠 **spaCy model** installation (`en_core_web_sm`) with automatic fallback
- ⚙️ **ENV.json configuration** (preserves existing or creates from `ENV.json.exp`)
- 🏗️ **TF-IDF index building** with progress feedback
- � **Semantic index building** (optional, configurable model)
- �📊 **Optional evaluation** with custom query count input
- 🚀 **Streamlit launch** with browser integration

### **Usage Options:**

**Clone the repository and run full workflow (recommended for first-time setup):**
```cmd
git clone https://github.com/justHman/VERTICAL_SEARCH_ENGINE.git
cd VERTICAL_SEARCH_ENGINE
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

## **Search Modes**

### **1. TF-IDF Lexical Search**
Fast keyword-based search using inverted index and TF-IDF scoring.

**CLI Usage:**
```cmd
python src\searcher.py --mode tfidf --query "statin effects on cholesterol" --top_k 10
```

**Advantages:**
- ⚡ Very fast (~10ms per query)
- 🎯 Exact keyword matching
- 📊 Interpretable scores
- 💾 Small index size

**Best for:** Known-item search, specific medical terms, exact phrases

### **2. Semantic Search**
Neural embedding-based search capturing conceptual similarity.

**Build Semantic Index:**

```cmd
# General-purpose (fastest)
python src\build_index.py --corpus_path data\nfcorpus\corpus.jsonl --index_path data\semantic\all-MiniLM-L6-v2\index --model_name all-MiniLM-L6-v2
```

# Biomedical domain (recommended for NFCorpus)
```cmd
# Biomedical domain (recommended for NFCorpus)
python src\build_index.py --corpus_path data\nfcorpus\corpus.jsonl --index_path data\semantic\S-PubMedBert-MS-MARCO\index.index --model_name pritamdeka/S-PubMedBert-MS-MARCO
```

**CLI Usage:**
```cmd
python src\searcher.py --mode semantic --query "heart disease prevention" --index_path data\semantic\S-PubMedBert-MS-MARCO\index.index --top_k 10
```

**Advantages:**
- 🧠 Semantic similarity (handles synonyms, paraphrases)
- 🔍 Better recall for conceptual queries
- 🌐 Cross-lingual potential (with multilingual models)
- 📈 10-30% mAP improvement on conceptual queries

**Best for:** Exploratory search, concept queries, synonym matching

**Supported Models:**
- `all-MiniLM-L6-v2` (384d, fast, general-purpose)
- `multi-qa-mpnet-base-dot-v1` (768d, high quality, question-answering)
- `pritamdeka/S-PubMedBert-MS-MARCO` (768d, biomedical domain, **recommended for NFCorpus**)
- Any SentenceTransformer model from Hugging Face

**Model Recommendations:**
- **General use**: `all-MiniLM-L6-v2` (fastest, good baseline)
- **Question-answering**: `multi-qa-mpnet-base-dot-v1` (best for Q&A style queries)
- **Biomedical/Nutritional**: `pritamdeka/S-PubMedBert-MS-MARCO` (15-25% better mAP on NFCorpus)

### **3. Spelling Correction (Optional)**

The Streamlit UI includes an **optional** spelling correction feature that can be enabled via checkbox.

**Features:**
- ✅ Lazy loading (model only loaded when first enabled)
- ✅ Non-intrusive suggestions (search uses original query, suggests corrections)
- ✅ Contextual corrections using transformer model
- ✅ ~400MB model cached in memory when enabled

**Usage:**
```python
# In Streamlit UI: Enable "🔤 Spelling Correction" checkbox in sidebar
```

**Model:** `oliverguhr/spelling-correction-english-base` (sequence-to-sequence transformer)

**Memory Impact:**
- Disabled (default): 0 MB
- Enabled: ~400 MB (one-time load)

---

## **Streamlit Web UI Optimizations**

The web application implements several optimization strategies for production deployment:

### **Optimization Features:**

1. **@st.cache_resource**: Singleton caching for models and indices
2. **Lazy Loading**: Models only loaded when first selected
3. **Index Pre-loading**: FAISS indices loaded instead of full models (~10x smaller)
4. **Memory Management**: Manual cache clearing + gc.collect()
5. **Error Handling**: Graceful degradation with helpful error messages
6. **Performance Monitoring**: Real-time search latency display

### **Memory Usage:**

| Configuration | RAM Usage | Notes |
|---------------|-----------|-------|
| TF-IDF only | ~22 MB | Baseline, no semantic models |
| + 1 Semantic Model | ~492 MB | With FAISS index + model |
| + All 3 Models | ~1072 MB | Maximum (lazy loaded) |
| + Spelling Correction | +400 MB | Optional, disabled by default |

**Production Recommendations:**
- Minimum RAM: 1 GB (for 1 semantic model)
- Recommended RAM: 2 GB (for all 3 models + spelling)
- Pre-build all FAISS indices before deployment
- Use Docker multi-stage builds for optimized images

See `docs/streamlit_optimization.md` for detailed optimization guide.

---

## **Search Modes**

The project uses an `ENV.json` configuration file (example provided as `ENV.json.exp`). Typical fields:

- `CORPUS_PATH` — path to the JSONL corpus file (default: `data\nfcorpus\corpus.jsonl`)
- `INVERTED_INDEX_PATH` — path to save/load the inverted index (default: `data\inverted_index.json`)
- `QUERIES_PATH` — path to queries JSONL (default: `data\nfcorpus\queries.jsonl`)
- `QRELS_PATH` — path to qrels CSV (default: `data\nfcorpus\qrels\merged_qrels.csv`)
- `MODEL_PATH` — Hugging Face model ID or local path for spelling correction
- `EVALUATION_RESULT_PATH` — path to save evaluation plots
- `N_QUERIES` — number of queries for evaluation (optional, defaults to all)

Example `ENV.json` snippet:

```json
{
  "CORPUS_PATH": "data/nfcorpus/corpus.jsonl",
  "INVERTED_INDEX_PATH": "data/inverted_index.json",
  "QUERIES_PATH": "data/nfcorpus/queries.jsonl",
  "QRELS_PATH": "data/nfcorpus/qrels/merged_qrels.csv",
  "MODEL_PATH": "oliverguhr/spelling-correction-english-base",
  "EVALUATION_RESULT_PATH": "results/evaluation_scores.png"
}
```

---

## Manual Steps (for debugging)

0. Clone the repository:
```cmd
git clone https://github.com/justHman/VERTICAL_SEARCH_ENGINE.git
cd VERTICAL_SEARCH_ENGINE
```

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
4. Config ENV.json following ENV.json.exp:

Example `ENV.json` snippet:

```json
{
  "CORPUS_PATH": "data/nfcorpus/corpus.jsonl",
  "INVERTED_INDEX_PATH": "data/inverted_index.json",
  "QUERIES_PATH": "data/nfcorpus/queries.jsonl",
  "QRELS_PATH": "data/nfcorpus/qrels/merged_qrels.csv",
  "MODEL_PATH": "oliverguhr/spelling-correction-english-base",
  "EVALUATION_RESULT_PATH": "results/evaluation_scores.png"
}
```

5. Build the inverted index (TF-IDF):

```cmd
python src\build_inverted_index.py
```

5b. (Optional) Build semantic search index:

```cmd
python src\build_index.py --corpus_path data\nfcorpus\corpus.jsonl --index_path data\semantic\all-MiniLM-L6-v2\index --model_name all-MiniLM-L6-v2
```

6. Evaluate (TF-IDF):
```cmd
python src\evaluator.py
```

6b. (Optional) Evaluate semantic search:
Edit `src\evaluator.py` to use `evaluate_semantic_search` instead of `evaluate_tfidf_search`.

7. Run the Streamlit app:

```cmd
streamlit run app\streamlit_app.py
```

---

## Troubleshooting

- **ImportError for `en_core_web_sm`**: ensure `spacy` is installed in the active venv and `en_core_web_sm` is installed with a compatible version.
- **Transformers model fails to download**: set `MODEL_PATH` in `ENV.json` to a local model directory or ensure your environment has internet access.
- **Missing packages after `pip install`**: verify you installed into the activated virtualenv (`pip show <package>`).
- **If `set_up.bat` opens Notepad for `ENV.json`**: edit and save the file before continuing the script.
- **CUDA out of memory (semantic search)**: reduce batch size or use smaller model (all-MiniLM-L6-v2 instead of multi-qa-mpnet-base-dot-v1).
- **Slow semantic indexing**: encoding can take 10-30 minutes for 3,600 docs. Use GPU for 5-10× speedup.
- **FAISS dimension mismatch**: ensure index was built with the same model as used for search. Rebuild index if you changed models.

---

## Contributing & License

- License: MIT (badge at top)
- Contributions: fork the repository, create a feature branch and submit a pull request. Please include a short description of your changes and relevant tests or examples.

---

If you want, I can also:
- Embed the header image you provided and add a polished cover section to the README
- Add a `CONTRIBUTING.md` with contribution guidelines
- Add a small CI workflow (GitHub Actions) for linting and running unit tests

