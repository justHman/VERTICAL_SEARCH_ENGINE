import streamlit as st
import os
import sys
import gc
import time as time_module

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from utils.loader import load_corpus, load_inverted_index, load_env
from src.searcher import tfidf_search, get_in4, semantic_search
from utils.processor import Text2Tokens, correct_text, normalize_text

ENV = load_env()

# ================== OPTIMIZATION STRATEGIES ==================
# 1. Cache resources to avoid reloading
# 2. Lazy loading - only load when needed
# 3. Memory-efficient index storage
# 4. Singleton pattern for models
# =============================================================

# Available semantic models configuration
SEMANTIC_MODELS = {
    "tfidf": {
        "name": "TF-IDF (Lexical)",
        "display": "🔤 TF-IDF",
        "description": "Fast keyword-based search",
        "requires_model": False,
        "memory": "~50MB",
        "speed": "Very Fast (10-50ms)"
    },
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "display": "⚡ MiniLM",
        "path": "data/semantic/all-MiniLM-L6-v2/index.index",
        "description": "384d, fastest semantic search",
        "requires_model": True,
        "memory": "~80MB",
        "speed": "Fast (100-200ms)"
    },
    "multi-qa-mpnet": {
        "name": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "display": "🎯 MPNet Q&A",
        "path": "data/semantic/multi-qa-mpnet-base-dot-v1/index.index",
        "description": "768d, optimized for questions",
        "requires_model": True,
        "memory": "~420MB",
        "speed": "Medium (150-300ms)"
    },
    "pubmedbert": {
        "name": "NeuML/pubmedbert-base-embeddings",
        "display": "🧬 PubMedBERT",
        "path": "data/semantic/S-PubMedBert-MS-MARCO/index.index",
        "description": "768d, best for biomedical queries",
        "requires_model": True,
        "memory": "~420MB",
        "speed": "Medium (150-300ms)"
    }
}

@st.cache_resource(show_spinner=False)
def load_corpus_cached():
    """Cache corpus in memory - loaded once per session"""
    return load_corpus(ENV["CORPUS_PATH"])

@st.cache_resource(show_spinner=False)
def load_inverted_index_cached():
    """Cache inverted index - loaded once per session"""
    return load_inverted_index(ENV["INVERTED_INDEX_PATH"])

@st.cache_resource(show_spinner=False)
def get_semantic_engine(_corpus, model_key):
    """
    Lazy load semantic search engine with caching.
    Uses leading underscore to avoid hashing large corpus object.
    Only creates engine when first requested for a model.
    """
    if model_key not in SEMANTIC_MODELS or model_key == "tfidf":
        return None
    
    model_info = SEMANTIC_MODELS[model_key]
    model_path = model_info["path"]
    
    # Check if index exists
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Index not found: `{model_path}`")
        return None
    
    try:
        # Create engine and load pre-built index (much faster than loading model)
        engine = semantic_search(_corpus, model_name=model_info["name"])
        engine.load_index(model_path)
        return engine
    except Exception as e:
        st.error(f"❌ Failed to load {model_info['display']}: {str(e)}")
        return None

def check_index_availability(model_key):
    """Check if semantic index exists without loading it"""
    if model_key == "tfidf" or model_key not in SEMANTIC_MODELS:
        return True
    return os.path.exists(SEMANTIC_MODELS[model_key]["path"])

# Page configuration
st.set_page_config(
    page_title="Natural Food Corpus Search Engine",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .header-subtitle {
        color: #e0e0e0;
        font-size: 1.2rem;
        text-align: center;
    }
    
    /* Search box styling */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 8px;
        border: 2px solid #667eea;
    }
    
    /* Result card styling */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    
    .result-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .result-title a {
        color: #667eea;
        text-decoration: none;
        transition: color 0.2s;
    }
    
    .result-title a:hover {
        color: #764ba2;
        text-decoration: underline;
    }
    
    .result-score {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    
    .result-id {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .result-snippet {
        color: #333;
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 0.75rem;
    }
    
    /* Highlight styling */
    mark {
        background-color: #ffd700;
        padding: 0.2rem 0.3rem;
        border-radius: 3px;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Stats box */
    .stats-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .stats-label {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    
    .stats-value {
        color: #333;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Search button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-size: 1.1rem;
        width: 100%;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Example queries */
    .example-query {
        background: #f0f0f0;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem;
        cursor: pointer;
        font-size: 0.9rem;
        transition: background 0.2s;
    }
    
    .example-query:hover {
        background: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with optimized loading
if 'corpus' not in st.session_state:
    with st.spinner('🔄 Loading corpus data...'):
        st.session_state.corpus = load_corpus_cached()
        st.session_state.inverted_index = load_inverted_index_cached()
        st.session_state.total_docs = len(st.session_state.corpus)
        st.session_state.id2doc = {doc['_id']: doc for doc in st.session_state.corpus}

if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

if 'search_mode' not in st.session_state:
    st.session_state.search_mode = "tfidf"

if 'last_search_time' not in st.session_state:
    st.session_state.last_search_time = 0

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🥗 Natural Food Corpus Search Engine</div>
    <div class="header-subtitle">Explore nutrition research and dietary information</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### � Search Mode")
    
    # Model selection with status indicators
    model_options = []
    model_keys = []
    for key, info in SEMANTIC_MODELS.items():
        available = check_index_availability(key)
        status = "✅" if available else "❌"
        label = f"{status} {info['display']}"
        model_options.append(label)
        model_keys.append(key)
    
    selected_idx = st.selectbox(
        "Select search algorithm:",
        range(len(model_options)),
        format_func=lambda i: model_options[i],
        index=model_keys.index(st.session_state.search_mode) if st.session_state.search_mode in model_keys else 0,
        help="✅ = Ready | ❌ = Index not built"
    )
    
    selected_model = model_keys[selected_idx]
    st.session_state.search_mode = selected_model
    
    # Show model info
    model_info = SEMANTIC_MODELS[selected_model]
    with st.expander("ℹ️ Model Information", expanded=False):
        st.markdown(f"""
        **Description:** {model_info['description']}  
        **Memory:** {model_info.get('memory', 'N/A')}  
        **Speed:** {model_info.get('speed', 'N/A')}
        """)
        
        if selected_model != "tfidf" and not check_index_availability(selected_model):
            st.error(f"⚠️ Index not found. Build it with:")
            st.code(f"python src/build_index.py --model_name {model_info['name']}", language="bash")
    
    st.markdown("---")
    st.markdown("###  Search Settings")
    
    num_results = st.slider(
        "Number of results",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )
    
    # Spelling correction toggle
    use_spelling_correction = st.checkbox(
        "🔤 Spelling Correction",
        value=False,
        help="Enable spelling correction (loads ~400MB model on first use)"
    )
    
    # TF-IDF specific settings
    if selected_model == "tfidf":
        alpha = st.slider(
            "TF-IDF Alpha (title weight)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values give more weight to title matches"
        )
    else:
        alpha = 0.7  # Default for non-TF-IDF modes
    
    max_chars = st.slider(
        "Max snippet length",
        min_value=100,
        max_value=500,
        value=300,
        step=50
    )
    
    window = st.slider(
        "Context window",
        min_value=5,
        max_value=20,
        value=11,
        step=1,
        help="Number of words around matched terms"
    )
    
    st.markdown("---")
    st.markdown("### 📈 Statistics")
    
    st.markdown(f"""
    <div class="stats-box">
        <div class="stats-label">Total Documents</div>
        <div class="stats-value">{st.session_state.total_docs:,}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stats-box">
        <div class="stats-label">Active Mode</div>
        <div class="stats-value" style="font-size: 1.2rem;">{model_info['display']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.last_search_time > 0:
        st.markdown(f"""
        <div class="stats-box">
            <div class="stats-label">Last Search Time</div>
            <div class="stats-value" style="font-size: 1.2rem;">{st.session_state.last_search_time:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)
    

# Main content
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    # Example queries
    st.markdown("### 💡 Example Queries")
    example_queries = [
        "statin effects on cholesterol",
        "vitamin D deficiency symptoms",
        "omega-3 fatty acids benefits",
        "gluten intolerance diet",
        "calcium absorption factors"
    ]
    
    cols = st.columns(len(example_queries))
    for idx, example in enumerate(example_queries):
        with cols[idx]:
            if st.button(f"📝 {example[:20]}...", key=f"example_{idx}", use_container_width=True):
                st.session_state.current_query = example
    
    st.markdown("---")
    
    # Search interface
    st.markdown("### 🔍 Search Query")
    query = st.text_input(
        "Enter your search query:",
        value=st.session_state.current_query,
        placeholder="e.g., statin effects on cholesterol",
        label_visibility="collapsed"
    )
    
    col_search1, col_search2, col_search3 = st.columns([3, 1, 3])
    with col_search2:
        search_button = st.button("🔎", type="primary", use_container_width=True)
    
    # Perform search with optimization
    if search_button and query:
        st.session_state.current_query = query
        
        # Check if index is available
        if not check_index_availability(selected_model):
            st.error(f"❌ Index for {model_info['display']} not found. Please build it first.")
        else:
            start_time = time_module.time()
            
            with st.spinner(f'🔍 Searching with {model_info["display"]}...'):
                # Always use original query for search (respect user intent)
                search_query = query
                corrected_query = query  # Will be updated if spell check enabled
                
                try:
                    if selected_model == "tfidf":
                        # TF-IDF Search
                        tokens = Text2Tokens(search_query)
                        
                        if not tokens:
                            st.warning("⚠️ Please enter a valid search query.")
                        else:
                            results = tfidf_search(
                                tokens,
                                st.session_state.inverted_index,
                                st.session_state.total_docs,
                                n=num_results,
                                alpha=alpha
                            )
                            
                            end_time = time_module.time()
                            st.session_state.last_search_time = end_time - start_time
                            
                            # Get spelling correction suggestion (after search, non-blocking)
                            if use_spelling_correction:
                                try:
                                    corrected_query = correct_text(query)
                                except Exception as e:
                                    corrected_query = query  # Fallback to original if correction fails
                            
                            if results:
                                st.session_state.search_results = {
                                    'query': query,
                                    'corrected_query': corrected_query,
                                    'search_query': search_query,
                                    'tokens': tokens,
                                    'results': results,
                                    'mode': 'tfidf',
                                    'model': model_info["display"]
                                }
                            else:
                                st.session_state.search_results = None
                                st.error("❌ No results found. Try different keywords.")
                    
                    else:
                        # Semantic Search with lazy loading
                        engine = get_semantic_engine(st.session_state.corpus, selected_model)
                        
                        if engine is None:
                            st.error(f"❌ Failed to load {model_info['display']}. Check if index exists.")
                            st.session_state.search_results = None
                        else:
                            # Perform search
                            results = engine.search(search_query, top_k=num_results)
                            tokens = Text2Tokens(search_query)  # For highlighting
                            
                            end_time = time_module.time()
                            st.session_state.last_search_time = end_time - start_time
                            
                            # Get spelling correction suggestion (after search, non-blocking)
                            if use_spelling_correction:
                                try:
                                    corrected_query = correct_text(query)
                                except Exception as e:
                                    corrected_query = query  # Fallback to original if correction fails
                            
                            if results:
                                st.session_state.search_results = {
                                    'query': query,
                                    'corrected_query': corrected_query,
                                    'search_query': search_query,
                                    'tokens': tokens,
                                    'results': results,
                                    'mode': 'semantic',
                                    'model': model_info["display"]
                                }
                            else:
                                st.session_state.search_results = None
                                st.error("❌ No results found. Try different keywords.")
                
                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")
                    st.session_state.search_results = None
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
    

    def highlight_differences(original, corrected):
        """
        Highlight differences between original and corrected query
        Returns HTML string with differences marked
        """
        original_words = original.lower().split()
        corrected_words = corrected.lower().split()
        
        # If lengths are different, just show both
        if len(original_words) != len(corrected_words):
            return f'<span style="color: #666; text-decoration: line-through;">{original}</span> → <span style="color: #667eea; font-weight: bold;">{corrected}</span>'
        
        # Compare word by word
        result_original = []
        result_corrected = []
        
        for orig, corr in zip(original_words, corrected_words):
            if orig != corr:
                result_original.append(f'<span style="color: #dc3545; text-decoration: line-through; background-color: #ffe6e6; padding: 2px 4px; border-radius: 3px;">{orig}</span>')
                result_corrected.append(f'<span style="color: #28a745; background-color: #e6ffe6; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{corr}</span>')
            else:
                result_original.append(orig)
                result_corrected.append(corr)
        
        return ' '.join(result_original), ' '.join(result_corrected)

    # Display results
    if st.session_state.search_results:
        st.markdown("---")
        
        # Show search info
        search_info = st.session_state.search_results
        
        col_info1, col_info2 = st.columns([3, 1])
        with col_info1:
            # Show original query
            st.markdown(f"### 📋 Results for: *\"{search_info['query']}\"*", unsafe_allow_html=True)
            
            # Show spelling suggestion if different (non-intrusive)
            if use_spelling_correction and normalize_text(search_info['query']) != normalize_text(search_info['corrected_query']):
                result_original, result_corrected = highlight_differences(
                    normalize_text(search_info['query']),
                    normalize_text(search_info['corrected_query'])
                )
                st.markdown(f"""
                <div>
                    <span style="color: white;">💡 <strong>Did you mean:</strong> {result_corrected}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col_info2:
            st.metric(
                label="Mode",
                value=search_info.get('model', 'Unknown'),
                delta=f"{st.session_state.last_search_time:.2f}s"
            )
        
        st.markdown(f"**Found {len(search_info['results'])} relevant documents**")
        st.markdown("")
        
        # Display results in batches to avoid memory issues
        for idx, (doc_id, score) in enumerate(search_info['results'].items(), 1):
            title, snippet_text, url = get_in4(
                doc_id,
                st.session_state.id2doc,
                st.session_state.inverted_index,
                search_info['search_query'],  # Use corrected query for highlighting
                max_chars=max_chars,
                window=window
            )
            
            # Create result card
            result_html = f"""
            <div class="result-card">
                <div class="result-header">
                    <div style="flex: 1;">
                        <div class="result-id">#{idx} • Document ID: {doc_id}</div>
                        <div class="result-title">
                            {'<a href="' + url + '" target="_blank">' + title + '</a>' if url else title}
                        </div>
                    </div>
                    <div class="result-score">Score: {score:.4f}</div>
                </div>
                <div class="result-snippet">{snippet_text}</div>
            </div>
            """
            
            st.markdown(result_html, unsafe_allow_html=True)
    
    elif query and search_button:
        st.info("👆 Enter a search query and click the Search button to get started!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔬 Natural Food Corpus Search Engine | Powered by TF-IDF & Neural Embeddings</p>
    <p style="font-size: 0.9rem;">Multi-modal search: Fast lexical matching + Semantic understanding</p>
</div>
""", unsafe_allow_html=True)

# Memory cleanup hint (optional, for extreme cases)
if st.sidebar.button("🗑️ Clear Cache", help="Clear all cached models to free memory"):
    st.cache_resource.clear()
    gc.collect()
    st.success("✅ Cache cleared! Reload page to start fresh.")
    st.experimental_rerun()