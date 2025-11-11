import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np
import torch
import time
import pickle
import re
import argparse
from tqdm import tqdm
from utils.processor import Text2Tokens, correct_text
from utils.caculator import compute_tfidf
from utils.processor import merge_ranges
from utils.loader import load_corpus, load_inverted_index, load_env
import faiss
from sentence_transformers import SentenceTransformer
ENV = load_env()

class semantic_search:
    def __init__(self, corpus, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name).to(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.index = None
        self.corpus = corpus
        self.documents = []
    
    def build_index(self, documents):
        """Xây dựng vector index"""
        self.documents = documents
        
        # Tạo embeddings
        # print(f"Encoding {len(documents)} documents")
        embeddings = []
        for doc in tqdm(documents, desc="Encoding documents"):
            embedding = self.model.encode(doc, convert_to_tensor=False)
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        
        # Khởi tạo FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (tương đương cosine)
        
        # Chuẩn hóa vectors để cosine similarity = inner product
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))

    def save_index(self, filepath):
        faiss.write_index(self.index, filepath)
        with open(filepath.replace('.index', '_docs.pkl'), 'wb') as f:
            pickle.dump(self.documents, f)

    def load_index(self, filepath):
        self.index = faiss.read_index(filepath)
        with open(filepath.replace('.index', '_docs.pkl'), 'rb') as f:
            self.documents = pickle.load(f)
    
    def search(self, query, top_k=None):
        """Tìm kiếm ngữ nghĩa"""
        if self.index is None:
            raise ValueError("Chưa build index!")
        
        # Embed query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Tìm kiếm
        if top_k:
            similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
        else:
            total_vectors = self.index.ntotal  
            similarities, indices = self.index.search(query_embedding.astype('float32'), total_vectors)
        
        results = {}
        for score, idx in zip(similarities[0], indices[0]):
            doc_id = self.corpus[idx]['_id']
            results[doc_id] = float(score)

        return results

def tfidf_search(tokens, inverted_index, total_docs, n=None, alpha=0.8):
    results = {}
    for token in tokens:
        if token in results:
            continue

        scores = compute_tfidf(token, inverted_index, total_docs, alpha=alpha)
        if not scores:
            continue

        results[token] = scores

    # normalized = {}
    # for term, doc_scores in results.items():
    #     vals = np.array(list(doc_scores.values()))
    #     min_v, max_v = vals.min(), vals.max()
    #     normalized[term] = {d: (v - min_v)/(max_v - min_v + 1e-9) for d, v in doc_scores.items()}

    # for term, doc_scores in normalized.items():
    #         for doc_id, score in doc_scores.items():
    #             if doc_id == "MED-10":
    #                 print(term, doc_id, score)

    combined = {}
    for term in results:
        for doc, val in results[term].items():
            combined[doc] = combined.get(doc, 0) + val
                
    if not combined:
        return {}
    
    vals = np.array(list(combined.values()))
    min_v, max_v = vals.min(), vals.max()
    final_scores = {d: (v - min_v) / (max_v - min_v + 1e-9) for d, v in combined.items()}
    if n:
        sorted_scores = sorted(final_scores.keys(), key=lambda doc_id: final_scores[doc_id], reverse=True)[:n]
    else:
        sorted_scores = sorted(final_scores.keys(), key=lambda doc_id: final_scores[doc_id], reverse=True)
    return {doc_id: final_scores[doc_id] for doc_id in sorted_scores}

def get_in4(doc_id, id2doc, inverted_index, query, max_chars=300, window=11):
    try:
        title = id2doc[doc_id].get('title', '')
        url = id2doc[doc_id].get('metadata', {}).get('url', '')
        text = id2doc[doc_id].get('text', '')
        text_tokens = text.split()
    except KeyError as e:
        print(f"Error getting doc info for {doc_id}: {e}")
        print(f"Available keys in doc: {id2doc[doc_id].keys() if doc_id in id2doc else 'doc_id not found'}")
        raise
    
    position_list = []
    tokens = Text2Tokens(query)
    for term in tokens:
        try:
            # Check if term exists in inverted_index and if doc_id has this term
            if term not in inverted_index:
                continue
            if doc_id not in inverted_index[term]:
                continue
            
            # Check if 'text' field exists in the document's term data
            if 'text' not in inverted_index[term][doc_id]:
                continue
            if 'positions' not in inverted_index[term][doc_id]['text']:
                continue
                
            positions = inverted_index[term][doc_id]['text']['positions']
            for postion in positions:
                start = max(0, postion - window)
                end = min(len(text_tokens), postion + window + 1)
                position_list.append((start, end))
        except (KeyError, TypeError) as e:
            print(f"Warning: Error processing term '{term}' for doc {doc_id}: {e}")
            continue

    merged = merge_ranges(position_list)

    if not merged:
        snippet_text = (text[:max_chars] + '...') if len(text) > max_chars else text
    else:
        parts = []
        for start, end in merged:
            part = " ".join(text_tokens[start:end])
            if start > 0:
                part = "..." + part
            if end < len(text_tokens):
                part = part + "..."
            parts.append(part)
        snippet_text = " ... ".join(parts)

        if len(snippet_text) > max_chars:
            snippet_text = snippet_text[:max_chars].rstrip() + "..."

    sorted_tokens = sorted(set(tokens), key=len, reverse=True)
    for term in sorted_tokens:
        if not term:
            continue

        pattern = r"\b" + re.escape(term) + r"\b"
        snippet_text = re.sub(pattern,
                              lambda m: f"<mark><strong>{m.group(0)}</strong></mark>",
                              snippet_text,
                              flags=re.IGNORECASE)

    return title, snippet_text, url

def main(args):
    corpus = load_corpus(args.corpus_path)
    inverted_index = load_inverted_index(args.inverted_index_path)

    query = args.query
    corrected_query = correct_text(query)

    mode = args.mode if args.mode else input("Choose search mode (semantic/tfidf) [tfidf]: ").strip().lower()
    if mode == "semantic":
        # Determine model name based on index path or use specified model
        model_name = args.model_name
        if not model_name and args.index_path:
            # Infer model from index path
            if "all-MiniLM-L6-v2" in args.index_path:
                model_name = "all-MiniLM-L6-v2"
            elif "multi-qa-mpnet-base-dot-v1" in args.index_path: 
                model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
            elif "pubmedbert" in args.index_path.lower() or "S-PubMedBert" in args.index_path:
                model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
            else:
                model_name = "all-MiniLM-L6-v2"  # Default
        elif not model_name:
            model_name = "all-MiniLM-L6-v2"  # Default
        
        print(f"Using semantic search with model: {model_name}")
        semantic_search_engine = semantic_search(corpus, model_name=model_name)
        
        if args.index_path:
            print(f"Loading index from: {args.index_path}")
            semantic_search_engine.load_index(args.index_path)
        else:
            print("Building index from scratch...")
            documents = [doc.get('text', '') for doc in corpus]
            semantic_search_engine.build_index(documents)

        start = time.time()
        results = semantic_search_engine.search(corrected_query, top_k=args.top_k)
        end = time.time()

    else:
        total_docs = len(corpus)
        
        start = time.time()
        tokens = Text2Tokens(corrected_query)
        results = tfidf_search(tokens, inverted_index, total_docs, n=args.top_k)
        end = time.time()

    id2doc = {doc['_id']: doc for doc in corpus}
    for doc_id, score in results.items():
        title, snippet_text, url = get_in4(doc_id, id2doc, inverted_index, corrected_query)

        print(doc_id, score)
        print(f'<a href="{url}">{title}</a>' if url else title)
        print(snippet_text)
        print("-" * 80)

    print(f"Search completed in {end - start:.4f} seconds.")
    print(f"Model used: {model_name if mode == 'semantic' else 'TF-IDF'}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search engine script with dual search modes (TF-IDF and Semantic).",
        epilog="""
Examples:
  # TF-IDF search
  python src/searcher.py --mode tfidf --query "statin effects on cholesterol"
  
  # Semantic search with all-MiniLM-L6-v2
  python src/searcher.py --mode semantic --query "heart disease prevention" --index_path data/semantic/all-MiniLM-L6-v2/index.index
  
  # Semantic search with PubMedBERT (auto-detect model from path)
  python src/searcher.py --mode semantic --query "vitamin D deficiency" --index_path data/semantic/S-PubMedBert-MS-MARCO/index.index
  
  # Semantic search with explicit model specification
  python src/searcher.py --mode semantic --query "omega-3 benefits" --index_path data/semantic/S-PubMedBert-MS-MARCO/index.index --model_name pritamdeka/S-PubMedBert-MS-MARCO
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["semantic", "tfidf"],
        default="tfidf",
        help="Search mode: 'semantic' (neural embeddings) or 'tfidf' (keyword matching). Default: tfidf"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="""Semantic model name (only for semantic mode). Options:
        - all-MiniLM-L6-v2 (fast, general)
        - sentence-transformers/multi-qa-mpnet-base-dot-v1 (Q&A optimized)
        - pritamdeka/S-PubMedBert-MS-MARCO (biomedical domain)
        If not specified, auto-detects from --index_path or uses default."""
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data\\nfcorpus\\corpus.jsonl",
        help="Path to the corpus file. Default: data/nfcorpus/corpus.jsonl"
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default=None,
        help="""Path to the semantic search index file (required for semantic mode).
        Examples:
        - data/semantic/all-MiniLM-L6-v2/index.index
        - data/semantic/S-PubMedBert-MS-MARCO/index.index
        - data/semantic/multi-qa-mpnet-base-dot-v1/index.index"""
    )
    parser.add_argument(
        "--inverted_index_path",
        type=str,
        default="data\\inverted_index.json",
        help="Path to the inverted index file for TF-IDF. Default: data/inverted_index.json"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="statin effects on cholesterol",
        help="Search query. Default: 'statin effects on cholesterol'"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return. Default: 5"
    )

    args = parser.parse_args()

    main(args)