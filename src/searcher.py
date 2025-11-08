import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np
import re
from utils.processor import Text2Tokens, correct_text
from utils.caculator import compute_tfidf
from utils.processor import merge_ranges
from utils.loader import load_corpus, load_inverted_index, load_env
ENV = load_env()

def search(tokens, inverted_index, total_docs, n=None, alpha=0.8):
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

def get_in4(doc_id, id2doc, inverted_index, tokens, max_chars=300, window=11):
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

def main():
    corpus = load_corpus(ENV["CORPUS_PATH"])
    inverted_index = load_inverted_index(ENV["INVERTED_INDEX_PATH"])
    total_docs = len(corpus)

    query = input("Enter your search query: ")
    if not query:
        query = "statin effects on cholesterol"
    corrected_query = correct_text(query)
    tokens = Text2Tokens(corrected_query)

    results = search(tokens, inverted_index, total_docs, n=10)

    id2doc = {doc['_id']: doc for doc in corpus}
    for doc_id, score in results.items():
        title, snippet_text, url = get_in4(doc_id, id2doc, inverted_index, tokens)

        print(doc_id, score)
        print(f'<a href="{url}">{title}</a>' if url else title)
        print(snippet_text)
        print("-" * 80)

if __name__ == "__main__":
    main()