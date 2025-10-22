import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np
from utils.processor import Text2Tokens
from utils.caculator import compute_tfidf
from utils.processor import merge_ranges
from utils.loader import load_corpus, load_inverted_index, load_env
ENV = load_env()

def search(tokens, inverted_index, total_docs, n=10, alpha=0.7):
    results = {}
    for token in tokens:
        scores = compute_tfidf(token, inverted_index, total_docs, alpha=alpha)
        if not scores:
            continue
        results[token] = scores

    normalized = {}
    for term, doc_scores in results.items():
        vals = np.array(list(doc_scores.values()))
        min_v, max_v = vals.min(), vals.max()
        normalized[term] = {d: (v - min_v)/(max_v - min_v + 1e-9) for d, v in doc_scores.items()}

    combined = {}
    for term in normalized:
        for doc, val in normalized[term].items():
            combined[doc] = combined.get(doc, 0) + val

    vals = np.array(list(combined.values()))
    min_v, max_v = vals.min(), vals.max()
    final_scores = {d: (v - min_v) / (max_v - min_v + 1e-9) for d, v in combined.items()}
    sorted_scores = sorted(final_scores.keys(), key=lambda doc_id: final_scores[doc_id], reverse=True)[:n]
    return {doc_id: final_scores[doc_id] for doc_id in sorted_scores}

def get_in4(doc_id, id2doc, inverted_index, tokens, max_chars=300, window=11):
    position_list = []
    for term in tokens:
        if doc_id not in inverted_index[term]:
            continue
        
        title = id2doc[doc_id]['title']
        url = id2doc[doc_id]['metadata']['url']
        text = id2doc[doc_id]['text']
        text_tokens = text.split()
        positions = inverted_index[term][doc_id]['text']['positions']
        for postion in positions:
            start = max(0, postion - window)
            end = min(len(text_tokens), postion + window + 1)
            position_list.append((start, end))

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

    for term in tokens:
        snippet_text = snippet_text.replace(term, f"<b>{term}</b>")

    return title, snippet_text, url

def main():
    corpus = load_corpus(ENV["CORPUS_PATH"])
    inverted_index = load_inverted_index(ENV["INVERTED_INDEX_PATH"])
    total_docs = len(corpus)

    query = "statin effects on cholesterol"
    tokens = Text2Tokens(query)

    results = search(tokens, inverted_index, total_docs, n=10)

    id2doc = {doc['_id']: doc for doc in corpus}
    for doc_id, score in results.items():
        title, snippet_text, url = get_in4(doc_id, id2doc, inverted_index, tokens)

        print(doc_id, score)
        print(f'<a href="{url}">{title}</a>' if url else title)
        print(snippet_text)

if __name__ == "__main__":
    main()