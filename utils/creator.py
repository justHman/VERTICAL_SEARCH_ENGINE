import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from utils.processor import Text2Tokens
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def inverted_index_builder(corpus, save_path=None):
    inverted_index = {}
    for doc in tqdm(corpus, desc="Building inverted index"):
        doc_id = doc.get('_id', None)
        if doc_id is None:
            continue

        title = doc.get('title', '').strip()
        tokens = Text2Tokens(title)
        for idx, token in enumerate(tokens):
            if token not in inverted_index:
                inverted_index[token] = {}
            if doc_id not in inverted_index[token]:
                inverted_index[token][doc_id] = {}
            if "title" not in inverted_index[token][doc_id]:
                inverted_index[token][doc_id]["title"] = {'count': 0, 'positions': []}
            inverted_index[token][doc_id]["title"]['count'] += 1
            inverted_index[token][doc_id]["title"]['positions'].append(idx)

        text = doc.get('text', '').strip()
        tokens = Text2Tokens(text)
        for idx, token in enumerate(tokens):
            if token not in inverted_index:
                inverted_index[token] = {}
            if doc_id not in inverted_index[token]:
                inverted_index[token][doc_id] = {}
            if "text" not in inverted_index[token][doc_id]:
                inverted_index[token][doc_id]["text"] = {'count': 0, 'positions': []}
            inverted_index[token][doc_id]["text"]['count'] += 1
            inverted_index[token][doc_id]["text"]['positions'].append(idx)

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(inverted_index, f, ensure_ascii=False, indent=2)
    return inverted_index


def fast_inverted_index_builder(corpus, save_path=None, max_workers=None):
    inverted_index = {}

    # Materialize corpus to list to know total length for tqdm
    docs = list(corpus)

    def _process_doc(doc):
        doc_id = doc.get('_id', None)
        if doc_id is None:
            return None
        per_doc_index = {}
        title = doc.get('title', '').strip()
        tokens = Text2Tokens(title)
        for idx, token in enumerate(tokens):
            per_doc_index.setdefault(token, {}).setdefault(doc_id, {}).setdefault("title", {'count':0,'positions':[]})
            per_doc_index[token][doc_id]["title"]['count'] += 1
            per_doc_index[token][doc_id]["title"]['positions'].append(idx)

        text = doc.get('text', '').strip()
        tokens = Text2Tokens(text)
        for idx, token in enumerate(tokens):
            per_doc_index.setdefault(token, {}).setdefault(doc_id, {}).setdefault("text", {'count':0,'positions':[]})
            per_doc_index[token][doc_id]["text"]['count'] += 1
            per_doc_index[token][doc_id]["text"]['positions'].append(idx)

        return per_doc_index

    workers = max_workers or os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_doc, doc): doc for doc in docs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building inverted index"):
            per_doc_idx = future.result()
            if not per_doc_idx:
                continue
            # merge per_doc_idx vào inverted_index (chạy trong main thread)
            for token, doc_map in per_doc_idx.items():
                if token not in inverted_index:
                    inverted_index[token] = {}
                for doc_id, fields in doc_map.items():
                    if doc_id not in inverted_index[token]:
                        inverted_index[token][doc_id] = {}
                    for field, info in fields.items():
                        if field not in inverted_index[token][doc_id]:
                            inverted_index[token][doc_id][field] = {'count': 0, 'positions': []}
                        inverted_index[token][doc_id][field]['count'] += info['count']
                        inverted_index[token][doc_id][field]['positions'].extend(info['positions'])

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(inverted_index, f, ensure_ascii=False, indent=2)
    return inverted_index
