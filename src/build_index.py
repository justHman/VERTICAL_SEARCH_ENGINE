import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from searcher import semantic_search
import argparse
from utils.loader import load_corpus

def main(args):
    corpus = load_corpus(args.corpus_path)
    documents = [doc.get('text', '') for doc in corpus]

    semantic_search_engine = semantic_search(corpus, model_name=args.model_name)
    print(f"Building index using model: {args.model_name}...")
    semantic_search_engine.build_index(documents)
    print("Index built successfully.")

    # Ensure index_path ends with .index
    index_path = args.index_path
    if not index_path.endswith('.index'):
        index_path += '.index'

    semantic_search_engine.save_index(index_path)
    print(f"Index saved to {index_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build search index from corpus.")
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data\\nfcorpus\\corpus.jsonl",
        help="Path to the corpus file. Defaults to the path in ENV.json."
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="data\semantic\multi-qa-mpnet-base-dot-v1/index",
        help="Path to save the generated index."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        help="Name of the sentence transformer model to use."
    )

    args = parser.parse_args()

    main(args)

    # Example CLI usage:
    # python src/build_index.py --corpus_path data/nfcorpus/corpus.jsonl --index_path data\semantic\multi-qa-mpnet-base-dot-v1/index.index --model_name sentence-transformers/multi-qa-mpnet-base-dot-v1
