import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from utils.loader import load_corpus
from utils.creator import inverted_index_builder, fast_inverted_index_builder

def main():
    corpus_path = r'data\nfcorpus\corpus.jsonl'
    save_path = 'inverted_index.json'

    corpus = load_corpus(corpus_path)
    inverted_index_builder(corpus, save_path=save_path)
    # fast_inverted_index_builder(corpus, save_path=save_path, max_workers=8)

if __name__ == "__main__":
    main()