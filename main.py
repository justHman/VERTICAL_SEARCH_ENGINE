from utils.loader import load_corpus, load_inverted_index, load_env
from src.searcher import search, get_in4
from utils.processor import Text2Tokens, correct_text
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
ENV = load_env()


if __name__ == "__main__":
    model_path = ENV["MODEL_PATH"]
    if not os.path.exists(model_path):
        model_path = "oliverguhr/spelling-correction-english-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path) 

    corpus = load_corpus(ENV["CORPUS_PATH"])
    inverted_index = load_inverted_index(ENV["INVERTED_INDEX_PATH"])
    total_docs = len(corpus)

    query = input("Enter your search query: ")
    corrected_query = correct_text(query)
    tokens = Text2Tokens(corrected_query)

    results = search(tokens, inverted_index, total_docs, n=10)

    max_chars = 300
    window = 11
    id2doc = {doc['_id']: doc for doc in corpus}
    for doc_id, score in results.items():
        title, snippet_text, url = get_in4(doc_id, id2doc, inverted_index, tokens)

        print(doc_id, score)
        print(f'<a href="{url}">{title}</a>' if url else title)
        print(snippet_text)