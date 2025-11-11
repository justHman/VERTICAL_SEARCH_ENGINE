import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from src.searcher import semantic_search
from src.visualize import visualize_evaluate, visualize_alpha_scores
from tqdm import tqdm
import pandas as pd
from searcher import tfidf_search
from utils.loader import load_corpus, load_inverted_index, load_env, load_queries
from utils.processor import Text2Tokens
from utils.caculator import Precision_at_k, AP
ENV = load_env()


def evaluate_tfidf_search(df, inverted_index, corpus, queries, n_queries, range_k=range(1,20), alpha=0.8):
    scores = {}

    mAP_check = False
    for k in tqdm(range_k, desc='k loop'):
        P_at_k = []
        AP_scores = None if mAP_check else [] 
        total = 0
        for query_id, query in tqdm(list(queries.items())[:n_queries], desc=f'queries@k={k}', leave=False):
            relevant_docs = df[df["query_id"] == query_id]["corpus_id"].tolist()
            if not relevant_docs:
                continue
            # relevant_docs = list(set(relevant_docs))

            tokens = Text2Tokens(query)
            results = tfidf_search(tokens, inverted_index, len(corpus), alpha=alpha)
            retrieved_docs = results.keys()
            P_at_k.append(Precision_at_k(relevant_docs, retrieved_docs, k))

            if AP_scores is not None:
                AP_scores.append(AP(relevant_docs, retrieved_docs))

            total += 1

        mP_at_k = sum(P_at_k) / total
        scores.setdefault("mPrecision@k", {})[k] = mP_at_k

        if AP_scores is not None:
            mAP = sum(AP_scores) / total
            mAP_check = True

    scores["mAP"] = mAP
    return scores 

def evaluate_semantic_search(df, index_path, corpus, queries, n_queries, model_name, range_k=range(1,20)): 
    semantic_search_engine = semantic_search(corpus, model_name=model_name)
    if index_path:
        semantic_search_engine.load_index(index_path)

    scores = {}

    mAP_check = False
    for k in tqdm(range_k, desc='k loop'):
        P_at_k = []
        AP_scores = None if mAP_check else [] 
        total = 0
        for query_id, query in tqdm(list(queries.items())[:n_queries], desc=f'queries@k={k}', leave=False):
            relevant_docs = df[df["query_id"] == query_id]["corpus_id"].tolist()
            if not relevant_docs:
                continue
            # relevant_docs = list(set(relevant_docs))

            # tokens = Text2Tokens(query)
            # results = tfidf_search(tokens, inverted_index, total_docs, alpha=alpha)
            results = semantic_search_engine.search(query)
            retrieved_docs = results.keys()

            P_at_k.append(Precision_at_k(relevant_docs, retrieved_docs, k))

            if AP_scores is not None:
                AP_scores.append(AP(relevant_docs, retrieved_docs))
            total += 1
        
        mP_at_k = sum(P_at_k) / total
        scores.setdefault("mPrecision@k", {})[k] = mP_at_k

        if AP_scores is not None:
            mAP = sum(AP_scores) / total
            mAP_check = True

    scores["mAP"] = mAP
    return scores 
 
def find_best_alpha(df, inverted_index, corpus, queries, n_queries, alphas, range_k=range(10, 11)):
    alpha_scores = {}

    for alpha in alphas:
        print(f"Evaluating alpha={alpha}...")
        scores = evaluate_tfidf_search(df, inverted_index, corpus, queries, n_queries, range_k=range_k, alpha=alpha)
        precision_at_10 = scores['mPrecision@k'].get(10, 0.0)  # Lấy Precision@10, mặc định là 0.0 nếu không có
        print(f"Precision@10={precision_at_10:.4f}, mAP={scores['mAP']:.4f}")
        alpha_scores[alpha] = scores

    # Find the alpha with the highest mAP
    best_alpha_mAP = max(alpha_scores, key=lambda a: alpha_scores[a]["mAP"])
    best_alpha_precision = max(alpha_scores, key=lambda a: alpha_scores[a]["mPrecision@k"].get(10, 0.0))
    precision_at_10 = alpha_scores[best_alpha_precision]['mPrecision@k'].get(10, 0.0)
    print(f"Best alpha by Precision@10: {best_alpha_precision} with Precision@10={precision_at_10:.4f}")
    print(f"Best alpha by mAP: {best_alpha_mAP} with mAP={alpha_scores[best_alpha_mAP]['mAP']:.4f}")
    
    return best_alpha_mAP, best_alpha_precision, alpha_scores

def main():
    df = pd.read_csv(ENV["QRELS_PATH"])
    corpus = load_corpus(ENV["CORPUS_PATH"])
    queries = load_queries(ENV["QUERIES_PATH"])
    # inverted_index = load_inverted_index(ENV["INVERTED_INDEX_PATH"])

    # Check for environment variable N_QUERIES from batch script
    n_queries_env = ENV["N_QUERIES"]
    if n_queries_env:
        try:
            n_queries = int(n_queries_env)
            n_queries = min(n_queries, len(df['query_id'].unique()))  # Don't exceed available queries
            print(f"Using {n_queries} queries for evaluation (from environment variable)")
        except ValueError:
            print(f"Invalid N_QUERIES value: {n_queries_env}. Using default.")
            n_queries = len(df['query_id'].unique())
    else:
        n_queries = len(df['query_id'].unique())


    # alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Các giá trị alpha cần thử nghiệm

    # # Tìm alpha tốt nhất
    # best_alpha_mAP, best_alpha_precision, alpha_scores = find_best_alpha(df, inverted_index, corpus, queries, n_queries, alphas)

    # # Trực quan hóa kết quả
    # visualize_alpha_scores(alpha_scores, save_path="results/images/alpha_scores_2.png")

    # tfidf_scores = evaluate_tfidf_search(df, inverted_index, corpus, queries, n_queries, range_k=range(1,20), alpha=best_alpha_precision)
    # print(f"Evaluation completed with {n_queries} queries.")
    # visualize_evaluate(tfidf_scores, n_queries, save_path='results/images/tfidf_evaluation.png')

    # model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    index_path = "data\semantic\multi-qa-mpnet-base-dot-v1\index.index"
    semantic_scores = evaluate_semantic_search(df, index_path, corpus, queries, n_queries, model_name, range_k=range(1,20))
    print(f"Evaluation completed with {n_queries} queries.")
    visualize_evaluate(semantic_scores, n_queries, save_path='results/images/multi-qa-mpnet-base-dot-v1.png')

if __name__ == "__main__":
    main()