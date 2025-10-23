import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import pandas as pd
from searcher import search
from utils.loader import load_corpus, load_inverted_index, load_env, load_queries
from utils.processor import Text2Tokens
from utils.caculator import Precision_at_k, AP
ENV = load_env()


def evaluate(df, inverted_index, total_docs, queries, n_queries):
    querie_ids = df['query_id'].unique()[:n_queries]
    scores = {}

    for k in tqdm(range(1, 20), desc='k loop'):
        P_at_k = []
        AP_scores = [] if k == 1 else None 
        for query_id in tqdm(querie_ids, desc=f'queries@k={k}', leave=False):
            relevant_docs = df[df["query_id"] == query_id]["corpus_id"].tolist()
            relevant_docs = list(set(relevant_docs))

            query = queries[query_id]
            tokens = Text2Tokens(query)
            results = search(tokens, inverted_index, total_docs)
            retrieved_docs = results.keys()
            P_at_k.append(Precision_at_k(relevant_docs, retrieved_docs, k))

            if AP_scores is not None:
                AP_scores.append(AP(relevant_docs, retrieved_docs))
        
        mP_at_k = sum(P_at_k) / n_queries
        scores.setdefault("mPrecision@k", {})[k] = mP_at_k

        if AP_scores is not None:
            mAP = sum(AP_scores) / n_queries

    scores["mAP"] = mAP
    return scores

def visualize(scores, n_queries, save_path='results/mPrecision_at_k_and_mAP.png'):
    mprec = scores.get("mPrecision@k", {})
    ks = sorted(mprec.keys())
    vals = [mprec[k] for k in ks]
    mAP = scores.get("mAP", 0.0)

    if not ks:
        print("No data to plot (ks empty).")
    else:
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1,1,1)

        ax.plot(ks, vals, marker='o', linewidth=2)
        ax.set_xlabel('k')
        ax.set_ylabel('mPrecision@k')
        ax.set_title(f"mPrecision@k (n_queries={n_queries}) — mAP={mAP:.4f}")
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.4f}"))

        step = max(1, (max(ks) - min(ks)) // 20)
        ax.set_xticks(np.arange(min(ks), max(ks)+1, step))

        max_idx = int(np.argmax(vals))
        max_k = ks[max_idx]
        max_v = vals[max_idx]
        ax.scatter([max_k], [max_v], color='red', zorder=5)
        ax.annotate(f"{max_v:.4f} @k={max_k}", (max_k, max_v),
                    xytext=(max_k, max_v + 0.01),
                    arrowprops=dict(arrowstyle="->", color="red"))

        ax_bar = fig.add_axes([0.75, 0.58, 0.18, 0.28])   
        ax_bar.bar(['mAP'], [mAP], color='C1')
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel('')
        ax_bar.set_title('mAP', fontsize=10)
        ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.4f}"))
        ax_bar.text(0, mAP + 0.01, f"{mAP:.4f}", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()

def main():
    df = pd.read_csv(ENV["QRELS_PATH"])
    corpus = load_corpus(ENV["CORPUS_PATH"])
    queries = load_queries(ENV["QUERIES_PATH"])
    inverted_index = load_inverted_index(ENV["INVERTED_INDEX_PATH"])
    total_docs = len(corpus)

    # Check for environment variable N_QUERIES from batch script
    n_queries_env = os.environ.get("N_QUERIES")
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

    scores = evaluate(df, inverted_index, total_docs, queries, n_queries)
    print(f"Evaluation completed with {n_queries} queries.")
    visualize(scores, n_queries, save_path=ENV["EVALUATION_RESULT_PATH"])

if __name__ == "__main__":
    main()