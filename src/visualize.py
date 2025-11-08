import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from utils.loader import load_env, load_inverted_index
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
ENV = load_env()

def visualize_wordcloud(data, title, stopwords=None, freq=False, save_path='results/wordcloud_corpus.png'):
    if freq:
        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            max_words=200,
            colormap="viridis",
            stopwords=stopwords
        ).generate_from_frequencies(data)
    else:
        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            max_words=200,
            colormap="viridis",
            stopwords=stopwords
        ).generate(data)

    plt.figure(figsize=(15, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.show()

    if save_path:
        wc.to_file(save_path)

# def visualize_evaluate(scores, n_queries, save_path=None):
#     mprec = scores.get("mPrecision@k", {})
#     ks = sorted(mprec.keys())
#     vals = [mprec[k] for k in ks]
#     mAP = scores.get("mAP", 0.0)

#     if not ks:
#         print("No data to plot (ks empty).")
#     else:
#         fig = plt.figure(figsize=(12,6))
#         ax = fig.add_subplot(1,1,1)

#         ax.plot(ks, vals, marker='o', linewidth=2)
#         ax.set_xlabel('k')
#         ax.set_ylabel('mPrecision@k')
#         ax.set_title(f"mPrecision@k (n_queries={n_queries}) — mAP={mAP:.4f}")
#         ax.grid(alpha=0.3)
#         ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.4f}"))

#         step = max(1, (max(ks) - min(ks)) // 20)
#         ax.set_xticks(np.arange(min(ks), max(ks)+1, step))

#         max_idx = int(np.argmax(vals))
#         max_k = ks[max_idx]
#         max_v = vals[max_idx]
#         ax.scatter([max_k], [max_v], color='red', zorder=5)
#         ax.annotate(f"{max_v:.4f} @k={max_k}", (max_k, max_v),
#                     xytext=(max_k, max_v + 0.01),
#                     arrowprops=dict(arrowstyle="->", color="red"))

#         ax_bar = fig.add_axes([0.75, 0.58, 0.18, 0.28])   
#         ax_bar.bar(['mAP'], [mAP], color='C1')
#         ax_bar.set_ylim(0, 1)
#         ax_bar.set_ylabel('')
#         ax_bar.set_title('mAP', fontsize=10)
#         ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.4f}"))
#         ax_bar.text(0, mAP + 0.01, f"{mAP:.4f}", ha='center', va='bottom', fontsize=9)

#         plt.tight_layout()
#         os.makedirs('results', exist_ok=True)
#         if save_path:
#             plt.savefig(save_path, dpi=200, bbox_inches='tight')
#         plt.show()

def visualize_evaluate(scores, n_queries, save_path=None):
    """
    Visualize mPrecision@k and highlight Precision@10 with its value on the plot.

    Args:
        scores: A dictionary containing mPrecision@k and mAP.
        n_queries: Number of queries used in the evaluation.
        save_path: Path to save the plot (optional).
    """
    mprec = scores.get("mPrecision@k", {})
    ks = sorted(mprec.keys())
    vals = [mprec[k] for k in ks]
    mAP = scores.get("mAP", 0.0)

    if not ks:
        print("No data to plot (ks empty).")
    else:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(ks, vals, marker='o', linewidth=2, label="mPrecision@k")
        ax.set_xlabel('k')
        ax.set_ylabel('mPrecision@k')
        ax.set_title(f"mPrecision@k (n_queries={n_queries}) — mAP={mAP:.4f}")
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.4f}"))

        step = max(1, (max(ks) - min(ks)) // 20)
        ax.set_xticks(np.arange(min(ks), max(ks) + 1, step))

        # Highlight Precision@10
        if 10 in ks:
            precision_at_10 = mprec[10]
            ax.scatter([10], [precision_at_10], color='orange', zorder=5, label=f"Precision@10 ({precision_at_10:.4f})")
            ax.annotate(f"{precision_at_10:.4f} @k=10",
                        (10, precision_at_10),
                        xytext=(10, precision_at_10 + 0.01),
                        arrowprops=dict(arrowstyle="->", color="orange"))

        # Highlight the maximum mPrecision@k
        max_idx = int(np.argmax(vals))
        max_k = ks[max_idx]
        max_v = vals[max_idx]
        ax.scatter([max_k], [max_v], color='red', zorder=5, label=f"Max mPrecision@k ({max_v:.4f})")
        ax.annotate(f"{max_v:.4f} @k={max_k}",
                    (max_k, max_v),
                    xytext=(max_k, max_v + 0.01),
                    arrowprops=dict(arrowstyle="->", color="red"))

        # Add a small bar chart for mAP
        ax_bar = fig.add_axes([0.75, 0.58, 0.18, 0.28])
        ax_bar.bar(['mAP'], [mAP], color='C1')
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel('')
        ax_bar.set_title('mAP', fontsize=10)
        ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.4f}"))
        ax_bar.text(0, mAP + 0.01, f"{mAP:.4f}", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.legend()
        plt.show()

def visualize_alpha_scores(alpha_scores, save_path=None):
    """
    Visualize mAP and Precision@10 for different alpha values, highlighting the maximum values.

    Args:
        alpha_scores: A dictionary of scores for each alpha.
        save_path: Path to save the plot (optional).
    """
    alphas = sorted(alpha_scores.keys())
    mAPs = [alpha_scores[alpha]["mAP"] for alpha in alphas]
    P10s = [alpha_scores[alpha]["mPrecision@k"].get(10, 0.0) for alpha in alphas]

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, mAPs, marker='o', label='mAP', color='blue')
    plt.plot(alphas, P10s, marker='o', label='Precision@10', color='green')

    # Highlight the maximum mAP
    max_mAP_idx = int(np.argmax(mAPs))
    max_mAP_alpha = alphas[max_mAP_idx]
    max_mAP_value = mAPs[max_mAP_idx]
    plt.scatter([max_mAP_alpha], [max_mAP_value], color='red', zorder=5, label=f"Max mAP ({max_mAP_value:.4f})")
    plt.annotate(f"{max_mAP_value:.4f} @alpha={max_mAP_alpha}",
                 (max_mAP_alpha, max_mAP_value),
                 xytext=(max_mAP_alpha, max_mAP_value + 0.01),
                 arrowprops=dict(arrowstyle="->", color="red"))

    # Highlight the maximum Precision@10
    max_P10_idx = int(np.argmax(P10s))
    max_P10_alpha = alphas[max_P10_idx]
    max_P10_value = P10s[max_P10_idx]
    plt.scatter([max_P10_alpha], [max_P10_value], color='orange', zorder=5, label=f"Max Precision@10 ({max_P10_value:.4f})")
    plt.annotate(f"{max_P10_value:.4f} @alpha={max_P10_alpha}",
                 (max_P10_alpha, max_P10_value),
                 xytext=(max_P10_alpha, max_P10_value + 0.01),
                 arrowprops=dict(arrowstyle="->", color="orange"))

    plt.xlabel("Alpha")
    plt.ylabel("Score")
    plt.title("mAP and Precision@10 vs Alpha")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # contents = {}
    # corpus = load_corpus(ENV['CORPUS_PATH'])
    # for doc in corpus:
    #     if "title" not in contents:
    #         contents["title"] = []
    #     contents["title"].append(doc.get("title", ""))

    #     if "text" not in contents:
    #         contents["text"] = []
    #     contents["text"].append(doc.get("text", ""))
    
    # full_title = " ".join([token for title in contents["title"] for token in Text2Tokens(title)])
    # visualize_wordcloud(full_title, "TITLE WORDCLOUD", save_path='results/title_wordcloud.png')

    # full_text = " ".join([token for text in contents["text"] for token in Text2Tokens(text)])
    # visualize_wordcloud(full_text, "TEXT WORDCLOUD", save_path='results/text_wordcloud.png')

    # combined = full_title + " " + full_text
    # visualize_wordcloud(combined, "COMBINED WORDCLOUD", save_path='results/combined_wordcloud.png')

    inverted_index = load_inverted_index(ENV['INVERTED_INDEX_PATH'])
    word_freq = {}
    for term, docs in inverted_index.items():
        for doc_id, contents in docs.items():
            if term not in word_freq:
                word_freq[term] = 0

            if "title" in contents:
                word_freq[term] += inverted_index[term][doc_id]["title"]["count"]

            if "text" in contents:
                word_freq[term] += inverted_index[term][doc_id]["text"]["count"]
    stopwords = set(STOPWORDS)
    stopwords.update(list(map(str, range(9999))))  # thêm tùy ý

    visualize_wordcloud(word_freq, "NFCORPUS WORDCLOUD", stopwords, freq=True, save_path='results/nfcorpus_wordcloud.png')