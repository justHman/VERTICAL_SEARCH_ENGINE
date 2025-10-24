import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from utils.processor import Text2Tokens
from utils.loader import load_corpus, load_env, load_inverted_index
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