import math

def compute_tfidf(term, inverted_index, total_docs, alpha=0.7):
    if term not in inverted_index:
        return {}

    title_df = 0
    text_df = 0
    for doc_id, sections in inverted_index[term].items():
        if "title" in sections:
            title_df += 1
        if "text" in sections:
            text_df += 1

    title_idf = math.log((total_docs + 1e-3) / (title_df + 1e-3)) + 1e-3  
    text_idf = math.log((total_docs + 1e-3) / (text_df + 1e-3)) + 1e-3

    tfidf_scores = {}
    for doc_id, sections in inverted_index[term].items():
        title_tf = sections.get("title", {}).get("count", 0)
        text_tf = sections.get("text", {}).get("count", 0)

        title_tfidf = title_tf*title_idf
        text_tfidf = text_tf*text_idf

        tfidf_scores[doc_id] = alpha*title_tfidf + (1 - alpha)*text_tfidf

    return tfidf_scores

if __name__ == "__main__":
    
    inverted_index = {
        "statin": {
            "MED-10": {
                "text": {"count": 11, "positions": [3, 26, 53, 67, 69, 78, 111, 142, 155, 174, 184]},
            },
            "MED-14": {
                "title": {"count": 1, "positions": [0]},
                "text": {"count": 8, "positions": [5, 34, 56, 78, 99, 123, 145, 167]},
            },
            "MED-20": {
                "title": {"count": 5, "positions": [15, 45, 78, 102, 130]},
                "text": {"count": 4, "positions": [12, 34, 56, 78]},
            },
        }
    }
    # Tổng số tài liệu trong corpus (giả sử có 20 tài liệu)
    total_docs = 20

    tokens = ["statin"]
    for token in tokens:
        print(f"TF-IDF scores for term '{token}':")
        scores = compute_tfidf(token, inverted_index, total_docs, alpha=0.7)
        if not scores:
            print("  Term not found in any document.")
            break
        for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{doc}: TF-IDF = {score:.4f}")

