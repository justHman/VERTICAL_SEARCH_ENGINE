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

    title_idf = math.log(((total_docs + 1e-3) / (title_df + 1e-3)) + 1)   
    text_idf = math.log(((total_docs + 1e-3) / (text_df + 1e-3)) + 1) 
    tfidf_scores = {}
    for doc_id, sections in inverted_index[term].items():
        title_tf = sections.get("title", {}).get("count", 0)
        text_tf = sections.get("text", {}).get("count", 0)

        title_tfidf = title_tf*title_idf
        text_tfidf = text_tf*text_idf

        tfidf_scores[doc_id] = alpha*title_tfidf + (1 - alpha)*text_tfidf

    return tfidf_scores

def Precision_at_k(relevant_docs, retrieved_docs, k):
    if k <= 0:
        return 0.0
    
    k = min(k, len(retrieved_docs))   # giới hạn theo số kết quả thực tế
    if k == 0:
        return 0.0
        
    retrieved_docs_at_k = list(retrieved_docs)[:k]
    relevant_set = set(relevant_docs)
    matched_docs = sum(1 for doc in retrieved_docs_at_k if doc in relevant_set)
    return matched_docs / k

def AP(relevant_docs, retrieved_docs):
    if not relevant_docs:
        return 0.0
    
    relevant_set = set(relevant_docs)
    total_p = 0.0
    relevant_retrieved = 0
    
    for k, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_set:
            relevant_retrieved += 1
            precision_at_k = relevant_retrieved / k
            total_p += precision_at_k
    
    return total_p / relevant_retrieved if relevant_retrieved > 0 else 0.0

def test_Precision_at_k():
    # Test case 1: Kết quả truy xuất có tài liệu liên quan
    relevant_docs = ["doc1", "doc2", "doc3"]
    retrieved_docs = ["doc1", "doc4", "doc2", "doc5"]
    k = 3
    precision = Precision_at_k(relevant_docs, retrieved_docs, k)
    print(f"Test case 1 - Precision@{k}: {precision:.4f}")
    assert precision == 2 / 3  # Có 2 tài liệu liên quan trong 3 tài liệu đầu tiên

    # Test case 2: Không có tài liệu liên quan trong kết quả truy xuất
    relevant_docs = ["doc6", "doc7"]
    retrieved_docs = ["doc1", "doc2", "doc3"]
    k = 3
    precision = Precision_at_k(relevant_docs, retrieved_docs, k)
    print(f"Test case 2 - Precision@{k}: {precision:.4f}")
    assert precision == 0.0  # Không có tài liệu nào liên quan

    # Test case 3: K nhỏ hơn số lượng tài liệu truy xuất
    relevant_docs = ["doc1", "doc2"]
    retrieved_docs = ["doc1", "doc2", "doc3", "doc4"]
    k = 2
    precision = Precision_at_k(relevant_docs, retrieved_docs, k)
    print(f"Test case 3 - Precision@{k}: {precision:.4f}")
    assert precision == 1.0  # Cả 2 tài liệu đầu tiên đều liên quan

    # Test case 4: K lớn hơn số lượng tài liệu truy xuất
    relevant_docs = ["doc1", "doc2", "doc3"]
    retrieved_docs = ["doc1", "doc2"]
    k = 5
    precision = Precision_at_k(relevant_docs, retrieved_docs, k)
    print(f"Test case 4 - Precision@{k}: {precision:.4f}")
    assert precision == 1.0  # Tất cả tài liệu truy xuất đều liên quan

    # Test case 5: K = 0
    relevant_docs = ["doc1", "doc2"]
    retrieved_docs = ["doc1", "doc2", "doc3"]
    k = 0
    precision = Precision_at_k(relevant_docs, retrieved_docs, k)
    print(f"Test case 5 - Precision@{k}: {precision:.4f}")
    assert precision == 0.0  # K = 0 nên Precision@k = 0

def test_ap():
    test_cases = [
        # (relevant_docs, retrieved_docs, expected_ap, description)
        (["doc1", "doc2", "doc3"], ["doc1", "doc2", "doc3"], 1.0, "Perfect retrieval"),
        (["doc1", "doc2", "doc3"], ["doc3", "doc2", "doc1"], 1.0, "Perfect but different order"),
        (["doc1", "doc2", "doc3"], ["doc4", "doc5", "doc6"], 0.0, "No relevant retrieved"),
        (["doc1", "doc2", "doc3"], ["doc1", "doc4", "doc5"], 1.0, "Only 1 relevant retrieved"),  # ĐÚNG theo chuẩn
        (["doc1", "doc2", "doc3"], ["doc1", "doc4", "doc2", "doc5", "doc3"], (1.0 + 2/3 + 3/5) / 3, "Multiple relevant"),
    ]
    
    for relevant, retrieved, expected, desc in test_cases:
        ap = AP(relevant, retrieved)
        print(f"{desc}: AP = {ap:.4f} (expected: {expected:.4f}) - {'✓' if abs(ap - expected) < 0.001 else '✗'}")



def test_AP():
    # Test case 1: Một số tài liệu liên quan trong kết quả truy xuất
    relevant_docs = ["doc1", "doc2", "doc3"]
    retrieved_docs = ["doc1", "doc4", "doc2", "doc5", "doc3"]
    ap = AP(relevant_docs, retrieved_docs)
    print(f"Test case 1 - AP: {ap:.4f}")
    assert ap == (1/1 + 2/3 + 3/5) / 3  # Average Precision = (1/1 + 2/3 + 3/5) / 3

    # Test case 2: Không có tài liệu liên quan trong kết quả truy xuất
    relevant_docs = ["doc6", "doc7"]
    retrieved_docs = ["doc1", "doc2", "doc3"]
    ap = AP(relevant_docs, retrieved_docs)
    print(f"Test case 2 - AP: {ap:.4f}")
    assert ap == 0.0  # Không có tài liệu nào liên quan, AP = 0.0

    # Test case 3: Tất cả tài liệu liên quan đều nằm trong kết quả truy xuất
    relevant_docs = ["doc1", "doc2", "doc3"]
    retrieved_docs = ["doc1", "doc2", "doc3"]
    ap = AP(relevant_docs, retrieved_docs)
    print(f"Test case 3 - AP: {ap:.4f}")
    assert ap == (1/1 + 2/2 + 3/3) / 3  # Average Precision = 1.0

    # Test case 4: Kết quả truy xuất không đầy đủ (chỉ chứa một phần tài liệu liên quan)
    relevant_docs = ["doc1", "doc2", "doc3"]
    retrieved_docs = ["doc1", "doc4", "doc5"]
    ap = AP(relevant_docs, retrieved_docs)
    print(f"Test case 4 - AP: {ap:.4f}")
    assert ap == (1/1) / 3  # Average Precision = 1.0 / 3 = 0.3333 (đúng)

    # Test case 5: Không có tài liệu liên quan nào (relevant_docs rỗng)
    relevant_docs = []
    retrieved_docs = ["doc1", "doc2", "doc3"]
    ap = AP(relevant_docs, retrieved_docs)
    print(f"Test case 5 - AP: {ap:.4f}")
    assert ap == 0.0  # Không có tài liệu liên quan, AP = 0.0

if __name__ == "__main__":
    test_ap()

    # inverted_index = {
    #     "statin": {
    #         "MED-10": {
    #             "text": {"count": 11, "positions": [3, 26, 53, 67, 69, 78, 111, 142, 155, 174, 184]},
    #         },
    #         "MED-14": {
    #             "title": {"count": 1, "positions": [0]},
    #             "text": {"count": 8, "positions": [5, 34, 56, 78, 99, 123, 145, 167]},
    #         },
    #         "MED-20": {
    #             "title": {"count": 5, "positions": [15, 45, 78, 102, 130]},
    #             "text": {"count": 4, "positions": [12, 34, 56, 78]},
    #         },
    #     }
    # }
    # # Tổng số tài liệu trong corpus (giả sử có 20 tài liệu)
    # total_docs = 20

    # tokens = ["statin"]
    # for token in tokens:
    #     print(f"TF-IDF scores for term '{token}':")
    #     scores = compute_tfidf(token, inverted_index, total_docs, alpha=0.7)
    #     if not scores:
    #         print("  Term not found in any document.")
    #         break
    #     for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    #         print(f"{doc}: TF-IDF = {score:.4f}")

