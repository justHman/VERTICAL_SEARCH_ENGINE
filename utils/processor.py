import re
import unicodedata
import spacy

nlp = spacy.load("en_core_web_sm")

def normalize_text(text):
    # Loại bỏ escape unicode như \u2009, \u00a9, ...
    text = re.sub(r'\\u[0-9a-fA-F]{4}', ' ', text)
    # Loại bỏ emoji/icon (ký tự ngoài BMP)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # Loại bỏ ký tự đặc biệt (giữ lại chữ, số, khoảng trắng)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Chuẩn hóa unicode (loại bỏ dấu, ký tự lạ)
    text = unicodedata.normalize('NFKD', text)
    # Chuyển về chữ thường
    text = text.lower()
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    return [token.text for token in nlp(text) if not token.is_space]

def remove_stopwords(tokens):
    return [token for token in tokens if not nlp.vocab[token].is_stop]

def lemmatize(tokens):
    return [nlp(token)[0].lemma_ for token in tokens]

def Text2Tokens(text):
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    tokens_no_stopwords = remove_stopwords(tokens)
    lemmas = lemmatize(tokens_no_stopwords)
    return lemmas

def merge_ranges(ranges):
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda x: x[0])
    merged = [list(ranges[0])]
    for s, e in ranges[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged

if __name__ == "__main__":
    sample_text = "Hello, World! This is a test text with emoji 😊 @   and unicode \\u2009 characters."
    processed = Text2Tokens(sample_text)
    print("Original Text:", sample_text)
    print("Processed Tokens:", processed)