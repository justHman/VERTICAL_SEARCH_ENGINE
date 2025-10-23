import json
import os 

def load_corpus(path):
    corpus = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())  
            corpus.append(data)
    return corpus

def load_inverted_index(path):
    with open(path, 'r', encoding='utf-8') as file:
        inverted_index = json.load(file)
    return inverted_index

def load_env(path="ENV.json"):
    with open(path, 'r', encoding='utf-8') as file:
        env = json.load(file)
    return env

def load_queries(path):
    queries = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            if "_id" in data and "text" in data:
                queries[data["_id"]] = data["text"]
    return queries

if __name__ == "__main__":
    env = load_env()
    print(env)