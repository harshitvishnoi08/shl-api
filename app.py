from flask import Flask, request, jsonify
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"

# Initialize Flask app
app = Flask(__name__)

# Load FAISS index and dataframe once at startup
index = faiss.read_index("shl_index.faiss")
with open("shl_dataframe.pkl", "rb") as f:
    df = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Helper: retrieve candidates
def retrieve_candidates(query, top_k=20):
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    candidates = []
    for score, idx in zip(D[0], I[0]):
        row = df.iloc[idx]
        candidates.append({
            "name": row["name"],
            "link": row["link"],
            "description": row["description"],
            "duration": row["duration"],
            "test_types": row["test_types"],
            "remote_testing": row["remote_testing"],
            "adaptive_irt": row["adaptive_irt"]
        })
    return candidates

# Helper: GROQ rerank
def groq_rerank(query, candidates, rerank_k=10):
    items = "\n".join(
        f"{i+1}. {c['name']} — {c['description'][:100]} (Duration: {c['duration']} mins)"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"You are an expert assessment recommender. The hiring need is:\n\n"
        f"“{query}”\n\n"
        f"Here are {len(candidates)} candidate assessments:\n{items}\n\n"
        f"Please rank the top {rerank_k} most relevant assessments by returning a comma-separated list "
        f"of their numbers in descending order of relevance (e.g., 3,1,5,...)."
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that ranks assessment tests from a given list."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    response = requests.post(GROQ_URL, headers=headers, json=payload)
    result = response.json()
    try:
        text = result["choices"][0]["message"]["content"]
        indices = [int(x)-1 for x in text.split(",") if x.strip().isdigit()]
    except Exception:
        indices = list(range(min(rerank_k, len(candidates))))
    return [candidates[i] for i in indices[:rerank_k]]

# Unified recommend
@app.route('/recommend', methods=['GET'])
def recommend_api():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    try:
        retrieve_k = int(request.args.get('retrieve_k', 20))
        rerank_k = int(request.args.get('rerank_k', 10))
    except ValueError:
        return jsonify({"error": "Parameters 'retrieve_k' and 'rerank_k' must be integers"}), 400

    candidates = retrieve_candidates(query, top_k=retrieve_k)
    results = groq_rerank(query, candidates, rerank_k=rerank_k)
    return jsonify({
        "query": query,
        "results": results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
