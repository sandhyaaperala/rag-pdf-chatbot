import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query, filename="vector_store.json", top_k=3):

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    query_embedding = model.encode([query])

    scores = []

    for item in data:

        embedding = np.array(item["embedding"]).reshape(1, -1)

        score = cosine_similarity(query_embedding, embedding)[0][0]

        scores.append((score, item["chunk"]))

    scores.sort(reverse=True)

    return scores[:top_k]


if __name__ == "__main__":

    query = input("Ask a question: ")

    results = retrieve(query)

    print("\nTop Results:\n")

    for score, chunk in results:

        print("Score:", round(score, 4))
        print("-" * 60)
        print(chunk[:500])
        print("\n")
