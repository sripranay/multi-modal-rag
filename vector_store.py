
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json

class VectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.index = None
        self.chunks = []

    def create_embeddings(self, chunks):
        texts = [c["content"] for c in chunks]
        self.chunks = chunks

        print("Embedding", len(texts), "chunks...")
        vectors = self.model.encode(texts, convert_to_numpy=True)
        self.embeddings = vectors

        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        self.index = index

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        faiss.write_index(self.index, path + ".faiss")

        with open(path + "_chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load(self, path):
        faiss_path = path + ".faiss"
        chunks_path = path + "_chunks.json"

        if not os.path.exists(faiss_path):
            raise FileNotFoundError("FAISS file not found: " + faiss_path)
        if not os.path.exists(chunks_path):
            raise FileNotFoundError("Chunks file not found: " + chunks_path)

        self.index = faiss.read_index(faiss_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, query, k=5):
        q_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk": chunk,
                "score": float(score),
                "source": chunk.get("source", "unknown")
            })
        return results
