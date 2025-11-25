# vector_store.py 
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(self.model_name)
        self.chunks = []
        self.vectors = None  # numpy array (n, d)

    def _embed_texts(self, texts):
        embs = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True)
        embs = embs.astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        embs = embs / norms
        return embs

    def create_embeddings(self, chunks):
        """
        chunks: list of dicts with 'content' key
        """
        self.chunks = chunks[:]
        texts = [str(c.get("content","")) for c in self.chunks]
        if len(texts) == 0:
            self.vectors = np.zeros((0,1), dtype=np.float32)
            return
        # batch embedding
        emb_list = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            emb_list.append(self._embed_texts(batch))
        self.vectors = np.vstack(emb_list)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # save chunks and vectors
        with open(path + "_chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        if self.vectors is not None:
            np.save(path + "_vectors.npy", self.vectors)

    def load(self, path):
        chunks_path = path + "_chunks.json"
        vectors_path = path + "_vectors.npy"
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
            self.chunks = []
        if os.path.exists(vectors_path):
            self.vectors = np.load(vectors_path).astype(np.float32)
        else:
            self.vectors = None

    def _search_numpy(self, query_emb, k=5):
        if self.vectors is None or self.vectors.shape[0] == 0:
            return []
        scores = np.dot(self.vectors, query_emb.astype(np.float32))
        if k >= len(scores):
            idxs = np.argsort(-scores)
        else:
            idxs = np.argpartition(-scores, k-1)[:k]
            idxs = idxs[np.argsort(-scores[idxs])]
        return [(float(scores[i]), int(i)) for i in idxs]

    def search(self, query, k=5):
        if not query:
            return []
        q_emb = self._embed_texts([query])[0]
        norm = np.linalg.norm(q_emb)
        if norm > 0:
            q_emb = q_emb / norm
        hits = self._search_numpy(q_emb, k=k)
        results = []
        for score, idx in hits:
            chunk = self.chunks[idx] if idx < len(self.chunks) else {}
            results.append({"chunk": chunk, "score": score})
        return results
