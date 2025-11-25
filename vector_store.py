# vector_store.py
"""
Simple vector store using sentence-transformers + FAISS (with numpy fallback).
Expected chunk format: list of dicts with at least:
    {
      "id": optional unique id,
      "content": "text content",
      "source": "source or page",
      "type": "text" / "table" / "image",
      ...
    }
"""

import os
import pickle
from typing import List, Dict, Optional

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embed_batch_size: int = 64,
    ):
        """
        model_name: huggingface sentence-transformers model id
        embed_batch_size: embed documents in batches to reduce memory usage
        """
        self.model_name = model_name
        self.embed_batch_size = embed_batch_size
        self.model = None  # lazy load
        self.index = None
        self.vectors = None  # numpy array (n, d) of normalized vectors
        self.chunks: List[Dict] = []  # metadata / original chunks

    def _ensure_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Return L2-normalized embeddings as float32 numpy array shape (n, d)
        """
        self._ensure_model()
        embs = self.model.encode(texts, batch_size=self.embed_batch_size, convert_to_numpy=True)
        embs = embs.astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        embs = embs / norms
        return embs

    def create_embeddings(self, chunks: List[Dict]):
        """
        Create embeddings for provided chunks and build an in-memory index.
        Overwrites any existing index in memory.
        """
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list of dicts")

        self.chunks = chunks[:]  # shallow copy of metadata

        # Collect text to embed (use chunk['content'] or empty string)
        texts = [str(c.get("content", "")) for c in self.chunks]
        if len(texts) == 0:
            # create empty index
            self.vectors = np.zeros((0, 1), dtype=np.float32)
            self.index = None
            return

        # Compute embeddings in batches to limit memory usage
        emb_list = []
        batch = []
        for i, t in enumerate(texts, start=1):
            batch.append(t)
            if len(batch) >= self.embed_batch_size:
                emb_list.append(self._embed_texts(batch))
                batch = []
        if batch:
            emb_list.append(self._embed_texts(batch))

        self.vectors = np.vstack(emb_list).astype(np.float32)

        if _FAISS_AVAILABLE:
            d = self.vectors.shape[1]
            # Use inner product on normalized vectors -> equivalent to cosine similarity
            index = faiss.IndexFlatIP(d)
            # convert to contiguous float32
            index.add(self.vectors)
            self.index = index
        else:
            # no faiss; we'll rely on numpy-based brute force search
            self.index = None

    def save(self, path: str):
        """
        Save index and metadata to disk.
        `path` is the base path (without extension). We'll create:
            path + ".faiss"  -> faiss index (if faiss available)
            path + "_vectors.npy" -> numpy vectors fallback
            path + "_chunks.pkl" -> metadata pickle
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # save metadata (chunks)
        chunks_path = f"{path}_chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        # save vectors & index
        if self.vectors is None:
            # nothing to save
            return

        vectors_path = f"{path}_vectors.npy"
        np.save(vectors_path, self.vectors)

        if _FAISS_AVAILABLE and self.index is not None:
            faiss_path = f"{path}.faiss"
            faiss.write_index(self.index, faiss_path)

    def load(self, path: str):
        """
        Load index and metadata from disk. Path is base path as used in save().
        """
        chunks_path = f"{path}_chunks.pkl"
        vectors_path = f"{path}_vectors.npy"
        faiss_path = f"{path}.faiss"

        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        # load vectors if present
        if os.path.exists(vectors_path):
            self.vectors = np.load(vectors_path).astype(np.float32)
        else:
            self.vectors = None

        # load faiss index if available
        if _FAISS_AVAILABLE and os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
        else:
            # if faiss not available or faiss file missing fallback to None (use numpy search)
            self.index = None

    def _search_faiss(self, query_emb: np.ndarray, k: int = 5):
        """
        query_emb should be (d,) normalized vector
        Returns list of (score, idx)
        """
        if self.index is None:
            return []
        # faiss expects shape (1, d)
        q = np.expand_dims(query_emb.astype(np.float32), axis=0)
        D, I = self.index.search(q, k)
        # D: similarity scores (inner-product), I: indices
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append((float(score), int(idx)))
        return results

    def _search_numpy(self, query_emb: np.ndarray, k: int = 5):
        """
        Brute-force search using numpy. Returns list of (score, idx)
        """
        if self.vectors is None or self.vectors.shape[0] == 0:
            return []
        # inner product (vectors and query already normalized -> cosine similarity)
        scores = np.dot(self.vectors, query_emb.astype(np.float32))
        # get top-k indices
        if k >= len(scores):
            idxs = np.argsort(-scores)
        else:
            idxs = np.argpartition(-scores, k - 1)[:k]
            idxs = idxs[np.argsort(-scores[idxs])]
        results = [(float(scores[i]), int(i)) for i in idxs]
        return results

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for `query` and return top-k results as a list of dicts:
          {"chunk": <chunk dict>, "score": <similarity score>}
        """
        if not query:
            return []

        # embed query
        query_emb = self._embed_texts([query])[0]  # shape (d,)
        # ensure normalized (should be already)
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        # use faiss if available & index built
        if _FAISS_AVAILABLE and self.index is not None:
            hits = self._search_faiss(query_emb, k=k)
        else:
            hits = self._search_numpy(query_emb, k=k)

        results = []
        for score, idx in hits:
            chunk = self.chunks[idx] if idx < len(self.chunks) else {}
            results.append({"chunk": chunk, "score": score})
        return results


# Optional quick test if run as script
if __name__ == "__main__":
    # small sanity check / demo
    docs = [
        {"content": "The cat sat on the mat", "source": "doc1", "type": "text"},
        {"content": "Dog and wolf are canines", "source": "doc2", "type": "text"},
        {"content": "Economic report with GDP numbers", "source": "doc3", "type": "text"},
    ]
    vs = VectorStore()
    vs.create_embeddings(docs)
    print("Vectors shape:", None if vs.vectors is None else vs.vectors.shape)
    res = vs.search("gdp numbers", k=2)
    print("Search results:", res)
