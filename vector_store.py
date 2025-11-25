# vector_store.py
"""
VectorStore with optional FAISS backend and a NumPy fallback.

API:
    vs = VectorStore(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs.create_embeddings(chunks)          # chunks: list of dicts {id, content, type, ...}
    vs.save(path)                         # saves files under `path` directory
    vs.load(path)                         # loads previously saved index
    hits = vs.search(query, k=5)          # returns list of dicts: {'chunk': chunk, 'score':score}
"""

import os
import json
import errno
from typing import List, Dict, Any
import pathlib

# Embedding/model imports
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None
    _st_error = e

# numeric ops
try:
    import numpy as np
except Exception as e:
    np = None
    _np_error = e

# Optional faiss
try:
    import faiss
    _has_faiss = True
except Exception:
    faiss = None
    _has_faiss = False

# helper
def _ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", normalize: bool = True):
        if SentenceTransformer is None:
            raise ImportError(f"sentence-transformers is required but not installed: {_st_error}")

        if np is None:
            raise ImportError(f"numpy is required but not installed: {_np_error}")

        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: np.ndarray = None  # shape (N, D)
        self.index = None
        self.dim = None
        self._faiss_index_path = None

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        # sentence-transformers returns np.array
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        if self.normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms
        return embs

    def create_embeddings(self, chunks: List[Dict[str, Any]]):
        """
        chunks: list of dicts with at least 'content' key
        """
        if not chunks:
            raise ValueError("Empty chunks list")

        self.chunks = chunks
        texts = [c.get("content", "") for c in chunks]
        vectors = self._encode_texts(texts)  # (N, D)
        self.vectors = vectors.astype(np.float32)
        self.dim = self.vectors.shape[1]

        # Try to build FAISS index if available
        if _has_faiss:
            try:
                self.index = faiss.IndexFlatIP(self.dim)  # inner product on normalized vectors -> cosine
                self.index.add(self.vectors)
                return
            except Exception as e:
                # fallback to numpy if faiss construction fails
                print(f"FAISS index construction failed, falling back to NumPy search: {e}")
                self.index = None
        else:
            # no faiss available
            self.index = None

    def save(self, dirpath: str):
        """
        Save vectors and metadata:
          - dirpath/metadata.json  (list of chunks)
          - dirpath/vectors.npz    (numpy arrays)
          - optional: dirpath/faiss.index
        """
        _ensure_dir(dirpath)
        # metadata
        meta_path = os.path.join(dirpath, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        # vectors
        vec_path = os.path.join(dirpath, "vectors.npz")
        np.savez_compressed(vec_path, vectors=self.vectors)

        # faiss index if present
        if _has_faiss and self.index is not None:
            try:
                faiss.write_index(self.index, os.path.join(dirpath, "faiss.index"))
            except Exception as e:
                # non-fatal
                print(f"Failed to save FAISS index: {e}")

    def load(self, dirpath: str):
        """
        Load previously saved store. Fills self.chunks and self.vectors.
        If faiss.index is present and faiss is available, load it.
        """
        meta_path = os.path.join(dirpath, "metadata.json")
        vec_path = os.path.join(dirpath, "vectors.npz")
        if not os.path.exists(meta_path) or not os.path.exists(vec_path):
            raise FileNotFoundError("No stored vector store found at: " + dirpath)

        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        loader = np.load(vec_path, allow_pickle=True)
        self.vectors = loader["vectors"].astype(np.float32)
        self.dim = self.vectors.shape[1]

        # try to load faiss
        if _has_faiss:
            faiss_path = os.path.join(dirpath, "faiss.index")
            if os.path.exists(faiss_path):
                try:
                    self.index = faiss.read_index(faiss_path)
                    return
                except Exception as e:
                    print(f"Failed loading FAISS index; falling back to NumPy: {e}")
                    self.index = None
        else:
            self.index = None

    def _numpy_search(self, query_vector: np.ndarray, k: int = 5):
        # both vectors are normalized so cosine == dot product
        # query_vector: (D,), self.vectors: (N, D)
        if self.vectors is None:
            return []

        # compute dot products
        scores = np.dot(self.vectors, query_vector.astype(np.float32))
        # get top-k
        topk_idx = np.argsort(-scores)[:k]
        results = []
        for idx in topk_idx:
            results.append({"chunk": self.chunks[int(idx)], "score": float(scores[int(idx)])})
        return results

    def search(self, query: str, k: int = 5):
        """
        Accepts text query, returns list of {chunk, score}
        """
        if self.vectors is None or not self.chunks:
            return []

        q_vec = self._encode_texts([query])[0]  # shape (D,)
        if self.index is not None and _has_faiss:
            # FAISS returns distances; using IndexFlatIP on normalized vectors returns similarity scores
            D, I = self.index.search(np.expand_dims(q_vec.astype(np.float32), axis=0), k)
            results = []
            for score, idx in zip(D[0], I[0]):
                if int(idx) < 0:
                    continue
                results.append({"chunk": self.chunks[int(idx)], "score": float(score)})
            return results
        else:
            # numpy fallback
            return self._numpy_search(q_vec, k=k)
