# vector_store.py
import os
import pickle
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

class VectorStore:
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.vectors = None  # numpy array (n, d)
        self.chunks = None   # list of chunk dicts
        self._use_faiss = False
        try:
            import faiss  # optional
            self.faiss = faiss
            self._use_faiss = True
        except Exception:
            self.faiss = None

    def load(self, path):
        """
        path is directory that contains vectors.npy and chunks.pkl
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        vectors_path = os.path.join(path, "vectors.npy")
        chunks_path = os.path.join(path, "chunks.pkl")
        if not os.path.exists(vectors_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError("Vector store files missing in " + path)
        if np is None:
            raise RuntimeError("numpy is required for VectorStore")
        self.vectors = np.load(vectors_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        # build faiss index if available
        if self._use_faiss:
            d = self.vectors.shape[1]
            self.index = self.faiss.IndexFlatIP(d)
            self.faiss.normalize_L2(self.vectors)
            self.index.add(self.vectors)
        else:
            self.index = None

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(path, "vectors.npy"), self.vectors)
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def create_embeddings(self, chunks, embeddings):
        """
        If embeddings already provided (numpy array), use them and store
        chunks: list of dicts
        embeddings: numpy ndarray shape (n, d)
        """
        if np is None:
            raise RuntimeError("numpy required")
        self.chunks = chunks
        self.vectors = embeddings
        if self._use_faiss:
            d = embeddings.shape[1]
            self.index = self.faiss.IndexFlatIP(d)
            self.faiss.normalize_L2(self.vectors)
            self.index.add(self.vectors)

    def search(self, query_vector, k=5):
        """
        query_vector: ndarray (d,) or text query (not handled here).
        Returns list of dicts: {"chunk": chunk, "score": score}
        """
        if self.vectors is None:
            return []
        if isinstance(query_vector, (list, tuple)):
            import numpy as _np
            q = _np.array(query_vector)
        else:
            q = query_vector
        if self._use_faiss and self.index is not None:
            import numpy as _np
            _np.testing.assert_array_almost_equal(q.shape, (self.vectors.shape[1],), err_msg="query vector has wrong shape")
            q_norm = q.reshape(1, -1).astype("float32")
            self.faiss.normalize_L2(q_norm)
            D, I = self.index.search(q_norm, k)
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                results.append({"chunk": self.chunks[idx], "score": float(score)})
            return results
        else:
            # numpy-based cosine similarity
            import numpy as _np
            if q.ndim == 1:
                q = q.reshape(1, -1)
            # normalize
            v = self.vectors
            qn = q / ( _np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            vn = v / ( _np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            scores = (vn @ qn.T).squeeze()  # dot product with normalized vectors -> cosine sims
            idxs = scores.argsort()[::-1][:k]
            results = []
            for idx in idxs:
                results.append({"chunk": self.chunks[idx], "score": float(scores[idx])})
            return results
