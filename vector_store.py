# vector_store.py (safe version for Streamlit Cloud)
"""
A safe vector store that:
- Loads sentence-transformers
- Falls back to simple search if FAISS unavailable
- Avoids numpy/npx type annotation errors
"""

import os
import json
import pickle

# Safe import handling
try:
    import numpy as np
except Exception:
    np = None     # Prevent AttributeError: NoneType.ndarray


class VectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print("SentenceTransformer not available:", e)
            self.model = None

        # Try FAISS
        try:
            import faiss
            self.faiss = faiss
        except Exception:
            self.faiss = None

    # ----------------------------------------------------------------------

    def _encode_texts(self, texts):
        """Encode texts to embeddings using available model."""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Install sentence-transformers.")

        vectors = self.model.encode(texts, convert_to_numpy=True)
        # Ensure proper shape
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        return vectors

    # ----------------------------------------------------------------------

    def create_embeddings(self, chunks):
        """Create embeddings for chunks."""
        self.chunks = chunks
        texts = []
        for c in chunks:
            if isinstance(c, dict):
                text = c.get("content") or c.get("text") or ""
            else:
                text = str(c)
            texts.append(text)

        vectors = self._encode_texts(texts)

        # If FAISS is available
        if self.faiss is not None:
            dim = vectors.shape[1]
            self.index = self.faiss.IndexFlatL2(dim)
            self.index.add(vectors.astype("float32"))
        else:
            # fallback: store vectors in memory
            self.index = vectors

    # ----------------------------------------------------------------------

    def save(self, path):
        """Save index + chunks"""
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        if self.faiss and isinstance(self.index, self.faiss.IndexFlatL2):
            self.faiss.write_index(self.index, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(self.index, f)

        # Save chunks
        with open(path + "_chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    # ----------------------------------------------------------------------

    def load(self, path):
        """Load index + chunks"""
        if self.faiss is not None and os.path.exists(path):
            try:
                self.index = self.faiss.read_index(path)
            except Exception:
                pass

        if self.index is None:
            # fallback pickle loader
            with open(path, "rb") as f:
                self.index = pickle.load(f)

        # Load chunks
        chunk_file = path + "_chunks.pkl"
        if os.path.exists(chunk_file):
            with open(chunk_file, "rb") as f:
                self.chunks = pickle.load(f)

    # ----------------------------------------------------------------------

    def search(self, query, k=5):
        """Search query using FAISS or fallback cosine similarity."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        q_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")

        # FAISS branch
        if self.faiss and hasattr(self.index, "search"):
            D, I = self.index.search(q_vec, k)
            results = [self.chunks[i] for i in I[0] if i < len(self.chunks)]
            return results

        # fallback numpy similarity
        if np is None:
            raise RuntimeError("Numpy not available for fallback search.")

        sims = np.dot(self.index, q_vec.T).flatten()
        topk = sims.argsort()[::-1][:k]
        return [self.chunks[i] for i in topk]
