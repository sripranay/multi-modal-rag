"""
vector_store.py

Simple VectorStore wrapper using sentence-transformers + faiss.
Methods:
  - create_embeddings(chunks): compute embeddings for each chunk and build FAISS
  - save(path): saves index and chunks metadata
  - load(path): loads index and chunks metadata
  - search(query, k=5): returns list of dicts {"chunk": <chunk>, "score": <score>}
"""
import os
import json
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", dim=None):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.dim = dim

    def _ensure_model(self):
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            # set dim if not set
            if self.dim is None:
                self.dim = self.model.get_sentence_embedding_dimension()
            print("successfully loaded")

    def create_embeddings(self, chunks):
        """
        chunks: list of dicts with 'content' key
        """
        self._ensure_model()
        self.chunks = chunks
        texts = [c.get("content", "") for c in chunks]
        # encode in batches
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # ensure dtype float32 for faiss
        embeddings = np.array(embeddings).astype("float32")
        # build index
        self.index = faiss.IndexFlatIP(self.dim)  # inner product for cosine (we will normalize)
        # normalize embeddings for cosine
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"FAISS index with {self.index.ntotal} vectors")

    def save(self, path):
        """
        Save index and chunks metadata. 'path' is directory path.
        """
        os.makedirs(path, exist_ok=True)
        # index file
        idx_file = os.path.join(path, "index.faiss")
        faiss.write_index(self.index, idx_file)
        # metadata
        meta_file = os.path.join(path, "chunks.pkl")
        with open(meta_file, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"Saved index to {path}")

    def load(self, path):
        """
        Load index and metadata from directory.
        """
        idx_file = os.path.join(path, "index.faiss")
        meta_file = os.path.join(path, "chunks.pkl")
        if not os.path.exists(idx_file) or not os.path.exists(meta_file):
            raise FileNotFoundError("Index or metadata missing in " + path)
        self.index = faiss.read_index(idx_file)
        with open(meta_file, "rb") as f:
            self.chunks = pickle.load(f)
        # set dim
        self.dim = self.index.d
        # lazy-load model only when creating embeddings or searching with query->embedding
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")

    def search(self, query, k=5):
        """
        Returns list of dicts: {"chunk": chunk, "score": score}
        If model not loaded (for encoding), instantiate it transiently.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded")
        # ensure model available to encode query
        self._ensure_model()
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append({"chunk": self.chunks[int(idx)], "score": float(score)})
        return results
