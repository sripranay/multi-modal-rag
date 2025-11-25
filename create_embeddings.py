# create_embeddings.py
"""
Creates embeddings for chunks stored in data/processed/chunks.json.

Behaviour:
- Try to use sentence-transformers (all-MiniLM-L6-v2) if installed.
- If sentence-transformers (or torch) is unavailable, fall back to a simple
  deterministic count-based vectorizer implemented in pure Python + numpy.
- Saves:
    data/processed/embeddings.npy  (float32 matrix: n_chunks x dim)
    data/processed/metadata.json   (list of dicts with "index" and "text" for each chunk)
"""
import os
import json
import sys
from typing import List
from pathlib import Path

ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED / "chunks.json"
EMBED_PATH = PROCESSED / "embeddings.npy"
META_PATH = PROCESSED / "metadata.json"

def load_chunks(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"ERROR: chunks.json not found. Run process_document.py first. ({path})")
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a list of chunk strings.")
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks

def embed_with_sentence_transformers(chunks: List[str], model_name: str = "all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print("sentence-transformers not available:", e)
        raise

    print("Loading sentence-transformers model:", model_name)
    model = SentenceTransformer(model_name)
    print("Encoding chunks with sentence-transformers...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype("float32")

def fallback_count_embeddings(chunks: List[str], max_vocab:int=5000):
    """
    Simple fallback embedding:
    - Build vocabulary of most common words across chunks (lowercased, alpha tokens)
    - For each chunk compute term-frequency vector over that vocabulary
    - L2-normalize vectors
    """
    import re
    from collections import Counter
    import numpy as np

    print("Using fallback count-based embeddings (no sentence-transformers).")
    token_re = re.compile(r"[A-Za-z0-9]+")
    counters = []
    global_counter = Counter()
    for t in chunks:
        toks = token_re.findall(t.lower())
        c = Counter(toks)
        counters.append(c)
        global_counter.update(c)

    # take top-k frequent tokens as vocabulary
    vocab = [w for w, _ in global_counter.most_common(max_vocab)]
    vocab_index = {w:i for i,w in enumerate(vocab)}
    print(f"Fallback vocab size: {len(vocab)}")

    # build matrix
    mat = np.zeros((len(chunks), len(vocab)), dtype="float32")
    for i, c in enumerate(counters):
        for token, cnt in c.items():
            idx = vocab_index.get(token)
            if idx is not None:
                mat[i, idx] = cnt

    # L2-normalize rows (avoid div-by-zero)
    norms = (mat * mat).sum(axis=1) ** 0.5
    norms[norms==0] = 1.0
    mat = mat / norms[:, None]
    return mat

def save_embeddings(embeddings, meta_texts, embed_path: Path, meta_path: Path):
    import numpy as np
    embed_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(embed_path), embeddings)
    meta = [{"index": i, "text": text} for i, text in enumerate(meta_texts)]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved embeddings -> {embed_path} (shape: {embeddings.shape})")
    print(f"Saved metadata -> {meta_path} (n={len(meta)})")

def main(chunks_path=CHUNKS_PATH, embed_path=EMBED_PATH, meta_path=META_PATH):
    chunks = load_chunks(Path(chunks_path))
    embeddings = None
    # Try transformer embeddings first
    try:
        embeddings = embed_with_sentence_transformers(chunks)
    except Exception:
        # fallback
        embeddings = fallback_count_embeddings(chunks)

    # Ensure numpy array shape is correct
    try:
        import numpy as np
    except Exception as e:
        print("numpy is required but not available:", e)
        sys.exit(2)

    embeddings = np.asarray(embeddings, dtype="float32")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise RuntimeError("Embeddings shape mismatch")

    save_embeddings(embeddings, chunks, Path(embed_path), Path(meta_path))

if __name__ == "__main__":
    # allow optional custom path
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default=str(CHUNKS_PATH))
    p.add_argument("--embeddings", default=str(EMBED_PATH))
    p.add_argument("--meta", default=str(META_PATH))
    args = p.parse_args()
    main(chunks_path=args.chunks, embed_path=args.embeddings, meta_path=args.meta)
