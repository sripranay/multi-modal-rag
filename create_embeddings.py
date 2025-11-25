# create_embeddings.py
import json
import os
import pickle
from pathlib import Path

try:
    import config
    CHUNKS_PATH = config.CHUNKS_PATH
    VECTOR_STORE_DIR = config.VECTOR_STORE_DIR
    EMBEDDING_MODEL = config.EMBEDDING_MODEL
except Exception:
    CHUNKS_PATH = os.path.join("data", "processed", "chunks.json")
    VECTOR_STORE_DIR = os.path.join("data", "vector_store")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)

def load_chunks(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERROR: chunks.json not found. Run process_document.py first. ({path})")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_embeddings(texts, model_name=EMBEDDING_MODEL):
    # sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return vectors
    except Exception as e:
        # fallback to transformers pipeline (less ideal)
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            with torch.no_grad():
                encs = []
                for t in texts:
                    inputs = tokenizer(t, truncation=True, padding=True, return_tensors="pt")
                    out = model(**inputs, return_dict=True)
                    # mean pooling
                    last_hidden = out.last_hidden_state
                    mask = inputs['attention_mask'].unsqueeze(-1)
                    pooled = (last_hidden * mask).sum(1) / mask.sum(1)
                    encs.append(pooled[0].cpu().numpy())
                import numpy as np
                return np.vstack(encs)
        except Exception as e2:
            raise RuntimeError("No embedding model available: " + str(e))

def save_vector_store(vectors, chunks, out_dir=VECTOR_STORE_DIR):
    import numpy as np
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(out_dir, "vectors.npy"), vectors)
    with open(os.path.join(out_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    print("Saved vector store to", out_dir)

def main():
    print("STEP 2: Creating Embeddings\n")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks")
    texts = [c.get("content","")[:10000] for c in chunks]  # truncate very long content
    print("Computing embeddings with model:", EMBEDDING_MODEL)
    vectors = compute_embeddings(texts, EMBEDDING_MODEL)
    print("Embeddings shape:", getattr(vectors, "shape", None))
    save_vector_store(vectors, chunks)
    print("COMPLETE. Total vectors:", len(vectors))

if __name__ == "__main__":
    main()
