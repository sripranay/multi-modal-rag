# create_embeddings.py
import os
import sys
import json
from vector_store import VectorStore
from config import PROCESSED_DATA_DIR, VECTOR_STORE_DIR

"""
This script loads extracted chunks from `processed/`,
creates embeddings, builds vector index (FAISS or NumPy fallback),
and stores results under `data/vector_store/`.
"""

def load_chunks(processed_dir: str):
    json_path = os.path.join(processed_dir, "chunks.json")
    if not os.path.exists(json_path):
        print("ERROR: chunks.json not found. Run process_document.py first.")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    if not chunks:
        print("ERROR: No chunks found in chunks.json")
        sys.exit(1)

    print(f"Loaded {len(chunks)} chunks for embedding.")
    return chunks


def main():
    print("\n==============================")
    print("STEP 2: Creating Embeddings")
    print("==============================\n")

    # Load extracted chunks
    chunks = load_chunks(PROCESSED_DATA_DIR)

    # Initialize VectorStore
    print("Initializing embedding model...")
    vs = VectorStore("sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings
    print("Generating embeddings (this may take a moment)...")
    vs.create_embeddings(chunks)

    # Save vector store
    print(f"Saving vector store to: {VECTOR_STORE_DIR}")
    vs.save(VECTOR_STORE_DIR)

    print("\nEmbeddings created and stored successfully!")
    print(f"Total vectors: {len(chunks)}")


if __name__ == "__main__":
    main()
