# create_embeddings.py — FINAL WORKING VERSION
import os
import json
import numpy as np
from vector_store import VectorStore
import config

def main(chunks_path: str = None):
    print("\n====================================================")
    print("STEP 2: Creating Embeddings")
    print("====================================================\n")

    # Determine chunks.json location
    chunks_file = chunks_path or config.CHUNKS_PATH

    if not os.path.exists(chunks_file):
        print(f"ERROR: chunks.json not found at: {chunks_file}")
        print("Run process_document.py first.")
        return

    # Load chunks
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Build embeddings
    vs = VectorStore(model_name=config.EMBEDDING_MODEL)
    vs.build(chunks)

    # Save FAISS and metadata
    os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
    vs.save(config.VECTOR_STORE_PATH)

    print("\n✅ Embeddings created and saved successfully!")
    print(f"Vector store saved at: {config.VECTOR_STORE_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=str, help="Path to chunks.json", default=None)
    args = parser.parse_args()

    main(args.chunks)
