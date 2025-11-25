# create_embeddings.py
"""
Create embeddings from pre-processed chunks and build/save a vector store.

Usage:
    python create_embeddings.py              # uses config.CHUNKS_PATH
    python create_embeddings.py --chunks path/to/chunks.json
"""

import os
import json
import argparse
import sys
import traceback

try:
    import config
except Exception:
    print("ERROR: could not import config.py. Make sure this file exists and is on PYTHONPATH.")
    raise

# Prefer local VectorStore implementation. Make sure vector_store.py is present.
try:
    from vector_store import VectorStore
except Exception as e:
    print("ERROR: could not import VectorStore from vector_store.py.")
    print("Detail:", e)
    raise

def load_chunks(path: str):
    if not os.path.exists(path):
        print(f"ERROR: chunks file not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return chunks
    except Exception as exc:
        print(f"ERROR: failed to load chunks from {path}: {exc}")
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", help="Path to extracted chunks JSON", default=None)
    parser.add_argument("--vector_store", help="Path to save vector store", default=None)
    args = parser.parse_args()

    chunks_path = args.chunks or getattr(config, "CHUNKS_PATH", None) or getattr(config, "CHUNKS_PATH", None)
    if not chunks_path:
        chunks_path = getattr(config, "CHUNKS_PATH", None) or getattr(config, "CHUNKS_PATH", None)

    if not chunks_path:
        print("ERROR: No chunks path provided and config.CHUNKS_PATH not found.")
        sys.exit(1)

    print("="*60)
    print("STEP 2: Creating Embeddings")
    print("="*60)
    print()

    chunks = load_chunks(chunks_path)
    if chunks is None:
        print("ERROR: chunks.json not found. Run process_document.py first.")
        sys.exit(1)

    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    text_count = sum(1 for c in chunks if c.get("type") == "text")
    table_count = sum(1 for c in chunks if c.get("type") == "table")
    image_count = sum(1 for c in chunks if c.get("type") == "image")
    print(f"  - Text chunks: {text_count}")
    print(f"  - Tables: {table_count}")
    print(f"  - Images: {image_count}")
    print()

    # instantiate vector store
    vs_path = args.vector_store or getattr(config, "VECTOR_STORE_PATH", None)
    model_name = getattr(config, "EMBEDDING_MODEL", None)
    if not model_name:
        print("WARNING: config.EMBEDDING_MODEL not set. Using sentence-transformers/all-MiniLM-L6-v2 as fallback.")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print(f"Creating VectorStore with model: {model_name}")
    try:
        vs = VectorStore(model_name=model_name)
    except Exception as e:
        print("ERROR: Failed to create VectorStore instance. See details below:")
        traceback.print_exc()
        sys.exit(1)

    print("Creating embeddings... (this may take a while depending on model/hardware)")
    try:
        vs.create_embeddings(chunks)
    except Exception as e:
        print("ERROR: create_embeddings failed.")
        traceback.print_exc()
        sys.exit(1)

    # Save vector store
    if not vs_path:
        print("No VECTOR_STORE_PATH provided in args or config — attempting to use config.VECTOR_STORE_PATH")
        vs_path = getattr(config, "VECTOR_STORE_PATH", None)

    if not vs_path:
        print("WARNING: No path to save vector store provided — skipping save step.")
    else:
        # ensure directory exists
        parent = os.path.dirname(vs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            vs.save(vs_path)
            print(f"Saved vector store to: {vs_path}")
        except Exception as e:
            print(f"ERROR: failed to save vector store to {vs_path}: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("COMPLETE")
    print(f"Total vectors: {len(chunks)}")

if __name__ == "__main__":
    main()
