#!/usr/bin/env python3
"""
create_embeddings.py

Usage:
    python create_embeddings.py
    python create_embeddings.py --chunks data/processed/extracted_chunks.json --model sentence-transformers/all-MiniLM-L6-v2
"""

import json
import os
import argparse
import sys
import traceback

# prefer local config
try:
    import config
except Exception:
    config = None

def main():
    parser = argparse.ArgumentParser(description="Create embeddings and build/save vector store")
    parser.add_argument("--chunks", "-c", help="Path to extracted chunks JSON file", default=None)
    parser.add_argument("--model", "-m", help="Embedding model name", default=None)
    parser.add_argument("--vector-dir", "-v", help="Directory to save vector store", default=None)
    args = parser.parse_args()

    # determine paths from args or config.py
    chunks_path = args.chunks or (getattr(config, "CHUNKS_PATH", None) if config else None)
    embedding_model = args.model or (getattr(config, "EMBEDDING_MODEL", None) if config else None)
    vector_store_dir = args.vector_dir or (getattr(config, "VECTOR_STORE_DIR", None) if config else None)

    print("\n" + "="*70)
    print("STEP 2: Creating Embeddings")
    print("="*70 + "\n")

    if not chunks_path:
        print("ERROR: No chunks path provided and config.CHUNKS_PATH not found.")
        sys.exit(1)

    if not os.path.exists(chunks_path):
        print(f"ERROR: chunks.json not found at: {chunks_path}")
        print("Run process_document.py first to extract chunks.")
        sys.exit(1)

    # load chunks
    try:
        print(f"Loading extracted chunks from: {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        print("Failed to load chunks JSON:")
        traceback.print_exc()
        sys.exit(1)

    print(f"Loaded {len(chunks)} chunks")
    text_count = sum(1 for c in chunks if c.get("type") == "text")
    table_count = sum(1 for c in chunks if c.get("type") == "table")
    image_count = sum(1 for c in chunks if c.get("type") == "image")
    print(f" - Text chunks: {text_count}")
    print(f" - Table chunks: {table_count}")
    print(f" - Image chunks: {image_count}")

    # ensure vector store directory exists (if provided)
    if vector_store_dir:
        os.makedirs(vector_store_dir, exist_ok=True)
        print(f"Vector store directory: {vector_store_dir}")

    # import VectorStore (your implementation)
    try:
        from vector_store import VectorStore
    except Exception:
        print("Failed to import VectorStore from vector_store.py. Traceback:")
        traceback.print_exc()
        sys.exit(1)

    # create embeddings
    try:
        print("\nCreating embeddings...")
        if embedding_model:
            print(f"Using embedding model: {embedding_model}")
            vs = VectorStore(model_name=embedding_model)
        else:
            print("No embedding model specified. Using default VectorStore() constructor.")
            vs = VectorStore()

        # create embeddings from chunks
        # expected method: create_embeddings(chunks) OR create_embeddings(list_of_texts)
        if hasattr(vs, "create_embeddings"):
            vs.create_embeddings(chunks)
        else:
            # fallback: try to encode texts manually if API differs
            print("VectorStore.create_embeddings not found; trying fallback encode->add API.")
            texts = [c.get("content","") for c in chunks]
            if hasattr(vs, "encode") and hasattr(vs, "add"):
                vectors = vs.encode(texts)
                vs.add(vectors, chunks)
            else:
                raise RuntimeError("VectorStore does not expose create_embeddings or encode/add methods.")
    except Exception:
        print("Error while creating embeddings:")
        traceback.print_exc()
        sys.exit(1)

    # save vector store
    try:
        save_path = vector_store_dir or (getattr(config, "VECTOR_STORE_PATH", None) if config else None)
        if not save_path:
            # if vector_store_dir was a directory path, pass that; else try default
            save_path = getattr(config, "VECTOR_STORE_DIR", None) if config else None

        # attempt sensible save API: vs.save(path) or vs.save_index(path)
        if hasattr(vs, "save"):
            # if config stores a directory, pass that; else pass path string
            try:
                vs.save(save_path)
                print(f"Saved vector store to: {save_path}")
            except Exception:
                # last resort: if save expects a file path, try file inside directory
                if os.path.isdir(save_path):
                    try_path = os.path.join(save_path, "faiss_index")
                    vs.save(try_path)
                    print(f"Saved vector store to: {try_path}")
                else:
                    raise
        elif hasattr(vs, "persist"):
            vs.persist()
            print("Vector store persisted (vs.persist()).")
        else:
            print("Warning: VectorStore has no save/persist method; skipping save.")
    except Exception:
        print("Failed to save vector store:")
        traceback.print_exc()
        sys.exit(1)

    print("\nCOMPLETE")
    print(f"Total vectors: {len(chunks)}")
    print("Processing and indexing complete.\n")

if __name__ == "__main__":
    main()
