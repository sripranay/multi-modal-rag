# config.py (robust create_directories)
import os
import tempfile
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
VECTOR_STORE_DIR = os.path.join(DATA_DIR, 'vector_store')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

def create_directories():
    desired_dirs = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_STORE_DIR, IMAGES_DIR]
    for d in desired_dirs:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except NotADirectoryError:
            # if a path component is a file, fallback to temp dir
            fallback_root = os.path.join(tempfile.gettempdir(), "multimodal_rag_data")
            print(f"Warning: path {d} not creatable; switching to fallback {fallback_root}")
            # rewrite all paths to a fallback directory and create them
            global DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_STORE_DIR, IMAGES_DIR
            DATA_DIR = fallback_root
            RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
            PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
            VECTOR_STORE_DIR = os.path.join(DATA_DIR, 'vector_store')
            IMAGES_DIR = os.path.join(DATA_DIR, 'images')
            Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
            Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
            Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
            Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
            break
    print("All directories created (or fallback used)")
