# config.py
import os
import tempfile
from pathlib import Path

# Base directory where the app is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Target data directory inside the project
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

# Default PDF path (gets overwritten by Streamlit app)
PDF_PATH = os.path.join(RAW_DATA_DIR, "default.pdf")


def set_pdf_path(path):
    """
    Update the path of the PDF to be processed.
    """
    global PDF_PATH
    PDF_PATH = path


def create_directories():
    """
    Create required directories. If the project directory is not writable,
    switch to a fallback temporary directory.
    """
    global DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_STORE_DIR, IMAGES_DIR

    desired_dirs = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_STORE_DIR, IMAGES_DIR]

    try:
        for d in desired_dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        print("All directories created")
    except Exception as e:
        print(f"Warning: Cannot create directories in repo. Using fallback. Error: {e}")

        # Fallback to temporary directory on Streamlit Cloud
        fallback_root = os.path.join(tempfile.gettempdir(), "multimodal_rag_data")

        DATA_DIR = fallback_root
        RAW_DATA_DIR = os.path.join(fallback_root, "raw")
        PROCESSED_DATA_DIR = os.path.join(fallback_root, "processed")
        VECTOR_STORE_DIR = os.path.join(fallback_root, "vector_store")
        IMAGES_DIR = os.path.join(fallback_root, "images")

        # Create fallback dirs
        Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
        Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)

        print("Fallback directories created successfully")
