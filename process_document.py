import os
import json
import pdfplumber
from PyPDF2 import PdfReader

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
IMAGES_DIR = "data/images"

CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.json")


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    print("All directories created")


def extract_text_pdfplumber(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print("pdfplumber failed:", e)
        return None


def extract_text_pypdf2(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print("PyPDF2 failed:", e)
        return None


def chunk_text(text, max_len=800):
    words = text.split()
    chunks, current = [], []

    for w in words:
        current.append(w)
        if len(current) >= max_len:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def process_document(selected_file_path):
    ensure_dirs()

    if not selected_file_path:
        print("ERROR: No file selected")
        return False

    if not os.path.exists(selected_file_path):
        print("ERROR: File not found:", selected_file_path)
        return False

    print("Found PDF")
    print("Processing document:", selected_file_path)

    # --- Try pdfplumber
    text = extract_text_pdfplumber(selected_file_path)

    # --- Fallback
    if not text:
        text = extract_text_pypdf2(selected_file_path)

    if not text:
        print("ERROR: Neither pdfplumber nor PyPDF2 available.")
        return False

    # --- Chunk text
    chunks = chunk_text(text)

    # Save chunks
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved chunks to {CHUNKS_PATH}")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python process_document.py <pdf_path>")
        exit()

    process_document(sys.argv[1])
