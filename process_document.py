# process_document.py
import os
import json
from typing import List, Optional

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
IMAGES_DIR = "data/images"
CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.json")


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    print("All directories created")


def extract_text_pypdf2(pdf_path: str) -> Optional[str]:
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        print("PyPDF2 import failed:", e)
        return None

    try:
        reader = PdfReader(pdf_path)
        texts = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
        return "\n".join(texts)
    except Exception as e:
        print("PyPDF2 extraction failed:", e)
        return None


def extract_text_pdfminer(pdf_path: str) -> Optional[str]:
    try:
        # pdfminer.six based extraction (fallback)
        from io import StringIO
        from pdfminer.high_level import extract_text_to_fp
    except Exception as e:
        print("pdfminer import failed:", e)
        return None

    try:
        output_string = StringIO()
        with open(pdf_path, "rb") as fh:
            extract_text_to_fp(fh, output_string)
        return output_string.getvalue()
    except Exception as e:
        print("pdfminer extraction failed:", e)
        return None


def chunk_text(text: str, max_tokens: int = 800) -> List[str]:
    # simple chunk by words; adjust max_tokens as needed
    words = text.split()
    chunks = []
    cur = []
    for w in words:
        cur.append(w)
        if len(cur) >= max_tokens:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def process_document(pdf_path: str) -> bool:
    ensure_dirs()

    if not pdf_path:
        print("ERROR: No file path provided.")
        return False

    if not os.path.exists(pdf_path):
        print("ERROR: File not found:", pdf_path)
        return False

    print("Found PDF")
    print("Processing document:", pdf_path)

    # Try PyPDF2 first (pure python)
    text = extract_text_pypdf2(pdf_path)
    if text and text.strip():
        print("Extracted text using PyPDF2")
    else:
        print("PyPDF2 extraction empty or failed, trying pdfminer...")
        text = extract_text_pdfminer(pdf_path)

    if not text or not text.strip():
        print("ERROR: Could not extract text with PyPDF2 or pdfminer.")
        return False

    chunks = chunk_text(text)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python process_document.py <pdf_path>")
        sys.exit(1)

    target = sys.argv[1]
    ok = process_document(target)
    if not ok:
        sys.exit(2)
