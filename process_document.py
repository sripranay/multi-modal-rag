#!/usr/bin/env python3
"""
process_document.py

Usage:
  python process_document.py [--input <pdf_path>]

This script extracts text, tables (if pdfplumber available) and saves an
array of "chunks" to config.CHUNKS_PATH as UTF-8 JSON.

Chunk format:
  {
    "type": "text"|"table"|"image",
    "content": "....",
    "page": <page_number>,
    "source": "<filename>"
  }
"""
import json
import os
import sys
import argparse
from pathlib import Path

import config

# Try multiple PDF backends
HAS_PDFPLUMBER = False
HAS_PYPDF2 = False
HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    pass

try:
    import PyPDF2
    HAS_PYPDF2 = True
except Exception:
    pass

try:
    import fitz  # pymupdf
    HAS_PYMUPDF = True
except Exception:
    pass

def ensure_dirs():
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    print("All directories created")

def extract_with_pdfplumber(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                chunks.append({"type":"text","content":text,"page":i,"source":os.path.basename(pdf_path)})
            # tables
            try:
                tables = page.extract_tables()
                if tables:
                    for t in tables:
                        # join rows with pipes for simple representation
                        rows = ["\t".join([cell if cell is not None else "" for cell in row]) for row in t]
                        table_text = "\n".join(rows)
                        chunks.append({"type":"table","content":table_text,"page":i,"source":os.path.basename(pdf_path)})
            except Exception:
                # ignore table extraction errors
                pass
    return chunks

def extract_with_pypdf2(pdf_path):
    chunks = []
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                chunks.append({"type":"text","content":text.strip(),"page":i,"source":os.path.basename(pdf_path)})
    except Exception as e:
        print("PyPDF2 extraction failed:", e)
    return chunks

def extract_with_pymupdf(pdf_path):
    chunks = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc, start=1):
            try:
                text = page.get_text("text") or ""
            except Exception:
                text = ""
            if text.strip():
                chunks.append({"type":"text","content":text.strip(),"page":i,"source":os.path.basename(pdf_path)})
        doc.close()
    except Exception as e:
        print("pymupdf extraction failed:", e)
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Path to PDF to process", default=None)
    args = parser.parse_args()

    ensure_dirs()

    # Determine input PDF
    pdf_path = args.input or config.PDF_PATH
    if not pdf_path:
        print("No PDF provided and config.PDF_PATH is empty. Exiting.")
        sys.exit(1)

    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    print("Found PDF")
    print(f"Processing document: {pdf_path}")

    chunks = []

    # Prefer pdfplumber for tables; fallback to other readers for text
    if HAS_PDFPLUMBER:
        try:
            chunks = extract_with_pdfplumber(pdf_path)
            print(f"Extracted {len([c for c in chunks if c['type']=='text'])} text chunks (pdfplumber)")
            print(f"Extracted {len([c for c in chunks if c['type']=='table'])} tables (pdfplumber)")
        except Exception as e:
            print("pdfplumber extraction failed:", e)
            chunks = []
    else:
        print("pdfplumber not available.")

    if not chunks:
        # try PyPDF2
        if HAS_PYPDF2:
            chunks = extract_with_pypdf2(pdf_path)
            print(f"Extracted {len(chunks)} text chunks (PyPDF2)")
        elif HAS_PYMUPDF:
            chunks = extract_with_pymupdf(pdf_path)
            print(f"Extracted {len(chunks)} text chunks (pymupdf)")
        else:
            print("ERROR: Neither pdfplumber nor PyPDF2 nor pymupdf available.")
            # still write empty chunks file so app doesn't crash
            chunks = []

    # Save chunks JSON
    try:
        with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(chunks)} chunks to {config.CHUNKS_PATH}")
    except Exception as e:
        print("Failed to save chunks:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
