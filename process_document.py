# process_document.py
import os
import json
import argparse
from pathlib import Path

# adjust to your config paths OR import config if present
try:
    import config
    RAW_DIR = config.RAW_DATA_DIR
    PROCESSED_DIR = config.PROCESSED_DIR
    CHUNKS_PATH = config.CHUNKS_PATH
except Exception:
    RAW_DIR = os.path.join("data", "raw")
    PROCESSED_DIR = os.path.join("data", "processed")
    CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.json")

Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

def extract_with_pdfplumber(pdf_path):
    import pdfplumber
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                chunks.append({"type":"text", "page": i, "content": text, "source": os.path.basename(pdf_path)})
            # extract tables
            try:
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables, start=1):
                    # Convert table (list of rows) to simple text representation
                    table_text = "\n".join(["\t".join([cell if cell is not None else "" for cell in row]) for row in table])
                    chunks.append({"type":"table", "page": i, "content": table_text, "source": os.path.basename(pdf_path)})
            except Exception:
                pass
    return chunks

def extract_with_pypdf2(pdf_path):
    from PyPDF2 import PdfReader
    chunks = []
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            chunks.append({"type":"text", "page": i, "content": text, "source": os.path.basename(pdf_path)})
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="input PDF path (optional)", default=None)
    args = parser.parse_args()

    if args.input:
        pdf_path = args.input
    else:
        # choose first PDF in RAW_DIR if exists
        files = [p for p in os.listdir(RAW_DIR) if p.lower().endswith(".pdf")]
        if not files:
            print(f"PDF not found: {os.path.join(RAW_DIR, 'default.pdf')}")
            return
        pdf_path = os.path.join(RAW_DIR, files[0])

    print("Found PDF")
    print("Processing document:", pdf_path)

    chunks = []
    # try pdfplumber first
    try:
        import pdfplumber  # noqa: F401
        chunks = extract_with_pdfplumber(pdf_path)
    except Exception as e:
        print("pdfplumber not available or failed:", str(e))
        # try pypdf2
        try:
            import PyPDF2  # noqa: F401
            chunks = extract_with_pypdf2(pdf_path)
        except Exception as e2:
            print("ERROR: Neither pdfplumber nor PyPDF2 available.")
            return

    # If no chunks found, warn
    if not chunks:
        print("No chunks extracted from PDF.")
    else:
        print(f"Extracted {len(chunks)} chunks")
        # save chunks
        Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print("Saved chunks to", CHUNKS_PATH)

if __name__ == "__main__":
    main()
