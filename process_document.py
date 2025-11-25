import json
import os
import sys
import pathlib
import pdfplumber
from document_processor import process_pdf

import config

# Get input PDF from cmd args
def get_input_pdf():
    if len(sys.argv) >= 3 and sys.argv[1] == "--input":
        return sys.argv[2]
    return None

def main():
    pdf_path = get_input_pdf()

    if not pdf_path:
        print("ERROR: No input PDF provided. Run with: python process_document.py --input file.pdf")
        return

    if not os.path.exists(pdf_path):
        print(f"ERROR: File does not exist: {pdf_path}")
        return

    # ensure directories
    pathlib.Path(config.PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

    print("Processing:", pdf_path)
    chunks = process_pdf(pdf_path)

    chunks_path = config.CHUNKS_PATH
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("Saved chunks to", chunks_path)

if __name__ == "__main__":
    main()
