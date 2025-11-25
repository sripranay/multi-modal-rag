"""
process_document.py — Cloud-safe version
Extracts:
  • text (always – pdfplumber or PyPDF2 fallback)
  • tables (if pdfplumber)
  • images (if pdfplumber)
  • OCR on images (optional)
Outputs chunks → saved to processed/extracted_chunks.json
"""

import os
import json
import sys
import pathlib
import config

# -----------------------------
# IMPORTS WITH SAFE FALLBACKS
# -----------------------------

# pdfplumber (best quality)
try:
    import pdfplumber
    _PDFPLUMBER_OK = True
except Exception:
    _PDFPLUMBER_OK = False

# PyPDF2 fallback (always available + light)
try:
    import PyPDF2
    _PYPDF2_OK = True
except Exception:
    _PYPDF2_OK = False

# OCR optional
try:
    import pytesseract
    from PIL import Image
    _OCR_OK = True
except Exception:
    _OCR_OK = False


def extract_with_pdfplumber(pdf_path):
    """Extract text, tables, images using pdfplumber (best)."""
    chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            # TEXT
            text = page.extract_text() or ""
            if text.strip():
                chunks.append({
                    "page": page_num,
                    "type": "text",
                    "content": text,
                    "source": pdf_path
                })

            # TABLES
            try:
                tables = page.extract_tables()
                for t in tables:
                    if t:
                        table_str = "\n".join([", ".join(row) for row in t])
                        chunks.append({
                            "page": page_num,
                            "type": "table",
                            "content": table_str,
                            "source": pdf_path
                        })
            except:
                pass

            # IMAGES + OCR
            try:
                for img in page.images:
                    bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                    im = page.crop(bbox).to_image(resolution=200)
                    pil_image = Image.frombytes("RGB", im.original.size, im.original.tobytes())

                    ocr_text = ""
                    if _OCR_OK:
                        try:
                            ocr_text = pytesseract.image_to_string(pil_image)
                        except:
                            pass

                    chunks.append({
                        "page": page_num,
                        "type": "image",
                        "content": ocr_text if ocr_text.strip() else "[image extracted]",
                        "source": pdf_path
                    })
            except:
                pass

    return chunks


def extract_with_pypdf2(pdf_path):
    """Fallback extractor: text-only, no tables/images."""
    chunks = []
    reader = PyPDF2.PdfReader(pdf_path)

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except:
            text = ""

        chunks.append({
            "page": page_num,
            "type": "text",
            "content": text,
            "source": pdf_path
        })

    return chunks


# --------------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------------

def main():
    # command-line override: python process_document.py --input path.pdf
    input_path = None
    if len(sys.argv) > 2 and sys.argv[1] == "--input":
        input_path = sys.argv[2]

    # else use config default
    if not input_path:
        input_path = config.PDF_PATH

    if not os.path.exists(input_path):
        print(f"\nERROR: PDF not found: {input_path}")
        return

    print("\n======================================================================")
    print("STEP 1: Document Processing")
    print("======================================================================\n")

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    print(" Found PDF")
    print(f"Processing document: {input_path}")

    # -----------------------------------
    # Choose extraction method
    # -----------------------------------
    chunks = []

    if _PDFPLUMBER_OK:
        print(" Using pdfplumber for full extraction (text/tables/images)")
        chunks = extract_with_pdfplumber(input_path)

    elif _PYPDF2_OK:
        print(" pdfplumber missing → Using PyPDF2 fallback (text only)")
        chunks = extract_with_pypdf2(input_path)

    else:
        print("ERROR: Neither pdfplumber nor PyPDF2 available.")
        return

    print(f"Extracted {len(chunks)} chunks")

    # -----------------------------------
    # Save chunks
    # -----------------------------------
    save_path = config.CHUNKS_PATH
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved chunks to: {save_path}")
    print("\nProcessing complete.\n")


if __name__ == "__main__":
    main()
