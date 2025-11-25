# process_document.py
"""
Process a PDF document: extract text, tables, images (with optional OCR),
and write extracted chunks to config.CHUNKS_PATH as JSON.

Usage:
    python process_document.py --input path/to/file.pdf
    python process_document.py          # uses config.PDF_PATH by default
"""

import os
import json
import argparse
import io
import sys
from pathlib import Path

import config

# try imports (fail gracefully)
try:
    import pdfplumber
except Exception as e:
    print("ERROR: pdfplumber is required. Install with: pip install pdfplumber")
    raise

try:
    from PIL import Image
except Exception as e:
    print("ERROR: pillow is required. Install with: pip install pillow")
    raise

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# helper: safe text
def safe_text(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            return str(s)
        except Exception:
            return ""
    return s

def extract_text_chunks(page, page_number, max_chunk_chars=1500):
    """
    Extracts text from a pdfplumber page and returns a list of text chunk dicts.
    We split long page text into smaller LLM-friendly chunks.
    """
    raw = page.extract_text() or ""
    raw = safe_text(raw).strip()
    if not raw:
        return []

    chunks = []
    start = 0
    text_len = len(raw)
    while start < text_len:
        end = min(start + max_chunk_chars, text_len)
        chunk_text = raw[start:end].strip()
        if chunk_text:
            chunks.append({
                "type": "text",
                "content": chunk_text,
                "page": page_number,
                "source": f"page:{page_number}"
            })
        start = end
    return chunks

def extract_table_chunks(page, page_number):
    """
    Extract tables from a page using pdfplumber's table extraction.
    Convert each table to a simple CSV-like string representation.
    """
    table_chunks = []
    tables = page.extract_tables()
    if not tables:
        return table_chunks

    for tbl_idx, tbl in enumerate(tables, start=1):
        # Each tbl is list-of-rows (rows are lists of cells)
        # Convert to a simple CSV-like text for embedding
        try:
            lines = []
            for row in tbl:
                # some rows may be None, map to empty cells
                safe_row = [("" if c is None else str(c)).strip() for c in row]
                lines.append("| " + " | ".join(safe_row) + " |")
            table_text = "\n".join(lines).strip()
            if table_text:
                table_chunks.append({
                    "type": "table",
                    "content": table_text,
                    "page": page_number,
                    "source": f"page:{page_number}#table:{tbl_idx}"
                })
        except Exception as e:
            # skip table if something goes wrong
            print(f"Warning: failed to parse table on page {page_number}: {e}")
            continue

    return table_chunks

def extract_images_with_ocr(page, page_number, images_dir):
    """
    Extract images from page (pdfplumber) and optionally run OCR (pytesseract).
    Returns list of image chunks (with 'content' being OCR text if available, else empty).
    """
    image_chunks = []
    # pdfplumber's page.images gives metadata; to get image stream, use page.extract_image()
    images = page.images
    if not images:
        return image_chunks

    for img_idx, img_meta in enumerate(images, start=1):
        try:
            # pdfplumber provides a method to extract an image by object
            # The page.crop(...) + to_image approach can also render; pdfplumber 0.5+ has extract_image
            img_obj = page.extract_image(img_meta["object_id"]) if "object_id" in img_meta else None
            img_bytes = None
            if img_obj and "image" in img_obj:
                img_bytes = img_obj["image"]
            else:
                # fallback: render the page to an image and crop bounding box
                try:
                    bbox = (img_meta.get("x0", 0), img_meta.get("top", 0), img_meta.get("x1", 0), img_meta.get("bottom", 0))
                    # pdfplumber image coordinate has top/bottom; use to_image
                    pil = page.to_image(resolution=150).original
                    left = int(bbox[0])
                    top = int(bbox[1])
                    right = int(bbox[2]) if bbox[2] else pil.width
                    bottom = int(bbox[3]) if bbox[3] else pil.height
                    cropped = pil.crop((left, top, right, bottom))
                    bio = io.BytesIO()
                    cropped.save(bio, format="PNG")
                    img_bytes = bio.getvalue()
                except Exception:
                    img_bytes = None

            if not img_bytes:
                continue

            # Save image to disk
            img_name = f"page_{page_number}_img_{img_idx}.png"
            img_path = os.path.join(images_dir, img_name)
            with open(img_path, "wb") as imf:
                imf.write(img_bytes)

            ocr_text = ""
            if TESSERACT_AVAILABLE:
                try:
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    # Optional: can set pytesseract.pytesseract.tesseract_cmd if not on PATH
                    ocr_text = pytesseract.image_to_string(pil_img).strip()
                    if ocr_text is None:
                        ocr_text = ""
                except Exception as oe:
                    print(f"Warning: OCR failed on page {page_number} image {img_idx}: {oe}")
                    ocr_text = ""
            else:
                # mark OCR not available
                ocr_text = ""

            image_chunks.append({
                "type": "image",
                "content": ocr_text,
                "page": page_number,
                "source": f"page:{page_number}#image:{img_idx}",
                "image_path": img_path
            })
        except Exception as e:
            print(f"Warning: failed to extract image on page {page_number}: {e}")
            continue

    return image_chunks

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Process a PDF and extract text/tables/images.")
    parser.add_argument("--input", "-i", type=str, help="Path to PDF to process", default=None)
    args = parser.parse_args()

    # create directories
    config.create_directories()
    print("\nDirectory structure:")
    print(f"  Raw data: {config.RAW_DATA_DIR}")
    print(f"  Processed data: {config.PROCESSED_DATA_DIR}")
    print(f"  Vector store: {config.VECTOR_STORE_DIR}")
    print(f"  Images: {config.IMAGES_DIR}")
    print()

    # If input provided, update config.PDF_PATH
    if args.input:
        pdf_input = os.path.abspath(args.input)
        if not os.path.exists(pdf_input):
            print(f"ERROR: Provided input path does not exist: {pdf_input}")
            return
        config.set_pdf_path(pdf_input)
        print(f"Using provided PDF: {pdf_input}")
    else:
        print(f"Using configured PDF: {config.PDF_PATH}")

    pdf_path = config.PDF_PATH
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at {pdf_path}")
        return

    print("\n======================================================================")
    print("STEP 1: Document Processing")
    print("======================================================================\n")

    chunks = []
    total_text_chunks = 0
    total_table_chunks = 0
    total_image_chunks = 0

    # ensure images dir exists
    images_dir = config.IMAGES_DIR
    os.makedirs(images_dir, exist_ok=True)

    print("\n Found PDF")
    print(f"Processing document: {pdf_path}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                # Extract text into chunks
                text_chunks = extract_text_chunks(page, page_number=i)
                for t in text_chunks:
                    chunks.append(t)
                total_text_chunks += len(text_chunks)

                # Extract tables
                table_chunks = extract_table_chunks(page, page_number=i)
                for t in table_chunks:
                    chunks.append(t)
                total_table_chunks += len(table_chunks)

                # Extract images and OCR
                image_chunks = extract_images_with_ocr(page, page_number=i, images_dir=images_dir)
                for im in image_chunks:
                    chunks.append(im)
                total_image_chunks += len(image_chunks)

        total_chunks = len(chunks)
        print(f"Extracted {total_text_chunks} text chunks")
        print(f"Extracted {total_table_chunks} tables")
        if TESSERACT_AVAILABLE:
            print(f"Extracted {total_image_chunks} images with OCR")
        else:
            if total_image_chunks > 0:
                print(f"Extracted {total_image_chunks} images (OCR skipped - tesseract not available)")
            else:
                print("Extracted 0 images")

        print(f" Total chunks: {total_chunks}\n")
    except Exception as e:
        print(f"ERROR: Failed to process PDF: {e}")
        raise

    # Save chunks to JSON (utf-8)
    try:
        os.makedirs(os.path.dirname(config.CHUNKS_PATH), exist_ok=True)
        with open(config.CHUNKS_PATH, "w", encoding="utf-8") as fout:
            json.dump(chunks, fout, ensure_ascii=False, indent=2)
        print(f"Extracted {len(chunks)} chunks")
        print(f"  - Text chunks: {total_text_chunks}")
        print(f"  - Tables: {total_table_chunks}")
        print(f"  - Images (OCR): {total_image_chunks}")
        print("\nSaving data")
    except Exception as e:
        print(f"ERROR: failed to save extracted chunks: {e}")
        raise

    print("\nSTEP 1 completed successfully.\n")

if __name__ == "__main__":
    main()
