"""
02_preprocessing.py - Extract and clean text from the PDFs
==========================================================
This is the step people underestimate most, and it sets the quality ceiling
for everything downstream.

The problem it solves:
Most arXiv papers are typeset in two columns. With plain get_text("text"),
PyMuPDF can interleave lines from the left and right columns, producing
scrambled prose. Every later step would then be built on garbage.

The fix: get_text("blocks") returns text blocks with their coordinates
(x0, y0, x1, y1), so we sort them ourselves into real reading order.

We also clean:
- stop at the References section (it adds no answers and pollutes the index)
- drop very short blocks (page numbers, running headers, footers)
- drop blocks that are mostly digits (raw result tables)
- rejoin words hyphenated across a line break

Output: data/clean_blocks.json - clean text blocks, each tagged with its page.

Run:
    python 02_preprocessing.py
"""

import json
import re
import sys

import fitz  # PyMuPDF

import config

MIN_BLOCK_CHARS = 60       # shorter than this is usually a header or page number
MAX_DIGIT_RATIO = 0.35     # more digits than this is usually a raw table
COLUMN_TOLERANCE = 20      # pixel slack when deciding column boundaries
FULL_WIDTH_RATIO = 0.65    # wider than this spans the page (title, abstract)

REFERENCES_PATTERN = re.compile(
    r"^\s*(references|bibliography|acknowledg(e)?ments)\s*$",
    re.IGNORECASE,
)


def sort_blocks_reading_order(page) -> list[str]:
    """
    Return block texts in correct reading order, handling two-column pages.

    Each PyMuPDF block is (x0, y0, x1, y1, text, block_no, block_type),
    where block_type 0 means text and 1 means image.

    Sorting happens in three regions:
      region 0 = full-width blocks above the columns (title, abstract)
      region 1 = the columns (left column top to bottom, then right)
      region 2 = full-width blocks below the columns (wide tables, footers)
    """
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if not blocks:
        return []

    page_width = page.rect.width
    mid_x = page_width / 2

    left_only = [b for b in blocks if b[2] <= mid_x + COLUMN_TOLERANCE]
    right_only = [b for b in blocks if b[0] >= mid_x - COLUMN_TOLERANCE]
    is_two_column = len(left_only) >= 2 and len(right_only) >= 2

    if not is_two_column:
        return [b[4] for b in sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))]

    def is_full_width(b):
        return (b[2] - b[0]) > FULL_WIDTH_RATIO * page_width

    column_blocks = [b for b in blocks if not is_full_width(b)]
    if not column_blocks:
        return [b[4] for b in sorted(blocks, key=lambda b: round(b[1], 1))]

    columns_top = min(b[1] for b in column_blocks)
    columns_bottom = max(b[3] for b in column_blocks)

    def key(b):
        if is_full_width(b):
            if b[3] <= columns_top + COLUMN_TOLERANCE:
                return (0, round(b[1], 1), 0)
            if b[1] >= columns_bottom - COLUMN_TOLERANCE:
                return (2, round(b[1], 1), 0)
        center_x = (b[0] + b[2]) / 2
        column = 0 if center_x < mid_x else 1
        return (1, column, round(b[1], 1))

    return [b[4] for b in sorted(blocks, key=key)]


def clean_text(text: str) -> str:
    """Clean a single block of text."""
    # rejoin hyphenated line breaks: "represen-\ntation" -> "representation"
    text = re.sub(r"-\n\s*", "", text)
    # remaining newlines inside a block become spaces
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def is_noise(text: str) -> bool:
    """Is this block junk (header, page number, numeric table)?"""
    if len(text) < MIN_BLOCK_CHARS:
        return True
    digits = sum(c.isdigit() for c in text)
    if digits / max(len(text), 1) > MAX_DIGIT_RATIO:
        return True
    if text.count(" ") < 6:  # barely a sentence
        return True
    return False


def process_pdf(pdf_path, title: str) -> list[dict]:
    """Extract clean text blocks from a single paper."""
    doc = fitz.open(pdf_path)
    clean_blocks = []
    hit_references = False

    for page_number, page in enumerate(doc, start=1):
        if hit_references:
            break

        for raw in sort_blocks_reading_order(page):
            text = clean_text(raw)

            if REFERENCES_PATTERN.match(text[:40]):
                hit_references = True
                break

            if is_noise(text):
                continue

            clean_blocks.append({
                "source_file": pdf_path.name,
                "paper_title": title,
                "page": page_number,
                "text": text,
            })

    doc.close()
    return clean_blocks


def main():
    if not config.DOCUMENTS_FILE.exists():
        print("Missing data/documents.json - run 01_documents.py first.")
        sys.exit(1)

    documents = json.loads(config.DOCUMENTS_FILE.read_text(encoding="utf-8"))
    if not documents:
        print("data/documents.json is empty, so step 01 downloaded nothing.")
        print("Run 01_documents.py again and confirm it prints 'Downloaded N PDF(s)'.")
        sys.exit(1)

    titles = {d["filename"]: d["title"] for d in documents}

    all_blocks = []
    for doc_meta in documents:
        pdf_path = config.RAW_PDF_DIR / doc_meta["filename"]
        if not pdf_path.exists():
            print(f"Skipping missing file: {doc_meta['filename']}")
            continue

        title = titles.get(pdf_path.name, pdf_path.stem)
        blocks = process_pdf(pdf_path, title)
        all_blocks.extend(blocks)
        print(f"{len(blocks):>5} clean blocks | {title[:55]}")

    config.CLEAN_FILE.write_text(
        json.dumps(all_blocks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nTotal blocks: {len(all_blocks)}")
    print(f"Saved to: {config.CLEAN_FILE}")
    print("\nOpen that file and read a few blocks yourself before continuing.")
    print("If the text is scrambled, everything after this is built on garbage.")
    print("\nNext step: python 03_chunking.py")


if __name__ == "__main__":
    main()
