"""
03_chunking.py - Split text into retrievable chunks
===================================================
How this differs from naive fixed-size chunking:

1) It does not cut at page boundaries. An idea that starts at the bottom of
   page 4 and finishes at the top of page 5 stays in one chunk. We record a
   page range instead of a single page, so citations stay accurate.

2) It cuts at sentence boundaries. A chunk ending mid-sentence produces a
   noisy embedding.

3) Chunk size respects the embedding model's ceiling. 250 words is about
   330 tokens; multilingual-e5-small tops out at 512. If a chunk exceeds the
   ceiling the model truncates it silently, meaning retrieval searches a
   clipped passage while the LLM receives the full one. That mismatch quietly
   corrupts the whole system.

Output: data/chunks.json

Run:
    python 03_chunking.py
"""

import json
import re
import sys

import config

# Split on . ! ? followed by whitespace and a capital, digit, or Arabic letter
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\u0600-\u06FF])")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def build_chunks(blocks: list[dict]) -> list[dict]:
    """
    Walk each paper's blocks in order, accumulate sentences until the target
    size is reached, and carry a small overlap forward so context is not cut.
    """
    chunks = []
    counter = 0

    # Group by paper so two papers never share a chunk
    by_paper: dict[str, list[dict]] = {}
    for block in blocks:
        by_paper.setdefault(block["source_file"], []).append(block)

    for source_file, paper_blocks in by_paper.items():
        title = paper_blocks[0]["paper_title"]

        # Flatten to sentences, each tagged with its page
        sentences: list[tuple[str, int]] = []
        for block in paper_blocks:
            for sentence in split_sentences(block["text"]):
                sentences.append((sentence, block["page"]))

        buffer: list[tuple[str, int]] = []
        buffer_words = 0

        def flush():
            """Turn the buffer into a chunk and return the overlap tail."""
            nonlocal counter
            if not buffer:
                return []

            text = " ".join(s for s, _ in buffer)
            pages = [p for _, p in buffer]
            counter += 1
            chunks.append({
                "chunk_id": f"chunk_{counter:05d}",
                "text": text,
                "source_file": source_file,
                "paper_title": title,
                "page_start": min(pages),
                "page_end": max(pages),
                "word_count": len(text.split()),
            })

            overlap, words = [], 0
            for sentence, page in reversed(buffer):
                sentence_words = len(sentence.split())
                if words + sentence_words > config.CHUNK_OVERLAP_WORDS:
                    break
                overlap.insert(0, (sentence, page))
                words += sentence_words
            return overlap

        for sentence, page in sentences:
            sentence_words = len(sentence.split())

            # A single sentence longer than the limit gets hard-split
            if sentence_words > config.CHUNK_SIZE_WORDS:
                buffer = flush()
                buffer_words = sum(len(s.split()) for s, _ in buffer)
                words = sentence.split()
                for i in range(0, len(words), config.CHUNK_SIZE_WORDS):
                    piece = " ".join(words[i:i + config.CHUNK_SIZE_WORDS])
                    counter += 1
                    chunks.append({
                        "chunk_id": f"chunk_{counter:05d}",
                        "text": piece,
                        "source_file": source_file,
                        "paper_title": title,
                        "page_start": page,
                        "page_end": page,
                        "word_count": len(piece.split()),
                    })
                buffer, buffer_words = [], 0
                continue

            if buffer_words + sentence_words > config.CHUNK_SIZE_WORDS:
                buffer = flush()
                buffer_words = sum(len(s.split()) for s, _ in buffer)

            buffer.append((sentence, page))
            buffer_words += sentence_words

        flush()  # final chunk of this paper

    return chunks


def main():
    if not config.CLEAN_FILE.exists():
        print("Missing data/clean_blocks.json - run 02_preprocessing.py first.")
        sys.exit(1)

    blocks = json.loads(config.CLEAN_FILE.read_text(encoding="utf-8"))
    chunks = build_chunks(blocks)

    config.CHUNKS_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sizes = [c["word_count"] for c in chunks]
    print(f"Created {len(chunks)} chunk(s)")
    if sizes:
        print(f"Words per chunk - min {min(sizes)}, mean {sum(sizes)//len(sizes)}, max {max(sizes)}")
    print(f"Saved to: {config.CHUNKS_FILE}")
    print("\nNext step: python 04_vector_representation.py")


if __name__ == "__main__":
    main()
