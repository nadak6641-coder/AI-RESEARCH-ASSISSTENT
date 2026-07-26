"""
04_vector_representation.py - Turn text into vectors
====================================================
Each chunk becomes a vector that represents its meaning. Passages with
similar meaning end up close together, which is what makes semantic search
possible.

Two details that matter here:

1) The "passage: " prefix. e5 models were trained to tell stored passages
   apart from user queries. The wrong prefix measurably lowers accuracy.

2) The truncation check. We count how many chunks exceed the model's token
   ceiling and warn loudly. Without this check, oversized chunks are clipped
   silently and retrieval quietly degrades.

Output: data/embeddings.npy

Run:
    python 04_vector_representation.py
"""

import json
import sys

import numpy as np

import config


def main():
    if not config.CHUNKS_FILE.exists():
        print("Missing data/chunks.json - run 03_chunking.py first.")
        sys.exit(1)

    from sentence_transformers import SentenceTransformer

    chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"{len(chunks)} chunk(s) to encode")
    print(f"Loading model: {config.EMBEDDING_MODEL}")
    print("(first run downloads it, which takes a moment)")

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    max_tokens = model.max_seq_length
    print(f"Model token ceiling: {max_tokens}")

    texts = [config.PASSAGE_PREFIX + c["text"] for c in chunks]

    tokenizer = model.tokenizer
    lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts]
    over_limit = sum(1 for n in lengths if n > max_tokens)

    print(f"Longest chunk: {max(lengths)} tokens")
    if over_limit:
        print(f"WARNING: {over_limit} chunk(s) exceed the ceiling and will be truncated.")
        print("Lower CHUNK_SIZE_WORDS in config.py and rerun 03_chunking.py.")
    else:
        print("All chunks are under the ceiling, nothing will be truncated.")

    print("\nEncoding...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # required for clean cosine comparison
    )

    np.save(config.EMBEDDINGS_FILE, embeddings)

    print(f"\nShape (chunks x dimensions): {embeddings.shape}")
    print(f"Saved to: {config.EMBEDDINGS_FILE}")
    print("\nNext step: python 05_create_chroma_store.py")


if __name__ == "__main__":
    main()
