"""
06_retrieve_context.py - Retrieve supporting context
====================================================
A question is embedded with the same model used for the chunks, then matched
against ChromaDB to find the closest passages.

Differences from step 04:
- the prefix is "query: " rather than "passage: ", because e5 distinguishes them
- we return a similarity score per result so answer quality can be judged

This file is imported by 07_prompting.py and streamlit_app.py, and can also be
run directly for a quick check.

Run:
    python 06_retrieve_context.py "What is retrieval augmented generation?"
"""

import sys
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once; loading is the slow part."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    """Open the Chroma collection once."""
    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_collection(config.COLLECTION_NAME)


def retrieve(question: str, top_k: int = config.DEFAULT_TOP_K) -> list[dict]:
    """
    Return the top_k closest chunks for a question.

    Each result: {rank, text, paper_title, source_file, page_start,
    page_end, score}. Score runs 0 to 1; higher means closer.
    """
    model = get_model()
    collection = get_collection()

    query_embedding = model.encode(
        [config.QUERY_PREFIX + question],
        normalize_embeddings=True,
    ).tolist()

    raw = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for i, (doc, meta, distance) in enumerate(zip(
        raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ), start=1):
        results.append({
            "rank": i,
            "text": doc,
            "paper_title": meta.get("paper_title", "Unknown"),
            "source_file": meta.get("source_file", ""),
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "score": round(1 - distance, 4),  # cosine distance -> similarity
        })
    return results


def format_pages(result: dict) -> str:
    """Format a page range as 'p. 4' or 'pp. 4-5'."""
    start, end = result.get("page_start"), result.get("page_end")
    if start is None:
        return ""
    return f"p. {start}" if start == end else f"pp. {start}-{end}"


def list_indexed_papers() -> list[str]:
    """Unique paper titles currently in the index (used by the UI)."""
    collection = get_collection()
    raw = collection.get(include=["metadatas"])
    titles = {m.get("paper_title", "") for m in raw.get("metadatas", []) if m}
    return sorted(t for t in titles if t)


def main():
    if len(sys.argv) < 2:
        print('Usage: python 06_retrieve_context.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}\n")

    for result in retrieve(question):
        print(f"[{result['rank']}] score {result['score']:.3f} | "
              f"{result['paper_title'][:55]} ({format_pages(result)})")
        print(f"    {result['text'][:180]}...\n")


if __name__ == "__main__":
    main()
