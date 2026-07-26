"""
config.py - Shared settings for every pipeline step
====================================================
All constants live here so changing one value updates every step.
"""

import importlib.util
import sys
from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
DOCUMENTS_FILE = DATA_DIR / "documents.json"      # output of step 01
CLEAN_FILE = DATA_DIR / "clean_blocks.json"       # output of step 02
CHUNKS_FILE = DATA_DIR / "chunks.json"            # output of step 03
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"     # output of step 04
CHROMA_DIR = str(DATA_DIR / "chroma_store")       # output of step 05
COLLECTION_NAME = "research_papers"

# ---------- Embedding model ----------
# multilingual-e5-small:
#   - 512 token ceiling, comfortably above our chunk size, so nothing is
#     silently truncated during encoding
#   - multilingual, so an Arabic question can retrieve English passages
#   - small enough (~470MB) to run on Streamlit Community Cloud
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# e5 models were trained to distinguish stored passages from user queries.
# Using the wrong prefix measurably lowers retrieval quality.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Lighter fallback for constrained machines (English only, 256 token ceiling).
# If you switch to it, drop CHUNK_SIZE_WORDS to about 150 and rebuild steps 3-5.
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- Chunking ----------
# 250 words is roughly 330 tokens, safely under the 512 ceiling.
CHUNK_SIZE_WORDS = 250
CHUNK_OVERLAP_WORDS = 40

# ---------- Retrieval ----------
DEFAULT_TOP_K = 5

# ---------- Corpus topic ----------
# The index is deliberately built around ONE topic. A mixed-topic index cannot
# answer comparison questions, because unrelated papers share no vocabulary.
# Narrow this to sharpen the assistant; widen it to cover more ground.

TOPIC_LABEL = "AI model architectures and training"

# Terms are OR'd against title and abstract. Quote multi-word phrases.
TOPIC_TERMS = [
    "large language model",
    "multimodal model",
    "vision-language model",
    "model architecture",
    "instruction tuning",
]

# arXiv categories to search within.
#   cs.CL = computation and language     cs.LG = machine learning
#   cs.CV = computer vision              cs.AI = artificial intelligence
ARXIV_CATEGORIES = ["cs.CL", "cs.LG", "cs.CV"]

DEFAULT_MAX_PAPERS = 25

# Rank candidates by citation count (via Semantic Scholar) before downloading,
# so the index holds the most influential papers rather than the most recent.
# Falls back to arXiv relevance order if the lookup fails.
RANK_BY_CITATIONS = True

# ---------- APIs ----------
ARXIV_API = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"


def load_step(filename: str, alias: str):
    """
    Load one of the numbered pipeline files as a module.

    Python cannot `import 06_retrieve_context` because a module name may not
    start with a digit, so we load it by path instead.
    """
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    path = BASE_DIR / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
