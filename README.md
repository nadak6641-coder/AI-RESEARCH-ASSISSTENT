# Grounded — RAG Research Assistant

A retrieval-augmented research assistant. It builds a topic-focused index of
arXiv papers and answers questions **only** from those papers, citing the
passage behind every claim.

```
documents → preprocessing → chunking → vector representation
    → vector store → context retrieval → prompting → Streamlit UI
```

---

## File structure

| File | What it does | Output |
|---|---|---|
| `01_documents.py` | Searches arXiv by topic, ranks by citations, downloads PDFs | `data/raw_pdfs/`, `documents.json` |
| `02_preprocessing.py` | Extracts and cleans text, handles two-column layouts | `data/clean_blocks.json` |
| `03_chunking.py` | Splits text into sentence-aware, page-aware chunks | `data/chunks.json` |
| `04_vector_representation.py` | Encodes chunks as embeddings | `data/embeddings.npy` |
| `05_create_chroma_store.py` | Loads vectors into ChromaDB | `data/chroma_store/` |
| `06_retrieve_context.py` | Retrieves the closest passages for a question | — |
| `07_prompting.py` | Builds the prompt, calls OpenRouter | — |
| `streamlit_app.py` | The interface | — |
| `08_evaluate.py` | *(extra)* Measures retrieval quality | — |
| `check_setup.py` | *(extra)* Diagnoses what is missing | — |
| `config.py` | Shared settings | — |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### API key, locally

```bash
export OPENROUTER_API_KEY="your_key"        # Windows: set OPENROUTER_API_KEY=your_key
export OPENROUTER_MODEL="openai/gpt-4o-mini"
```

Or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and put
the key there.

> **Never** write a real key inside a Python file, and never commit `.env` or
> `secrets.toml`. Both are protected by `.gitignore`.

---

## Running the pipeline

```bash
python 01_documents.py --max 8      # 1. fetch papers
python 02_preprocessing.py          # 2. extract and clean
python 03_chunking.py               # 3. chunk
python 04_vector_representation.py  # 4. embed
python 05_create_chroma_store.py    # 5. store

# quick checks from the terminal before opening the UI
python 06_retrieve_context.py "What problem does this paper solve?"
python 07_prompting.py "What problem does this paper solve?"

streamlit run streamlit_app.py      # 6. the interface
```

**Stop after step 2 and read `data/clean_blocks.json` yourself.** If the text
is scrambled, everything after it is built on garbage. That single check sets
the quality ceiling of the whole project.

---

## Deploying to Streamlit Cloud

1. **Push to GitHub including `data/chroma_store/`.** It is deliberately not in
   `.gitignore`, so the deployed app finds a ready index instead of trying to
   rebuild it.
2. On [share.streamlit.io](https://share.streamlit.io) create an app pointing
   at `streamlit_app.py`.
3. **Manage app → Secrets**, then paste:

```toml
OPENROUTER_API_KEY = "your_real_key"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

4. Save. The app restarts on its own.

### Memory note
The free Streamlit tier has limited RAM. If the app dies on startup, switch the
embedding model in `config.py` to `sentence-transformers/all-MiniLM-L6-v2`,
which is much lighter — and **lower `CHUNK_SIZE_WORDS` to 150**, because that
model tops out at 256 tokens. Then rebuild steps 3 through 5.

---

## When something breaks

```bash
python check_setup.py
```

It inspects every step and reports what is missing.

| Error | Cause | Fix |
|---|---|---|
| `Collection [research_papers] does not exist` | Steps 01–05 never completed, or `data/chroma_store/` was not pushed | Run the steps in order; confirm the folder is on GitHub |
| Store exists but will not open | `chromadb` version differs between build time and read time | Same pinned version locally and on Streamlit, then rebuild step 05 |
| App dies with no message | Out of memory on the free tier | Switch to `all-MiniLM-L6-v2` and `CHUNK_SIZE_WORDS = 150` |
| `No API key found` | Secrets not configured | Manage app → Secrets |

### Confirming the store really reached GitHub

```bash
git status --ignored data/chroma_store   # should not be listed as ignored
git add -f data/chroma_store
git commit -m "add vector store"
git push
```

Then open the repo on GitHub and confirm `data/chroma_store/` contains both
`chroma.sqlite3` **and** a UUID-named subfolder. Both are required.

---

## Measuring quality instead of guessing

```bash
# 1. edit eval_set.json with questions about your own papers
# 2. record the current numbers
python 08_evaluate.py
# 3. change one setting in config.py, rebuild 03-05, measure again
```

| Metric | Meaning |
|---|---|
| `Hit@k` | Share of questions whose answer appeared in the top k results |
| `MRR` | Mean reciprocal rank of the first correct result (closer to 1 is better) |

---

## The corpus is deliberately narrow

`config.py` defines one topic. Everything is built around it:

```python
TOPIC_LABEL = "AI model architectures and training"
TOPIC_TERMS = ["large language model", "multimodal model", ...]
ARXIV_CATEGORIES = ["cs.CL", "cs.LG", "cs.CV"]
```

A mixed-topic index answers one paper at a time and the rest is dead weight.
When every paper sits in the same field they share vocabulary and metrics, so
comparison questions work: *"what approaches to X exist across these papers?"*

Papers are also ranked by citation count before download, so the index holds
the most influential work rather than the most recent. Change `TOPIC_TERMS` and
rerun steps 01 through 05 to point the assistant at a different field.

---

## Design decisions worth knowing

**1. Chunk size is tied to the embedding model's ceiling.**
`multilingual-e5-small` tops out at 512 tokens. A chunk above the ceiling is
truncated **silently**, so retrieval searches a clipped passage while the LLM
receives the full one. That mismatch corrupts the system quietly. Hence
`CHUNK_SIZE_WORDS = 250` (~330 tokens), and step 04 prints an explicit warning
if any chunk exceeds the limit.

**2. A multilingual model, on purpose.**
The papers are English, but questions can be asked in Arabic. An English-only
model would return near-random results for an Arabic query.

**3. The `query:` and `passage:` prefixes.**
e5 models were trained to distinguish stored passages from user queries. Using
the wrong prefix measurably lowers accuracy. Small detail, real difference.

**4. Two-column reading order.**
Most arXiv papers are two-column. Step 02 uses `get_text("blocks")` and sorts
by coordinates: full-width blocks first, then the left column top to bottom,
then the right.

**5. Chunks are not cut at page boundaries.**
An idea spanning pages 4 and 5 stays in one chunk, and we record a page range
(`page_start` → `page_end`) so citations remain precise.

**6. The interface shows evidence before the answer.**
A RAG answer is downstream of retrieval, so the match strength of every
retrieved passage sits *above* the answer rather than hidden below it. Citation
numbers in the answer map directly onto the numbered source cards.

---

## Submission checklist

- [ ] All numbered files present, plus `requirements.txt`
- [ ] No real API key in the ZIP or on GitHub
- [ ] Streamlit secrets configured as valid TOML
- [ ] The deployed app runs
- [ ] Answers use retrieved context
- [ ] Answers cite their sources

---

## Next improvements

- **Hybrid search**: combine BM25 keyword matching with semantic search to
  catch exact terms that embeddings miss
- **Reranking**: retrieve 30 candidates, rerank with a cross-encoder, keep 5
- **Query rewriting**: reformulate the question before searching
- Measure each change with `08_evaluate.py` before and after
