"""
check_setup.py - Diagnose the project
=====================================
Checks every pipeline step and tells you exactly what is missing and how to
fix it. Run this first whenever the app throws an error.

Run:
    python check_setup.py
"""

import json
import sys
from pathlib import Path

import config

OK = "[ok]  "
BAD = "[FAIL]"
WARN = "[warn]"

problems = []


def line(status, label, detail=""):
    print(f"{status} {label}" + (f"\n         {detail}" if detail else ""))


def human_size(path: Path) -> str:
    size = path.stat().st_size if path.is_file() else sum(
        f.stat().st_size for f in path.rglob("*") if f.is_file()
    )
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


print("=" * 64)
print("  RAG project diagnostics")
print("=" * 64)
print(f"\nProject folder: {config.BASE_DIR}")
print(f"Data folder:    {config.DATA_DIR}\n")

# ---------- 1. Installed packages ----------
print("--- Packages ---")
try:
    import chromadb
    line(OK, f"chromadb {chromadb.__version__}")
    chroma_version = chromadb.__version__
except ImportError:
    line(BAD, "chromadb is not installed", "pip install -r requirements.txt")
    problems.append("chromadb not installed")
    chroma_version = None

try:
    import sentence_transformers
    line(OK, f"sentence-transformers {sentence_transformers.__version__}")
except ImportError:
    line(BAD, "sentence-transformers is not installed", "pip install -r requirements.txt")
    problems.append("sentence-transformers not installed")

# ---------- 2. Step outputs ----------
print("\n--- Pipeline outputs ---")

steps = [
    ("01", "PDFs", config.RAW_PDF_DIR, "python 01_documents.py"),
    ("01", "documents.json", config.DOCUMENTS_FILE, "python 01_documents.py"),
    ("02", "clean_blocks.json", config.CLEAN_FILE, "python 02_preprocessing.py"),
    ("03", "chunks.json", config.CHUNKS_FILE, "python 03_chunking.py"),
    ("04", "embeddings.npy", config.EMBEDDINGS_FILE, "python 04_vector_representation.py"),
    ("05", "chroma_store/", Path(config.CHROMA_DIR), "python 05_create_chroma_store.py"),
]

first_missing = None
for step, label, path, fix in steps:
    if path.exists():
        extra = ""
        if path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                extra = f"{len(data)} items, "
                if len(data) == 0:
                    problems.append(f"{label} is empty")
            except Exception:
                extra = "unreadable! "
        line(OK, f"[{step}] {label}", f"{extra}{human_size(path)}")
    else:
        line(BAD, f"[{step}] {label} is missing", f"Fix: {fix}")
        if first_missing is None:
            first_missing = fix
        problems.append(f"{label} missing")

# ---------- 3. Chroma contents ----------
print("\n--- Chroma store ---")
chroma_path = Path(config.CHROMA_DIR)

if not chroma_path.exists():
    line(BAD, "The folder does not exist")
elif chroma_version:
    sqlite = chroma_path / "chroma.sqlite3"
    if sqlite.exists():
        line(OK, "chroma.sqlite3 present", human_size(sqlite))
    else:
        line(BAD, "chroma.sqlite3 missing", "Folder exists but is empty - rerun step 05")
        problems.append("chroma.sqlite3 missing")

    index_dirs = [d for d in chroma_path.iterdir() if d.is_dir()]
    if index_dirs:
        line(OK, f"Index folders: {len(index_dirs)}", ", ".join(d.name[:12] for d in index_dirs))
    else:
        line(WARN, "No index folders",
             "If you pushed to GitHub, confirm they were not filtered out")

    try:
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        names = [c if isinstance(c, str) else c.name for c in client.list_collections()]

        if not names:
            line(BAD, "No collections in the store", "Store exists but is empty - rerun step 05")
            problems.append("store is empty")
        else:
            line(OK, f"Collections found: {names}")

            if config.COLLECTION_NAME in names:
                count = client.get_collection(config.COLLECTION_NAME).count()
                if count > 0:
                    line(OK, f"'{config.COLLECTION_NAME}' holds {count} vector(s)")
                else:
                    line(BAD, f"'{config.COLLECTION_NAME}' exists but is empty")
                    problems.append("collection is empty")
            else:
                line(BAD, f"'{config.COLLECTION_NAME}' not found",
                     f"Found {names} instead - change COLLECTION_NAME in config.py "
                     f"or rerun step 05")
                problems.append("collection name mismatch")

    except Exception as exc:
        line(BAD, f"Cannot open the store: {type(exc).__name__}", str(exc)[:200])
        print("\n         Most common cause: the store was built with a different")
        print("         chromadb version than the one installed now. Pin the same")
        print("         version locally and on Streamlit, then rebuild step 05.")
        problems.append("cannot open store")

# ---------- 4. API key ----------
print("\n--- OpenRouter key ---")
try:
    prompting = config.load_step("07_prompting.py", "prompting")
    if prompting.OPENROUTER_API_KEY:
        line(OK, "Key found in the environment", f"Model: {prompting.OPENROUTER_MODEL}")
    else:
        line(WARN, "No key in environment variables",
             "Fine on Streamlit Cloud, which reads from Secrets. "
             "Retrieval works; generation will not.")
except Exception as exc:
    line(BAD, f"Could not load 07_prompting.py: {exc}")
    problems.append("07_prompting.py broken")

# ---------- Verdict ----------
print("\n" + "=" * 64)
if not problems:
    print("  Everything checks out. Start the app:")
    print("     streamlit run streamlit_app.py")
else:
    print(f"  {len(problems)} problem(s) found:")
    for p in problems:
        print(f"     - {p}")
    if first_missing:
        print(f"\n  Start here: {first_missing}")
        print("  Then continue through step 05 in order.")
print("=" * 64)

sys.exit(1 if problems else 0)
