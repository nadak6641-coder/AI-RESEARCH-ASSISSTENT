"""
streamlit_app.py - The interface
================================
Calls step 06 (retrieval) and step 07 (generation).

Design intent: a RAG answer is only as good as what was retrieved, so the
evidence is not hidden behind an expander. The match strength of every
retrieved passage is shown *above* the answer, and citation numbers in the
answer map directly onto the numbered source cards below it.

The API key is read from Streamlit secrets when deployed and from environment
variables when running locally. No key is written in the code.

Run:
    streamlit run streamlit_app.py
"""

import html
import re

import streamlit as st

import config

st.set_page_config(
    page_title="Grounded - Research Assistant",
    page_icon="⁝",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------- Load pipeline steps ----------
rag = config.load_step("07_prompting.py", "prompting")

# ---------- Key: Streamlit secrets first, environment second ----------
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass


# ============================================================
#  Styling
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap');

:root {
  --paper:   #EDF0F4;
  --surface: #FFFFFF;
  --ink:     #121D33;
  --muted:   #66728C;
  --accent:  #A3161A;
  --rule:    #D3DAE4;
}

.stApp { background: var(--paper); }
.block-container { padding-top: 2.6rem; max-width: 46rem; }

html, body, [class*="css"], .stMarkdown, p, li, div { font-family: 'IBM Plex Sans', system-ui, sans-serif; }

/* ---------- Masthead ---------- */
.eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem; font-weight: 500;
  letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 0.5rem;
}
.masthead {
  font-family: 'IBM Plex Serif', Georgia, serif;
  font-size: 2.6rem; font-weight: 600; line-height: 1.05;
  color: var(--ink); margin: 0 0 0.7rem 0;
}
.constraint {
  font-size: 0.96rem; line-height: 1.6; color: var(--muted);
  border-left: 2px solid var(--accent); padding-left: 0.9rem; margin-bottom: 1.4rem;
}
.constraint strong { color: var(--ink); font-weight: 500; }

/* ---------- Corpus strip ---------- */
.corpus {
  display: flex; gap: 2.2rem; padding: 0.85rem 1.1rem;
  background: var(--surface); border: 1px solid var(--rule); border-radius: 3px;
  margin-bottom: 1.8rem;
}
.corpus-item { display: flex; flex-direction: column; gap: 0.15rem; }
.corpus-num {
  font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem;
  font-weight: 600; color: var(--ink); line-height: 1;
}
.corpus-label {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem;
  letter-spacing: 0.13em; text-transform: uppercase; color: var(--muted);
}

/* ---------- Evidence meter ---------- */
.evidence-head {
  display: flex; justify-content: space-between; align-items: baseline;
  margin: 1.6rem 0 0.55rem 0;
}
.section-label {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
  letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted);
}
.verdict { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 500; }
.verdict.strong   { color: #1B6B4A; }
.verdict.moderate { color: #9A6510; }
.verdict.weak     { color: var(--accent); }

.meter {
  display: flex; gap: 4px; align-items: flex-end; height: 62px;
  padding: 0.7rem 0.9rem 0.5rem 0.9rem;
  background: var(--surface); border: 1px solid var(--rule); border-radius: 3px;
}
.seg { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 5px; }
.seg-bar {
  width: 100%; background: var(--ink); border-radius: 1px;
  animation: rise 340ms cubic-bezier(0.2, 0.8, 0.3, 1) both;
}
.seg-bar.moderate { background: #7E8AA6; }
.seg-bar.weak     { background: #C5CCD8; }
.seg-num {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
  color: var(--muted); line-height: 1;
}
@keyframes rise { from { height: 2px; opacity: 0; } }
@media (prefers-reduced-motion: reduce) { .seg-bar { animation: none; } }

/* ---------- Answer ---------- */
.answer {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 3px;
  padding: 1.4rem 1.5rem; margin-top: 0.5rem;
  font-size: 1.02rem; line-height: 1.75; color: var(--ink);
}
.cite {
  display: inline-block; min-width: 1.15rem; height: 1.15rem;
  font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 600;
  line-height: 1.15rem; text-align: center; color: var(--accent);
  border: 1px solid var(--accent); border-radius: 2px;
  margin: 0 0.12rem; vertical-align: 0.12rem;
}

/* ---------- Source cards ---------- */
.source {
  display: flex; gap: 0.95rem; background: var(--surface);
  border: 1px solid var(--rule); border-radius: 3px;
  padding: 0.9rem 1.1rem; margin-bottom: 0.55rem;
}
.source-rail { display: flex; flex-direction: column; align-items: center; gap: 0.4rem; min-width: 1.9rem; }
.source-num {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 600;
  color: var(--accent); border: 1px solid var(--accent); border-radius: 2px;
  width: 1.5rem; height: 1.5rem; line-height: 1.42rem; text-align: center;
}
.source-score { font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; color: var(--muted); }
.source-body { flex: 1; min-width: 0; }
.source-title {
  font-family: 'IBM Plex Serif', Georgia, serif; font-size: 0.92rem;
  font-weight: 600; color: var(--ink); line-height: 1.35; margin-bottom: 0.12rem;
}
.source-meta {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.63rem;
  color: var(--muted); margin-bottom: 0.45rem;
}
.source-text { font-size: 0.85rem; line-height: 1.6; color: #35415C; }

/* ---------- Footnote ---------- */
.footnote {
  font-size: 0.78rem; color: var(--muted); margin-top: 1.2rem;
  padding-top: 0.8rem; border-top: 1px solid var(--rule);
}

/* ---------- Native controls ---------- */
.stTextInput input {
  font-family: 'IBM Plex Sans', sans-serif; font-size: 0.98rem;
  border-radius: 3px; border: 1px solid var(--rule); background: var(--surface);
}
.stTextInput input:focus { border-color: var(--ink); box-shadow: none; }
.stButton button {
  font-family: 'IBM Plex Sans', sans-serif; font-size: 0.82rem;
  background: var(--surface); color: #35415C;
  border: 1px solid var(--rule); border-radius: 3px;
  padding: 0.5rem 0.8rem; text-align: left; line-height: 1.35;
}
.stButton button:hover { border-color: var(--ink); color: var(--ink); background: var(--surface); }
.stButton button[kind="primary"] {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem;
  letter-spacing: 0.1em; text-transform: uppercase; text-align: center;
  background: var(--ink); color: #FFFFFF; border: none; padding: 0.55rem 1.5rem;
}
.stButton button[kind="primary"]:hover { background: var(--accent); color: #FFFFFF; }
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--rule); }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  Helpers
# ============================================================
STRONG_AT = 0.86
MODERATE_AT = 0.79


def strength_of(score: float) -> str:
    if score >= STRONG_AT:
        return "strong"
    if score >= MODERATE_AT:
        return "moderate"
    return "weak"


def bar_height(score: float) -> int:
    """Map a similarity score onto a bar height, clamped to a useful range."""
    ratio = (score - 0.60) / (0.95 - 0.60)
    return max(6, min(100, int(ratio * 100)))


def render_answer(text: str) -> str:
    """Escape the answer, then style [1] style citation markers."""
    safe = html.escape(text)
    safe = re.sub(r"\[(?:Source\s*)?(\d+)\]", r'<span class="cite">\1</span>', safe)
    return safe.replace("\n", "<br>")


@st.cache_resource(show_spinner="Loading the index...")
def load_index():
    retrieval = config.load_step("06_retrieve_context.py", "retrieve_context")
    collection = retrieval.get_collection()
    retrieval.get_model()  # warm the model up front
    return retrieval, collection.count(), retrieval.list_indexed_papers()


# ============================================================
#  Masthead
# ============================================================
st.markdown('<div class="eyebrow">Retrieval-augmented research assistant</div>',
            unsafe_allow_html=True)
st.markdown('<h1 class="masthead">Grounded</h1>', unsafe_allow_html=True)
st.markdown(
    f'<div class="constraint">This index covers <strong>{html.escape(config.TOPIC_LABEL)}</strong>. '
    'Every answer is assembled from passages retrieved out of those papers, and '
    'every claim carries the number of the passage it came from. '
    '<strong>If the papers do not say it, neither does this.</strong></div>',
    unsafe_allow_html=True,
)


# ============================================================
#  Index
# ============================================================
try:
    retrieval, chunk_count, papers = load_index()
except Exception as exc:
    from pathlib import Path

    store = Path(config.CHROMA_DIR)
    st.error("The vector index is not available.")

    with st.expander("Diagnostics", expanded=True):
        st.markdown(f"**Expected path:** `{store}`")
        st.markdown(f"**Folder exists:** {'yes' if store.exists() else 'no'}")

        if store.exists():
            st.markdown(f"**Contents:** `{[p.name for p in store.iterdir()] or 'empty'}`")
            try:
                import chromadb
                client = chromadb.PersistentClient(path=config.CHROMA_DIR)
                found = [c if isinstance(c, str) else c.name for c in client.list_collections()]
                st.markdown(f"**Collections found:** `{found or 'none'}`")
                st.markdown(f"**Collection expected:** `{config.COLLECTION_NAME}`")
                st.markdown(f"**chromadb version:** `{chromadb.__version__}`")
            except Exception as inner:
                st.markdown(f"**Could not open the store:** `{inner}`")

        st.code(str(exc), language="text")

    st.markdown(
        "Run `python check_setup.py` to see which step is missing. "
        "On Streamlit Cloud the usual cause is that `data/chroma_store/` "
        "was never committed to GitHub."
    )
    st.stop()


st.markdown(
    f"""
    <div class="corpus">
      <div class="corpus-item">
        <div class="corpus-num">{len(papers)}</div>
        <div class="corpus-label">Papers</div>
      </div>
      <div class="corpus-item">
        <div class="corpus-num">{chunk_count}</div>
        <div class="corpus-label">Passages indexed</div>
      </div>
      <div class="corpus-item">
        <div class="corpus-num">{config.CHUNK_SIZE_WORDS}</div>
        <div class="corpus-label">Words per passage</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
#  Sidebar
# ============================================================
with st.sidebar:
    st.markdown('<div class="section-label">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider(
        "Passages per question", min_value=2, max_value=10,
        value=config.DEFAULT_TOP_K, label_visibility="visible",
        help="More passages widen the context but also add noise.",
    )

    st.markdown('<div class="section-label">Generation</div>', unsafe_allow_html=True)
    st.code(rag.OPENROUTER_MODEL, language=None)
    if not rag.OPENROUTER_API_KEY:
        st.warning("No API key. Retrieval works; answers will not generate.")

    st.markdown('<div class="section-label">In the index</div>', unsafe_allow_html=True)
    for title in papers:
        st.markdown(
            f'<div style="font-size:0.76rem;line-height:1.45;color:#66728C;'
            f'margin-bottom:0.45rem;">{html.escape(title)}</div>',
            unsafe_allow_html=True,
        )


# ============================================================
#  Question
# ============================================================
SUGGESTIONS = [
    "What architectures are described across these papers?",
    "How are the models evaluated?",
    "What limitations do the authors mention?",
    "What training data is used?",
]

if "question" not in st.session_state:
    st.session_state.question = ""


def use_suggestion(text: str):
    st.session_state.question = text


st.markdown('<div class="section-label" style="margin-bottom:0.4rem;">Try one</div>',
            unsafe_allow_html=True)
cols = st.columns(2)
for i, suggestion in enumerate(SUGGESTIONS):
    cols[i % 2].button(
        suggestion,
        key=f"sug_{i}",
        on_click=use_suggestion,
        args=(suggestion,),
        use_container_width=True,
    )

question = st.text_input(
    "Ask the papers",
    key="question",
    placeholder="Or type your own question",
    label_visibility="collapsed",
)
st.button("Ask", type="primary", disabled=not question.strip())

if not question.strip():
    st.markdown(
        f'<div class="footnote">This index covers {config.TOPIC_LABEL}. '
        'Questions it cannot answer get a plain "not in the sources" reply '
        'rather than a guess.</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
#  Retrieve
# ============================================================
with st.spinner("Searching the passages..."):
    contexts = retrieval.retrieve(question, top_k=top_k)

if not contexts:
    st.info("Nothing matched. Try rephrasing the question.")
    st.stop()

best = max(c["score"] for c in contexts)
verdict = strength_of(best)
verdict_text = {
    "strong": "Strong match",
    "moderate": "Partial match",
    "weak": "Weak match - treat the answer with care",
}[verdict]

segments = "".join(
    f'<div class="seg">'
    f'<div class="seg-bar {strength_of(c["score"])}" '
    f'style="height:{bar_height(c["score"])}%"></div>'
    f'<div class="seg-num">{c["rank"]}</div>'
    f'</div>'
    for c in contexts
)

st.markdown(
    f'<div class="evidence-head">'
    f'<span class="section-label">Evidence retrieved</span>'
    f'<span class="verdict {verdict}">{verdict_text}</span>'
    f'</div>'
    f'<div class="meter">{segments}</div>',
    unsafe_allow_html=True,
)


# ============================================================
#  Answer
# ============================================================
with st.spinner("Composing the answer..."):
    answer = rag.generate_answer(question, contexts)

st.markdown('<div class="evidence-head"><span class="section-label">Answer</span></div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="answer">{render_answer(answer)}</div>', unsafe_allow_html=True)


# ============================================================
#  Sources
# ============================================================
st.markdown('<div class="evidence-head"><span class="section-label">Sources</span></div>',
            unsafe_allow_html=True)

for ctx in contexts:
    snippet = ctx["text"][:340] + ("..." if len(ctx["text"]) > 340 else "")
    st.markdown(
        f"""
        <div class="source">
          <div class="source-rail">
            <div class="source-num">{ctx['rank']}</div>
            <div class="source-score">{ctx['score']:.2f}</div>
          </div>
          <div class="source-body">
            <div class="source-title">{html.escape(ctx['paper_title'])}</div>
            <div class="source-meta">{retrieval.format_pages(ctx)}</div>
            <div class="source-text">{html.escape(snippet)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footnote">Scores are cosine similarity between the question '
    'and each passage. They are relative, not absolute truth - a high score '
    'means the passage is close in meaning, not that the answer is correct.</div>',
    unsafe_allow_html=True,
)
