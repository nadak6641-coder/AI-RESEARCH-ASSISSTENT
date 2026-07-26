"""
01_documents.py - Build a topic-focused corpus
==============================================
Searches arXiv for papers on ONE topic, ranks them by influence, and
downloads the PDFs.

Why a single topic rather than trending papers:
A mixed-topic index answers one paper at a time and the rest is dead weight.
When every paper sits in the same field they share vocabulary and metrics, so
comparison questions ("what approaches to X exist?") actually work. Edit
TOPIC_TERMS in config.py to change the field.

Why we rank by citations:
Newest is not the same as most useful. A researcher entering a field wants the
influential work. Citation counts come from Semantic Scholar; if that lookup
fails we keep arXiv's own relevance order and carry on.

Why we do not use the arxiv library:
Its interface has changed repeatedly (version 4.0 removed download_pdf).
We call the public arXiv API directly and parse the Atom feed with the
standard library, then fetch PDFs from a URL pattern that has been stable
for years:  https://arxiv.org/pdf/{arxiv_id}

Output: PDFs in data/raw_pdfs/ and metadata in data/documents.json

Run:
    python 01_documents.py
    python 01_documents.py --max 30
"""

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET

import requests

import config

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StudentRAGProject/1.0)"}
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


# ============================================================
#  1) Search arXiv for the topic
# ============================================================

def build_search_query() -> str:
    """
    Compose an arXiv query: any of our categories AND any of our topic terms.

    Produces something like:
        (cat:cs.CL OR cat:cs.LG) AND (abs:"large language model" OR abs:"...")
    """
    categories = " OR ".join(f"cat:{c}" for c in config.ARXIV_CATEGORIES)
    terms = " OR ".join(f'abs:"{t}"' for t in config.TOPIC_TERMS)
    return f"({categories}) AND ({terms})"


def parse_atom(xml_text: str) -> list[dict]:
    """Parse the arXiv Atom feed into plain dictionaries."""
    root = ET.fromstring(xml_text)
    papers = []

    for entry in root.findall("atom:entry", ATOM_NS):
        def text_of(tag):
            node = entry.find(f"atom:{tag}", ATOM_NS)
            return node.text.strip().replace("\n", " ") if node is not None and node.text else ""

        entry_id = text_of("id")                       # http://arxiv.org/abs/2401.12345v1
        if not entry_id:
            continue
        arxiv_id = entry_id.rsplit("/", 1)[-1].split("v")[0]

        authors = [
            a.text.strip()
            for a in entry.findall("atom:author/atom:name", ATOM_NS)
            if a.text
        ]

        papers.append({
            "id": arxiv_id,
            "title": " ".join(text_of("title").split()),
            "abstract": " ".join(text_of("summary").split()),
            "published": text_of("published")[:10],
            "authors": authors[:6],
            "url": entry_id,
            "citations": None,
        })

    return papers


def search_arxiv(limit: int) -> list[dict]:
    """Fetch candidate papers from the arXiv API."""
    # Pull extra candidates so citation ranking has something to choose from
    candidates = limit * 3 if config.RANK_BY_CITATIONS else limit

    params = {
        "search_query": build_search_query(),
        "start": 0,
        "max_results": min(candidates, 100),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    print(f"Searching arXiv for: {config.TOPIC_LABEL}")
    print(f"   categories: {', '.join(config.ARXIV_CATEGORIES)}")
    print(f"   terms: {', '.join(config.TOPIC_TERMS)}")

    resp = requests.get(config.ARXIV_API, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    papers = parse_atom(resp.text)
    print(f"   Found {len(papers)} candidate(s)")
    return papers


# ============================================================
#  2) Rank by citation count
# ============================================================

def add_citation_counts(papers: list[dict]) -> list[dict]:
    """
    Ask Semantic Scholar how often each paper has been cited.
    One batch request for the whole list. On any failure we return the papers
    untouched and the caller keeps arXiv's relevance order.
    """
    if not papers:
        return papers

    ids = [f"ARXIV:{p['id']}" for p in papers]

    try:
        resp = requests.post(
            config.SEMANTIC_SCHOLAR_BATCH,
            params={"fields": "citationCount"},
            json={"ids": ids},
            headers=HEADERS,
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"   Citation lookup returned {resp.status_code}, keeping relevance order")
            return papers

        records = resp.json()
    except Exception as exc:
        print(f"   Citation lookup failed ({type(exc).__name__}), keeping relevance order")
        return papers

    if not isinstance(records, list) or len(records) != len(papers):
        print("   Citation lookup shape unexpected, keeping relevance order")
        return papers

    found = 0
    for paper, record in zip(papers, records):
        if isinstance(record, dict) and record.get("citationCount") is not None:
            paper["citations"] = record["citationCount"]
            found += 1

    print(f"   Citation counts for {found}/{len(papers)} paper(s)")
    return papers


def select_papers(papers: list[dict], limit: int) -> list[dict]:
    """Rank by citations when we have them, otherwise keep relevance order."""
    if not config.RANK_BY_CITATIONS:
        return papers[:limit]

    papers = add_citation_counts(papers)

    if any(p["citations"] is not None for p in papers):
        papers = sorted(papers, key=lambda p: (p["citations"] or -1), reverse=True)
        print("   Ranked by citation count")
    return papers[:limit]


# ============================================================
#  3) Download PDFs
# ============================================================

def download_pdf(arxiv_id: str, destination) -> bool:
    """
    Download a PDF straight from arXiv.
    Verifies the response really is a PDF rather than an HTML error page.
    """
    if destination.exists() and destination.stat().st_size > 10_000:
        print("      Already downloaded, skipping")
        return True

    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)

    for attempt in (1, 2):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=90, allow_redirects=True)
        except requests.RequestException as exc:
            print(f"      Attempt {attempt} failed: {type(exc).__name__}")
            time.sleep(3)
            continue

        if resp.status_code != 200:
            print(f"      Attempt {attempt}: HTTP {resp.status_code}")
            time.sleep(3)
            continue

        if not resp.content.startswith(b"%PDF"):
            print(f"      Attempt {attempt}: response was not a PDF")
            time.sleep(3)
            continue

        destination.write_bytes(resp.content)
        print(f"      Downloaded ({len(resp.content) / 1_048_576:.1f} MB)")
        return True

    return False


def download_all(papers: list[dict]) -> list[dict]:
    config.RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(papers)} paper(s):\n")
    documents = []

    for i, paper in enumerate(papers, 1):
        title = paper["title"] or f"arXiv {paper['id']}"
        safe = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
        filename = f"{i:02d}_{safe[:70].strip().replace(' ', '_') or paper['id']}.pdf"

        cited = f"  [{paper['citations']} citations]" if paper["citations"] is not None else ""
        print(f"  ({i}/{len(papers)}) {title[:58]}{cited}")

        if not download_pdf(paper["id"], config.RAW_PDF_DIR / filename):
            print("      Giving up on this paper")
            continue

        documents.append({
            "filename": filename,
            "title": title,
            "authors": paper["authors"],
            "published": paper["published"],
            "arxiv_id": paper["id"],
            "url": paper["url"],
            "citations": paper["citations"],
            "abstract": paper["abstract"],
        })

        time.sleep(1)  # be polite to arXiv

    config.DOCUMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.DOCUMENTS_FILE.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return documents


def main():
    parser = argparse.ArgumentParser(description="Build a topic-focused paper corpus")
    parser.add_argument("--max", type=int, default=config.DEFAULT_MAX_PAPERS,
                        help="how many papers to keep")
    args = parser.parse_args()

    try:
        candidates = search_arxiv(args.max)
    except Exception as exc:
        print(f"arXiv search failed: {exc}")
        print("Check your internet connection and try again.")
        sys.exit(1)

    if not candidates:
        print("\nNo papers matched. Your topic may be too narrow.")
        print("Widen TOPIC_TERMS or add categories in config.py.")
        sys.exit(1)

    papers = select_papers(candidates, args.max)
    documents = download_all(papers)

    if not documents:
        print("\nNothing downloaded. Open https://arxiv.org in a browser to")
        print("confirm it is reachable from your network.")
        sys.exit(1)

    print(f"\nCorpus: {len(documents)} paper(s) on {config.TOPIC_LABEL}")
    print(f"PDFs:     {config.RAW_PDF_DIR}")
    print(f"Metadata: {config.DOCUMENTS_FILE}")
    print("\nNext step: python 02_preprocessing.py")


if __name__ == "__main__":
    main()
