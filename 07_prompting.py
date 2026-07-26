"""
07_prompting.py - Build the prompt and generate the answer
==========================================================
The retrieved context and the user's question are combined into one prompt
and sent to a model through OpenRouter.

The most important line in this project is in the system prompt below: the
instruction to say "not in the sources" instead of guessing. Without it the
model invents confident, well-written, wrong answers, which is the worst
possible failure mode for a research assistant because the error looks
correct.

API key:
    - read from an environment variable or from Streamlit secrets
    - never hardcode a real key here, and never commit a .env file

Run:
    export OPENROUTER_API_KEY="..."      # Windows: set OPENROUTER_API_KEY=...
    python 07_prompting.py "What is retrieval augmented generation?"
"""

import os
import sys

import requests

import config

# Read from the environment. streamlit_app.py may overwrite these from secrets.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are a careful research assistant.

Rules you must follow:
1. Answer ONLY from the sources provided in the user message.
2. If the answer is not in the sources, say so plainly. Never invent facts,
   numbers, or citations.
3. Cite the source number in square brackets - [1], [2] - after every claim
   that comes from a source.
4. Answer in the same language as the question.
5. Be concise. Prefer accuracy over completeness."""


def build_prompt(question: str, contexts: list[dict]) -> str:
    """Build the user message: numbered sources followed by the question."""
    blocks = []
    for ctx in contexts:
        if ctx.get("page_start") is not None:
            page_label = (
                f"p.{ctx['page_start']}"
                if ctx.get("page_start") == ctx.get("page_end")
                else f"pp.{ctx.get('page_start')}-{ctx.get('page_end')}"
            )
        else:
            page_label = ""

        blocks.append(
            f"[{ctx['rank']}] \"{ctx['paper_title']}\" ({page_label})\n{ctx['text']}"
        )

    sources = "\n\n---\n\n".join(blocks)
    return f"## Sources\n\n{sources}\n\n## Question\n\n{question}\n\n## Answer"


def generate_answer(question: str, contexts: list[dict], timeout: int = 60) -> str:
    """Send the prompt to OpenRouter and return the answer text."""
    if not OPENROUTER_API_KEY:
        return ("No API key found. Set OPENROUTER_API_KEY as an environment "
                "variable or in Streamlit secrets.")

    if not contexts:
        return "Nothing was retrieved for this question, so there is nothing to answer from."

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, contexts)},
        ],
        "temperature": 0.1,  # deliberately low: stay close to the sources
    }

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return f"Could not reach OpenRouter: {exc}"

    if resp.status_code != 200:
        return f"OpenRouter returned {resp.status_code}: {resp.text[:300]}"

    try:
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as exc:
        return f"Unexpected response from OpenRouter: {exc}"


def answer_question(question: str, top_k: int = config.DEFAULT_TOP_K) -> dict:
    """Full cycle: retrieve, then generate. Returns answer plus sources."""
    retrieval = config.load_step("06_retrieve_context.py", "retrieve_context")
    contexts = retrieval.retrieve(question, top_k=top_k)
    return {
        "question": question,
        "answer": generate_answer(question, contexts),
        "sources": contexts,
    }


def main():
    if len(sys.argv) < 2:
        print('Usage: python 07_prompting.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = answer_question(question)

    print(f"Question: {result['question']}\n")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    print("\nSources:")
    for src in result["sources"]:
        print(f"  [{src['rank']}] {src['paper_title'][:60]} "
              f"(p. {src['page_start']}) - score {src['score']:.3f}")


if __name__ == "__main__":
    main()
