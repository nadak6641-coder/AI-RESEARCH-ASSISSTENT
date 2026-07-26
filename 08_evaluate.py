"""
08_evaluate.py - Measure retrieval quality (extra file, not required)
=====================================================================
Without measurement, every "improvement" is a guess. This gives you a number
you can compare before and after any change: chunk size, embedding model,
top_k, anything.

Two metrics:
- Hit@k : share of questions whose answer appeared in the top k results
- MRR   : mean reciprocal rank of the first correct result (closer to 1 is better)

How do we decide a result is "correct"? Expected keywords. If a retrieved
chunk contains enough of them, it counts. Approximate, but good enough to
compare two configurations honestly.

Usage:
1. Edit eval_set.json with questions about your own papers
2. Run: python 08_evaluate.py
3. Change one setting in config.py, rebuild, run again, compare
"""

import json
import sys

import config

EVAL_FILE = config.BASE_DIR / "eval_set.json"


def matches(text: str, keywords: list[str], threshold: float = 0.6) -> bool:
    """Does this text contain enough of the expected keywords?"""
    if not keywords:
        return False
    lowered = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lowered)
    return (hits / len(keywords)) >= threshold


def main():
    if not EVAL_FILE.exists():
        print(f"Missing {EVAL_FILE.name} - fill it with questions about your papers.")
        sys.exit(1)

    eval_set = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    if not eval_set:
        print("The evaluation set is empty.")
        sys.exit(1)

    retrieval = config.load_step("06_retrieve_context.py", "retrieve_context")
    top_k = config.DEFAULT_TOP_K

    hits = 0
    reciprocal_ranks = []

    print(f"Evaluating {len(eval_set)} question(s) at top_k = {top_k}\n")

    for item in eval_set:
        question = item["question"]
        keywords = item.get("expected_keywords", [])
        results = retrieval.retrieve(question, top_k=top_k)

        found_rank = None
        for result in results:
            if matches(result["text"], keywords):
                found_rank = result["rank"]
                break

        if found_rank:
            hits += 1
            reciprocal_ranks.append(1 / found_rank)
            status = f"hit at {found_rank}"
        else:
            reciprocal_ranks.append(0.0)
            status = "miss"

        print(f"  {status:<12} | {question[:60]}")

    total = len(eval_set)
    print("\n" + "=" * 55)
    print(f"  Hit@{top_k} : {hits}/{total} = {hits / total:.1%}")
    print(f"  MRR      : {sum(reciprocal_ranks) / total:.3f}")
    print("=" * 55)
    print("\nRecord these numbers, change one setting in config.py,")
    print("rebuild steps 03-05, and run this again to compare.")


if __name__ == "__main__":
    main()
