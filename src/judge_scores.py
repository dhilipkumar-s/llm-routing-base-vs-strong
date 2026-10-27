"""LLM-judge complexity scores: ask a cheap model to rate how hard each prompt is.

    python judge_scores.py --data data_final.jsonl --out judge.jsonl --judge meta-llama/llama-3.3-70b-instruct
    python simulate.py --data data_final.jsonl --responses responses_final.jsonl --outdir results_judge --scores_file judge.jsonl

The judge sees ONLY the prompt (never the answer), writes a 1-10 difficulty rating, and its own cost is saved as
router overhead. Resumes if interrupted. Pick a NON-reasoning judge: reasoning models spend the small output cap
thinking and return nothing. Expect well under $1 for ~800 prompts with a cheap judge.
"""
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from collect import URL, get_key
from simulate import PRICE

lock = threading.Lock()

TEMPLATE = (
    "You are rating how hard a task is for a SMALL language model (about 8 billion parameters) to answer correctly.\n"
    "1 = trivial, the small model almost always gets it right. 10 = extremely hard, the small model almost always "
    "gets it wrong (multi-step reasoning, expert knowledge, long calculations, tricky traps).\n"
    "Do NOT solve the task. Reply with a single integer from 1 to 10 and nothing else.\n\n"
    "TASK:\n{prompt}\n\nDifficulty (1-10):"
)


def ask(judge, prompt, max_chars, retries=4):
    body = {"model": judge,
            "messages": [{"role": "user", "content": TEMPLATE.format(prompt=prompt[:max_chars])}],
            "temperature": 0, "max_tokens": 16, "usage": {"include": True}}
    headers = {"Authorization": f"Bearer {get_key()}"}
    err = "unknown"
    for attempt in range(retries):
        try:
            r = requests.post(URL, json=body, headers=headers, timeout=120)
            r.raise_for_status()
            d = r.json()
            u = d.get("usage", {})
            text = (d["choices"][0].get("message") or {}).get("content") or ""
            m = re.search(r"\b(10|[1-9])\b", text)
            if not m:
                raise ValueError(f"could not parse a 1-10 rating from {text!r}")
            cost = u.get("cost")
            if cost is None:
                pin, pout = PRICE.get(judge, (0.0, 0.0))
                cost = (u.get("prompt_tokens", 0) * pin + u.get("completion_tokens", 0) * pout) / 1e6
            return {"score": float(m.group(1)), "cost": float(cost), "error": None}
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 ** attempt)
    return {"score": None, "cost": 0.0, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="judge.jsonl")
    ap.add_argument("--judge", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_chars", type=int, default=6000, help="truncate very long prompts to control cost")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (for a cheap test)")
    a = ap.parse_args()
    get_key()

    items = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit:
        items = items[:a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            done.add(json.loads(l)["id"])
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(todo)} prompts to rate with {a.judge} ({len(done)} already done)")

    def work(it):
        r = ask(a.judge, it["prompt"], a.max_chars)
        if r["score"] is None:
            print(f"[fail] {it['id']}: {r['error'][:100]}")
            return
        with lock:
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": it["id"], "score": r["score"], "cost": r["cost"]}) + "\n")

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    rows = [json.loads(l) for l in open(a.out, encoding="utf-8")]
    print(f"done: {len(rows)} scores in {a.out}; total judge cost ${sum(r['cost'] for r in rows):.4f}")
    if rows:
        from collections import Counter
        print("rating histogram:", dict(sorted(Counter(int(r["score"]) for r in rows).items())))


if __name__ == "__main__":
    main()
