"""Run every query through every model once and cache results to JSONL.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python collect.py --data data.jsonl --out responses.jsonl --workers 8

data.jsonl: one JSON object per line: {"id": "...", "prompt": "...", "answer": "...", "type": "numeric|mcq"}
Re-running skips (id, model) pairs already in the output, so it is safe to resume.
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"

# Pin exact slugs. VERIFY these on openrouter.ai/models before running; slugs change.
# tier: 0 = cheapest ... 3 = frontier. Prices are USD per 1M tokens (in, out) - fill from the model page.
MODELS = [
    # tier 0: small / weak-cheap models (these should actually fail on hard prompts)
    {"slug": "meta-llama/llama-3.1-8b-instruct",   "tier": 0, "in": 0.02, "out": 0.03},
    {"slug": "openai/gpt-oss-20b",                 "tier": 0, "in": 0.03, "out": 0.14},
    # tier 1: capable-cheap
    {"slug": "openai/gpt-oss-120b",                "tier": 1, "in": 0.04, "out": 0.17},
    {"slug": "meta-llama/llama-3.3-70b-instruct",  "tier": 1, "in": 0.10, "out": 0.32},
    {"slug": "qwen/qwen3-235b-a22b-2507",          "tier": 1, "in": 0.15, "out": 0.60},
    # tier 2: strong
    {"slug": "anthropic/claude-sonnet-5.5",        "tier": 2, "in": 0.0,  "out": 0.0},   # fill prices if cost not returned
    # tier 3: frontier (quality reference)
    {"slug": "anthropic/claude-opus-5.5",          "tier": 3, "in": 0.0,  "out": 0.0},   # fill prices if cost not returned
]

# Pin the provider per model if you want to remove serving differences, e.g.
# PROVIDER_PIN = {"openai/gpt-oss-120b": ["DeepInfra"]}
PROVIDER_PIN = {}

# Per-model output cap (overrides --max_tokens when lower). Claude answers here are short (<1000 tokens), and
# OpenRouter reserves credit for the MAX possible output, so a small cap avoids "402 Payment Required" on a low balance.
MAX_TOKENS = {"anthropic/claude-opus-5.5": 4096, "anthropic/claude-sonnet-5.5": 4096}

SYSTEM = "Answer concisely. End with 'Final answer: <answer>' (a number, or a single letter for multiple choice)."

lock = threading.Lock()


def get_key():
    """Use OPENROUTER_API_KEY if set, else read the first line of openrouter_key.txt
    (in the same folder as this script, or the current folder)."""
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if k:
        return k
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, "openrouter_key.txt"), "openrouter_key.txt"):
        if os.path.exists(path):
            k = open(path, encoding="utf-8-sig").readline().strip().strip('"').strip("'")
            if k:
                return k
    raise SystemExit("No API key found. Set OPENROUTER_API_KEY or create openrouter_key.txt "
                     "next to collect.py containing only your key.")


def call(model, prompt, max_tokens=4096, retries=4, temperature=0):
    body = {
        "model": model["slug"],
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if model["slug"] in PROVIDER_PIN:
        body["provider"] = {"order": PROVIDER_PIN[model["slug"]], "allow_fallbacks": False}
    headers = {"Authorization": f"Bearer {get_key()}"}
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = requests.post(URL, json=body, headers=headers, timeout=180)
            r.raise_for_status()
            d = r.json()
            u = d.get("usage", {})
            ch = d["choices"][0]
            text = (ch.get("message") or {}).get("content") or ""
            if not text.strip() and not u.get("completion_tokens"):
                raise ValueError("empty response with 0 output tokens (provider glitch); retrying")
            return {
                "text": text,
                "finish_reason": ch.get("finish_reason"),  # "length" = hit the max_tokens cap (truncated)
                "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0),
                "cost": u.get("cost"),  # actual cost reported by OpenRouter, if available
                "latency": time.time() - t0,
                "provider": d.get("provider"),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 ** attempt)
    return {"text": "", "finish_reason": None, "prompt_tokens": 0, "completion_tokens": 0, "cost": None,
            "latency": None, "provider": None, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="responses.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=4096,
                    help="output token cap per call; use 8192+ for AIME/GPQA with reasoning models")
    args = ap.parse_args()
    get_key()  # fail early with a clear message if no key

    items = [json.loads(l) for l in open(args.data) if l.strip()]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            if not r.get("error"):
                done.add((r["id"], r["model"]))

    jobs = [(it, m) for it in items for m in MODELS if (it["id"], m["slug"]) not in done]
    print(f"{len(jobs)} calls to make ({len(done)} cached)")

    def work(job):
        it, m = job
        res = call(m, it["prompt"], min(args.max_tokens, MAX_TOKENS.get(m["slug"], args.max_tokens)))
        rec = {"id": it["id"], "model": m["slug"], "run_date": time.strftime("%Y-%m-%d"), **res}
        with lock:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, jobs))
    print("done")


if __name__ == "__main__":
    main()
