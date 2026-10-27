"""Two extra probes for the base model (gpt-oss-120b), for BOTH the old and the fresh prompt set.

  logprob   : re-ask the base model with logprobs=true; score = - mean token log-probability of the returned answer tokens
              (low confidence = harder). Some providers return no logprobs for reasoning models: the script reports how many
              prompts got them; prompts without get score = NaN and the evaluation drops that router if too many are missing.
  verbal    : ask the base model, in a separate short call, how likely a small model is to answer the task correctly (0-100);
              score = 100 - stated probability.

    python collect_extra.py --data data_final.jsonl --out_prefix extra_old
    python collect_extra.py --data data_fresh.jsonl --out_prefix extra_fresh
Writes <prefix>_logprob.jsonl and <prefix>_verbal.jsonl in the score-file format ({"id","score","cost","reuse"}).
Costs a few cents per 100 prompts. Resumable (re-run skips finished ids). Put next to collect.py.
"""
import argparse, json, math, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests
from collect import URL, get_key, SYSTEM

BASE = "openai/gpt-oss-120b"; lock = threading.Lock()
VERB = ("Here is a task. Do NOT solve it. Estimate the probability (0-100) that a language model of your own kind answers "
        "it correctly on a single attempt. Reply with the integer only.\n\nTASK:\n{p}\n\nProbability (0-100):")

def post(body, retries=4):
    h = {"Authorization": f"Bearer {get_key()}"}; err = ""
    for k in range(retries):
        try:
            r = requests.post(URL, json=body, headers=h, timeout=180); r.raise_for_status(); return r.json()
        except Exception as e:  # noqa: BLE001
            err = str(e); time.sleep(2 ** k)
    return {"error": err}

def lp_call(prompt):
    d = post({"model": BASE, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], "temperature": 0,
              "max_tokens": 8000, "logprobs": True, "top_logprobs": 1, "usage": {"include": True}})
    if "choices" not in d: return None
    ch = d["choices"][0]; lps = [t["logprob"] for t in ((ch.get("logprobs") or {}).get("content") or []) if t.get("logprob") is not None]
    return dict(score=(-sum(lps) / len(lps)) if lps else float("nan"), n=len(lps), cost=(d.get("usage") or {}).get("cost") or 0.0)

def vb_call(prompt):
    d = post({"model": BASE, "messages": [{"role": "user", "content": VERB.format(p=prompt[:6000])}], "temperature": 0,
              "max_tokens": 3000, "usage": {"include": True}})
    if "choices" not in d: return None
    m = re.findall(r"\d+", (d["choices"][0].get("message") or {}).get("content") or "")
    v = min(100, int(m[-1])) if m else float("nan")
    return dict(score=100 - v if v == v else v, cost=(d.get("usage") or {}).get("cost") or 0.0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); get_key()
    items = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit: items = items[:a.limit]
    for tag, fn, reuse in (("logprob", lp_call, BASE), ("verbal", vb_call, None)):
        path = f"{a.out_prefix}_{tag}.jsonl"; done = set()
        if os.path.exists(path): done = {json.loads(l)["id"] for l in open(path) if l.strip()}
        todo = [it for it in items if it["id"] not in done]
        def work(it):
            r = fn(it["prompt"])
            if r is None: return
            row = {"id": it["id"], "score": r["score"], "cost": r["cost"]}
            if reuse: row["reuse"] = reuse
            with lock:
                with open(path, "a") as f: f.write(json.dumps(row) + "\n")
        with ThreadPoolExecutor(a.workers) as ex: list(ex.map(work, todo))
        rows = [json.loads(l) for l in open(path) if l.strip()]
        ok = sum(1 for r in rows if r["score"] == r["score"])
        print(f"{tag}: {len(rows)} written, {ok} with a usable score ({100*ok/max(1,len(rows)):.0f}%)")

if __name__ == "__main__": main()
