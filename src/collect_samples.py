"""Draw K extra sampled answers from ONE model per prompt (for self-consistency).

    python collect_samples.py --data data_final.jsonl --out samples_120b.jsonl --model openai/gpt-oss-120b --k 2 --temperature 0.8 --max_tokens 12000
    python collect_samples.py ... --limit 20      # cheap smoke test first

Resumable: skips (id, k) already done without error. Costs roughly k x the model's cost in responses file.
"""
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from collect import call, get_key

lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="samples.jsonl")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_tokens", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    get_key()
    items = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit:
        items = items[:a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("error"):
                done.add((r["id"], r["k"]))
    jobs = [(it, k) for it in items for k in range(a.k) if (it["id"], k) not in done]
    print(f"{len(jobs)} sampled calls to make with {a.model} ({len(done)} cached)")

    def work(job):
        it, k = job
        res = call({"slug": a.model}, it["prompt"], a.max_tokens, temperature=a.temperature)
        rec = {"id": it["id"], "k": k, "model": a.model, "temperature": a.temperature, **res}
        with lock:
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))
    rows = [json.loads(l) for l in open(a.out, encoding="utf-8")]
    ok = [r for r in rows if not r.get("error")]
    print(f"done: {len(ok)} ok, {len(rows) - len(ok)} error lines; cost ${sum(r['cost'] or 0 for r in ok):.4f}")


if __name__ == "__main__":
    main()
