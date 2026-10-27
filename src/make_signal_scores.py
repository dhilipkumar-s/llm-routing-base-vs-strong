"""Build complexity scores from cheap-model responses you ALREADY collected (no new API calls).

Two signals, both computed without looking at the gold answer:
  effort    : how many tokens a cheap model spent on the prompt (more tokens = harder). One cheap call per prompt.
  disagree  : whether two cheap models give different final answers (disagreement = harder). Two cheap calls.

The cost of those cheap calls is written as "cost" so simulate.py charges it to the router as overhead.

    python make_signal_scores.py --data data_final.jsonl --responses responses_final.jsonl --signal effort   --out effort.jsonl
    python make_signal_scores.py --data data_final.jsonl --responses responses_final.jsonl --signal disagree --out disagree.jsonl
    python simulate.py --data data_final.jsonl --responses responses_final.jsonl --outdir results_signals --scores_file effort.jsonl,disagree.jsonl

Note: the overhead is charged in full even when the router later picks that same cheap model (whose answer is
then already paid for). That is conservative: it slightly over-charges cheap-tier routes.
"""
import argparse
import json
import math
from collections import Counter

from simulate import PRICE, canon


def cost_of(r):
    if r.get("cost") is not None:
        return float(r["cost"])
    pin, pout = PRICE[r["model"]]
    return (r.get("prompt_tokens", 0) * pin + r.get("completion_tokens", 0) * pout) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--responses", default="responses.jsonl")
    ap.add_argument("--signal", choices=["effort", "disagree"], required=True)
    ap.add_argument("--models", default=None,
                    help="comma-separated model slugs. effort: first one is used. disagree: first two are used. "
                         "Default: meta-llama/llama-3.1-8b-instruct (+ openai/gpt-oss-20b for disagree)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    models = a.models.split(",") if a.models else ["meta-llama/llama-3.1-8b-instruct", "openai/gpt-oss-20b"]
    need = models[:1] if a.signal == "effort" else models[:2]
    if len(need) < (1 if a.signal == "effort" else 2):
        raise SystemExit("disagree needs two models in --models")

    items = {}
    for l in open(a.data, encoding="utf-8"):
        if l.strip():
            it = json.loads(l)
            items[it["id"]] = it
    recs = {}
    for l in open(a.responses, encoding="utf-8"):
        r = json.loads(l)
        if not r.get("error"):
            recs[(r["model"], r["id"])] = r

    out, skipped = [], 0
    for i, it in items.items():
        rs = [recs.get((m, i)) for m in need]
        if any(r is None for r in rs):
            skipped += 1
            continue
        over = sum(cost_of(r) for r in rs)
        if a.signal == "effort":
            score = math.log1p(rs[0].get("completion_tokens", 0))
        else:
            c1, c2 = canon(it, rs[0]["text"]), canon(it, rs[1]["text"])
            agree = c1 is not None and c1 == c2
            # disagreement dominates; total tokens only breaks ties
            score = (0.0 if agree else 1.0) + 0.01 * math.log1p(sum(r.get("completion_tokens", 0) for r in rs))
        out.append({"id": i, "score": score, "cost": over, "reuse": need[0]})
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"wrote {len(out)} scores to {a.out} ({skipped} prompts skipped: missing responses from {need})")
    if a.signal == "disagree":
        print("disagreement rate:", f"{sum(1 for o in out if o['score'] >= 1) / max(1, len(out)):.2f}")
    print(f"mean overhead per prompt: ${sum(o['cost'] for o in out) / max(1, len(out)):.6f}")


if __name__ == "__main__":
    main()
