"""Self-consistency difficulty score for a cheap model: do its own answers agree with each other?

Votes = the model's greedy answer already in responses.jsonl + the K sampled answers in the samples file.
score = 1 - (share of votes for the most common answer)   [0 = unanimous, higher = less consistent]
        + 0.01 x (number of distinct answers)             [tie-break]
Unparsable answers count as their own distinct vote. Overhead = greedy answer cost + sample costs, and the file
carries reuse=<model> so `simulate.py --credit_reuse` does not charge that model's answer twice.

    python selfcons_scores.py --data data_final.jsonl --responses responses_final.jsonl --samples samples_120b.jsonl --out selfcons_120b.jsonl
"""
import argparse, json
from collections import Counter, defaultdict
from simulate import PRICE, canon


def cost_of(r):
    if r.get("cost") is not None:
        return float(r["cost"])
    pin, pout = PRICE.get(r["model"], (0.0, 0.0))
    return (r.get("prompt_tokens", 0) * pin + r.get("completion_tokens", 0) * pout) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--responses", required=True)
    ap.add_argument("--samples", required=True); ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--out", default="selfcons.jsonl")
    ap.add_argument("--max_k", type=int, default=0, help="use only the first K samples per prompt (0 = all)")
    ap.add_argument("--min_samples", type=int, default=1, help="skip prompts with fewer successful samples")
    a = ap.parse_args()
    items = {}
    for l in open(a.data, encoding="utf-8"):
        if l.strip():
            it = json.loads(l); items[it["id"]] = it
    greedy = {}
    for l in open(a.responses, encoding="utf-8"):
        r = json.loads(l)
        if r["model"] == a.model and not r.get("error"):
            greedy[r["id"]] = r
    uniq = {}  # keep one successful sample per (id, k): retries can leave duplicates in the file
    for l in open(a.samples, encoding="utf-8"):
        r = json.loads(l)
        if r["model"] == a.model and not r.get("error") and (not a.max_k or r["k"] < a.max_k):
            uniq[(r["id"], r["k"])] = r
    samp = defaultdict(list)
    for (i_, _k), r in uniq.items():
        samp[i_].append(r)
    out, skipped = [], 0
    for i, it in items.items():
        if i not in greedy or len(samp[i]) < a.min_samples:
            skipped += 1; continue
        rs = [greedy[i]] + samp[i]
        votes = []
        for k, r in enumerate(rs):
            c = canon(it, r["text"])
            votes.append(c if c is not None else f"__unparsed{k}")
        top = Counter(votes).most_common(1)[0][1]
        score = (1 - top / len(votes)) + 0.01 * len(set(votes))
        out.append({"id": i, "score": score, "cost": sum(cost_of(r) for r in rs), "reuse": a.model})
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    inc = sum(1 for o in out if o["score"] >= 0.2) / max(1, len(out))
    print(f"wrote {len(out)} scores to {a.out} ({skipped} skipped); inconsistent on {inc:.0%} of prompts; "
          f"mean overhead ${sum(o['cost'] for o in out) / max(1, len(out)):.6f}/prompt")


if __name__ == "__main__":
    main()
