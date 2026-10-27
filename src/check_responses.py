"""Sanity-check a (small) collection run and project the cost of the full run.

    python check_responses.py --data small_hard.jsonl --responses small_resp.jsonl --full data_hard.jsonl --max_tokens 8192

Prints, per model: calls, errors, answers missing 'Final answer', likely truncated answers, accuracy,
cost; then accuracy by dataset source and a projected full-run cost. Safe to paste the output here.
"""
import argparse
import json
import re
from collections import Counter, defaultdict

from simulate import is_correct, canon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the small file you collected on")
    ap.add_argument("--responses", default="responses.jsonl")
    ap.add_argument("--full", default=None, help="the full data file, to project cost")
    ap.add_argument("--max_tokens", type=int, default=4096, help="the --max_tokens you used in collect.py")
    a = ap.parse_args()

    items = {}
    for l in open(a.data, encoding="utf-8"):
        if l.strip():
            it = json.loads(l)
            items[it["id"]] = it
    recs = [json.loads(l) for l in open(a.responses, encoding="utf-8") if l.strip()]
    by_model = defaultdict(list)
    for r in recs:
        if r["id"] in items:
            by_model[r["model"]].append(r)
    print(f"{len(items)} prompts in {a.data}; {len(recs)} response lines in {a.responses}")
    print("sources:", dict(Counter(it.get("source", "?") for it in items.values())))

    print(f"\n{'model':40s} {'ok':>4s} {'err':>4s} {'noFinal':>7s} {'trunc':>6s} {'acc':>6s} {'$total':>8s} {'$/item':>8s} {'avgOut':>7s}")
    src_cost = defaultdict(lambda: defaultdict(float))
    src_n = Counter(it.get("source", "?") for it in items.values())
    acc_src = defaultdict(lambda: defaultdict(list))
    problems = []
    for m, rs in sorted(by_model.items()):
        good = [r for r in rs if not r.get("error")]
        errs = [r for r in rs if r.get("error")]
        nofinal = sum(1 for r in good if not re.search(r"final answer:", r["text"] or "", re.I))
        trunc = sum(1 for r in good if r.get("finish_reason") == "length" or r.get("completion_tokens", 0) >= 0.98 * a.max_tokens)
        empty = sum(1 for r in good if not (r["text"] or "").strip())
        corr = [is_correct(items[r["id"]], r["text"]) for r in good]
        cost = sum(r["cost"] for r in good if r.get("cost") is not None)
        nocost = sum(1 for r in good if r.get("cost") is None)
        for r, c in zip(good, corr):
            s = items[r["id"]].get("source", "?")
            acc_src[m][s].append(c)
            if r.get("cost") is not None:
                src_cost[m][s] += r["cost"]
        avg_out = sum(r.get("completion_tokens", 0) for r in good) / max(1, len(good))
        print(f"{m:40s} {len(good):4d} {len(errs):4d} {nofinal:7d} {trunc:6d} {sum(corr) / max(1, len(corr)):6.3f} "
              f"{cost:8.4f} {cost / max(1, len(good)):8.5f} {avg_out:7.0f}")
        if errs:
            problems.append(f"{m}: {len(errs)} failed calls, e.g. {Counter(str(e['error'])[:90] for e in errs).most_common(1)[0][0]}")
        if good and nofinal / len(good) > 0.1:
            problems.append(f"{m}: {nofinal}/{len(good)} answers lack 'Final answer:' (format or truncation) - hurts accuracy unfairly")
        if empty:
            problems.append(f"{m}: {empty} empty answers (counted wrong). Usually token cap hit while reasoning, or a provider glitch")
        if good and trunc / len(good) > 0.05:
            problems.append(f"{m}: {trunc}/{len(good)} answers hit the token cap - raise --max_tokens")
        if nocost:
            problems.append(f"{m}: {nocost} responses have no cost field (cost falls back to token counts x prices in collect.py)")
        if len(good) < len(items):
            problems.append(f"{m}: only {len(good)}/{len(items)} prompts answered")

    print("\naccuracy by source (n in brackets):")
    srcs = sorted(src_n)
    print(f"{'model':40s} " + " ".join(f"{s[:10]:>12s}" for s in srcs))
    for m in sorted(by_model):
        print(f"{m:40s} " + " ".join(
            f"{(sum(acc_src[m][s]) / len(acc_src[m][s]) if acc_src[m][s] else float('nan')):6.2f}({len(acc_src[m][s]):3d})" for s in srcs))

    # a few raw samples of failures so formatting problems are visible
    print("\nsample of wrong answers (to eyeball extraction problems):")
    shown = 0
    for r in recs:
        if r["id"] in items and not r.get("error") and shown < 4 and is_correct(items[r["id"]], r["text"]) == 0:
            tail = (r["text"] or "").strip().replace("\n", " ")[-160:]
            print(f"  [{r['model'].split('/')[-1]}] gold={items[r['id']]['answer']!r} parsed={canon(items[r['id']], r['text'])!r} ... {tail!r}")
            shown += 1

    if a.full:
        full = Counter()
        for l in open(a.full, encoding="utf-8"):
            if l.strip():
                full[json.loads(l).get("source", "?")] += 1
        tot, missing = 0.0, []
        print(f"\nprojected cost of the full run ({sum(full.values())} prompts), from per-source average cost in the small run:")
        for m in sorted(by_model):
            proj = 0.0
            for s, n_full in full.items():
                if src_n[s] and acc_src[m][s]:
                    proj += src_cost[m][s] / len(acc_src[m][s]) * n_full
                elif s not in missing:
                    missing.append(s)
            tot += proj
            print(f"  {m:40s} ${proj:8.2f}")
        print(f"  {'TOTAL (approx)':40s} ${tot:8.2f}")
        if missing:
            print(f"  [note] no small-run data for sources {missing}: their cost is NOT included. Make the small file "
                  f"cover every source.")

    print("\nPROBLEMS:" if problems else "\nNo obvious problems found.")
    for p in problems:
        print("  -", p)


if __name__ == "__main__":
    main()
