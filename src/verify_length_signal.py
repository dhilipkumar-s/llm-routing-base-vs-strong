"""Recompute Spearman and AUC for the prompt-length scorer by hand, step by step, so you can check Table 4.

    python verify_length_signal.py --data data_final.jsonl --responses responses_final.jsonl --ladder meta-llama/llama-3.1-8b-instruct,meta-llama/llama-3.3-70b-instruct,openai/gpt-oss-120b,anthropic/claude-sonnet-5.5
    python verify_length_signal.py ... --sources mmlu_pro        # the MMLU-Pro columns
    python verify_length_signal.py ... --show 8                  # print 8 example prompts with all three facts

It uses the same split, difficulty definition and functions as simulate.py (seed 0, 50/50, stratified), so the numbers
should match the "length" row of quality.csv from your results_signals run. Use --all to score ALL prompts instead of
only the test half.
"""
import argparse
from collections import defaultdict
import numpy as np
from simulate import load, resolve_ladder, bucket, spearman, auc, rankdata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--responses", required=True)
    ap.add_argument("--ladder", default=None); ap.add_argument("--sources", default=None)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--test_frac", type=float, default=0.5)
    ap.add_argument("--min_coverage", type=float, default=0.9)
    ap.add_argument("--all", action="store_true"); ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()
    items, ids, names, OK, USD, CN = load(a.data, a.responses, a.min_coverage)
    if a.sources:
        want = set(a.sources.split(","))
        keep = [k for k, i in enumerate(ids) if items[i].get("source") in want]
        ids, OK = [ids[k] for k in keep], OK[keep]
    lad = resolve_ladder(names, a.ladder)
    li = [names.index(s) for s in lad]
    A_all = OK[:, li]
    d_all = 1 - OK.mean(1)                                   # true difficulty: share of the pool that fails
    rng = np.random.RandomState(a.seed)
    strata = defaultdict(list)
    for k, i in enumerate(ids):
        strata[bucket(items[i])].append(k)
    te = []
    for b in sorted(strata):
        idx = list(strata[b]); rng.shuffle(idx)
        te += idx[:int(round(len(idx) * a.test_frac))]
    te = np.array(sorted(te)) if not a.all else np.arange(len(ids))
    print(f"models in the pool ({len(names)}): {names}")
    print(f"cheapest model in the ladder (used for the AUC): {lad[0]}")
    print(f"prompts scored: {len(te)} ({'all' if a.all else 'test half of split seed ' + str(a.seed)})\n")
    length = np.array([len(items[ids[k]]["prompt"]) for k in te], float)   # the scorer's score = characters
    d = d_all[te]; fail = (A_all[te, 0] == 0)
    print(f"STEP 1  facts per prompt. First {a.show} examples:")
    print(f"        {'id':34s} {'chars':>6s} {'difficulty':>10s} {'cheapest failed?':>16s}")
    for k in range(min(a.show, len(te))):
        print(f"        {ids[te[k]][:34]:34s} {int(length[k]):6d} {d[k]:10.2f} {'yes' if fail[k] else 'no':>16s}")
    print(f"\nSTEP 2  Spearman = correlation of the RANKS of length and difficulty (ties get average ranks)")
    print(f"        ranks of first 5 lengths:      {rankdata(length)[:5]}")
    print(f"        ranks of first 5 difficulties: {rankdata(d)[:5]}")
    print(f"        Spearman(length, difficulty) = {spearman(length, d):.3f}   <- 'spearman_vs_difficulty' for length")
    n1, n0 = int(fail.sum()), int((~fail).sum())
    print(f"\nSTEP 3  AUC: {n1} prompts the cheapest model failed x {n0} it passed = {n1 * n0} pairs")
    pos, neg = length[fail], length[~fail]
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    print(f"        pairs where the failed prompt is the longer one (ties count half): {wins:.1f} of {n1 * n0}")
    print(f"        AUC by counting pairs = {wins / (n1 * n0):.3f}")
    print(f"        AUC by simulate.py's formula = {auc(length, fail):.3f}   <- 'auc_cheap_fails' for length")


if __name__ == "__main__":
    main()
