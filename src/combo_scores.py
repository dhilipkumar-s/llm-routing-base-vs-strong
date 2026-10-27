"""Graded combined complexity score from cheap signals (disagreement + judge + effort + prompt features).

A gradient-boosted model predicts empirical difficulty d (1 - fraction of pool models correct) from features that
are all available at routing time (never the gold answer or any large-model output). Scores are 5-fold
OUT-OF-FOLD, so no prompt's score comes from a model that saw its own label. Overhead = the cheap calls it needs.

    python combo_scores.py --data data_final.jsonl --responses responses_final.jsonl --judge judge.jsonl --out combo.jsonl
    python simulate.py --data data_final.jsonl --responses responses_final.jsonl --outdir results_combo --ladder <ladder B> --scores_file combo.jsonl,disagree.jsonl
    python simulate.py ... --sources mmlu_pro --outdir results_combo_mmlu --scores_file combo.jsonl,disagree.jsonl

Pass --no_judge to skip the judge (cheaper). Caveat: the model is fit on labels from the same dataset (cross-fitted);
report it as "a learned combiner trained in-distribution".
"""
import argparse, json, math
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from simulate import PRICE, canon, is_correct, MODELS


def cost_of(r):
    if r.get("cost") is not None:
        return float(r["cost"])
    pin, pout = PRICE[r["model"]]
    return (r.get("prompt_tokens", 0) * pin + r.get("completion_tokens", 0) * pout) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--responses", required=True)
    ap.add_argument("--judge", default=None); ap.add_argument("--no_judge", action="store_true")
    ap.add_argument("--cheap", default="meta-llama/llama-3.1-8b-instruct,openai/gpt-oss-20b")
    ap.add_argument("--samples", default=None, help="samples file from collect_samples.py: adds self-consistency features")
    ap.add_argument("--sc_model", default=None, help="model the samples came from (default: first of --cheap)")
    ap.add_argument("--category", action="store_true", help="add one-hot subject (needs data with a category field)")
    ap.add_argument("--out", default="combo.jsonl"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    c1, c2 = a.cheap.split(",")[:2]
    items = {}
    for l in open(a.data, encoding="utf-8"):
        if l.strip():
            it = json.loads(l); items[it["id"]] = it
    recs = {}
    for l in open(a.responses, encoding="utf-8"):
        r = json.loads(l)
        if not r.get("error"):
            recs[(r["model"], r["id"])] = r
    from collections import defaultdict
    samp = defaultdict(list)
    if a.samples:
        for l in open(a.samples, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("error") and r["model"] == (a.sc_model or c1):
                samp[r["id"]] = [x for x in samp[r["id"]] if x["k"] != r["k"]] + [r]  # one per (id, k)
    cats = sorted({it.get("category", "none") for it in items.values()}) if a.category else []
    judge = {}
    if a.judge and not a.no_judge:
        for l in open(a.judge, encoding="utf-8"):
            j = json.loads(l); judge[j["id"]] = j
    pool = [m["slug"] for m in MODELS]
    ids, X, y, over = [], [], [], []
    for i, it in items.items():
        r1, r2 = recs.get((c1, i)), recs.get((c2, i))
        if r1 is None or r2 is None or (not a.no_judge and a.judge and i not in judge):
            continue
        ok = [is_correct(it, recs[(m, i)]["text"]) for m in pool if (m, i) in recs]
        if len(ok) < 3:
            continue
        a1, a2 = canon(it, r1["text"]), canon(it, r2["text"])
        feats = [0.0 if (a1 is not None and a1 == a2) else 1.0,
                 float(a1 is None), float(a2 is None),
                 math.log1p(r1.get("completion_tokens", 0)), math.log1p(r2.get("completion_tokens", 0)),
                 math.log1p(len(it["prompt"])), float(it["type"] == "mcq")]
        o = cost_of(r1) + cost_of(r2)
        if a.samples:
            if not samp[i]:
                continue
            base = recs.get((a.sc_model or c1, i))
            vs = [canon(it, r["text"]) for r in [base] + samp[i]]
            vs = [v if v is not None else f"__u{k}" for k, v in enumerate(vs)]
            from collections import Counter
            feats += [1 - Counter(vs).most_common(1)[0][1] / len(vs), float(len(set(vs)))]
            o += sum((s.get("cost") or 0.0) for s in samp[i])
        if a.category:
            feats += [float(it.get("category", "none") == k) for k in cats]
        if judge:
            feats.append(judge[i]["score"]); o += judge[i]["cost"]
        ids.append(i); X.append(feats); y.append(1.0 - float(np.mean(ok))); over.append(o)
    X, y = np.array(X), np.array(y)
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=a.seed).split(X):
        m = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=a.seed)
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    with open(a.out, "w", encoding="utf-8") as f:
        for i, p, o in zip(ids, pred, over):
            f.write(json.dumps({"id": i, "score": float(p), "cost": float(o), "reuse": c1}) + "\n")
    from scipy.stats import spearmanr
    print(f"wrote {len(ids)} OOF scores to {a.out}; OOF spearman vs difficulty {spearmanr(pred, y)[0]:.3f}; "
          f"mean overhead ${np.mean(over):.6f}/prompt")


if __name__ == "__main__":
    main()
