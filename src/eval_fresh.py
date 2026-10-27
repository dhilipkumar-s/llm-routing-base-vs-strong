"""Frozen-configuration evaluation: choose every threshold on the OLD prompts only, then report ONCE on the FRESH prompts.

    python eval_fresh.py --old_data data_final.jsonl --old_resp responses_final.jsonl --fresh_data data_fresh.jsonl --fresh_resp responses_fresh.jsonl \
      --scorers selfcons3=selfcons_old.jsonl:selfcons_fresh.jsonl,disagree=dis_old.jsonl:dis_fresh.jsonl,judge=judge_old.jsonl:judge_fresh.jsonl,\
logprob=extra_old_logprob.jsonl:extra_fresh_logprob.jsonl,verbal=extra_old_verbal.jsonl:extra_fresh_verbal.jsonl,embed=embed_old.jsonl:embed_fresh.jsonl \
      --out fresh_results

Built-in scorers (no files): length, heuristic, tfidf (fit on old prompts), oracle_pair (ceiling), random. Rule: cheapest score cut-off whose OLD
accuracy is >= (target + margin) x Sonnet's old accuracy; the cut-off is in raw score units, so nothing is fitted on fresh prompts.
Reports savings, accuracy difference, retention, lift over a random mix at equal cost, with 95% bootstrap intervals over fresh prompts,
paired differences against self-consistency, and the share of... (single run: retention vs target is simply met or not).
Needs simulate.py/collect.py next to it.
"""
import argparse, json, math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate as S

BASE, STRONG = "openai/gpt-oss-120b", "anthropic/claude-sonnet-5.5"

def view(data, resp):
    items, ids, names, OK, USD, CN = S.load(data, resp, 0.9)
    b, s = names.index(BASE), names.index(STRONG)
    return dict(items=items, ids=ids, prompts=[items[i]["prompt"] for i in ids], okb=OK[:, b], oks=OK[:, s], Cb=USD[:, b], Cs=USD[:, s])

def read_scores(path, ids):
    rows = {}
    for l in open(path):
        if l.strip(): r = json.loads(l); rows[r["id"]] = r
    sc = np.array([rows[i]["score"] if i in rows else np.nan for i in ids], float)
    ov = np.array([rows[i].get("cost", 0.0) if i in rows else 0.0 for i in ids], float)
    return sc, ov, any(r.get("reuse") == BASE for r in rows.values())

def costs(V, esc, ov, reuse):
    return np.where(esc, V["Cs"], 0.0 if reuse else V["Cb"]) + ov

def pick(V, sc, ov, reuse, thr):
    cand = np.concatenate([[-np.inf], np.unique(sc[~np.isnan(sc)])[:-1]])      # escalate iff score > t
    best = None
    for t in cand:
        esc = np.nan_to_num(sc, nan=np.inf) > t
        acc = np.where(esc, V["oks"], V["okb"]).mean()
        if acc >= thr - 1e-12:
            c = costs(V, esc, ov, reuse).sum()
            if best is None or c < best[0] - 1e-15 or (abs(c - best[0]) < 1e-15 and acc > best[2]): best = (c, t, acc)
    return best[1] if best else -np.inf

def main():
    ap = argparse.ArgumentParser()
    for k in ("old_data", "old_resp", "fresh_data", "fresh_resp"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--scorers", default=""); ap.add_argument("--out", default="fresh_results")
    ap.add_argument("--target", type=float, default=0.95); ap.add_argument("--margins", default="0,0.02"); ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    O, F = view(a.old_data, a.old_resp), view(a.fresh_data, a.fresh_resp)
    print(f"old {len(O['ids'])} prompts, fresh {len(F['ids'])} prompts")
    sc = {}
    zo, zf = np.zeros(len(O["ids"])), np.zeros(len(F["ids"]))
    sc["length"] = (np.array([len(p) for p in O["prompts"]], float), np.array([len(p) for p in F["prompts"]], float), zo, zf, False)
    h = S.HeuristicScorer(); sc["heuristic"] = (h.score(None, O["prompts"]), h.score(None, F["prompts"]), zo, zf, False)
    # tfidf: difficulty target = old base failure share (pool share is not available for fresh); OOF on old, full fit for fresh
    fold = np.random.RandomState(3).permutation(len(O["ids"])) % 5; y = 1 - O["okb"]; oof = np.zeros(len(y))
    for f in range(5):
        va, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        m = S.TfidfRidgeScorer(); m.fit([O["prompts"][k] for k in tr], y[tr], None); oof[va] = m.score(None, [O["prompts"][k] for k in va])
    m = S.TfidfRidgeScorer(); m.fit(O["prompts"], y, None); sc["tfidf"] = (oof, m.score(None, F["prompts"]), zo, zf, False)
    gain = lambda V: ((V["okb"] == 0) & (V["oks"] == 1)).astype(float) + 1e-9 * np.arange(len(V["ids"]))
    sc["oracle_pair"] = (gain(O), gain(F), zo, zf, False)
    r = np.random.RandomState(1); sc["random"] = (r.rand(len(O["ids"])), r.rand(len(F["ids"])), zo, zf, False)
    for spec in filter(None, a.scorers.split(",")):
        name, files = spec.split("="); fo, ff = files.split(":")
        so, oo, ro = read_scores(fo, O["ids"]); sf, of, rf = read_scores(ff, F["ids"])
        miss = np.isnan(sf).mean()
        if miss > 0.05: print(f"[skip] {name}: {miss:.0%} of fresh scores missing"); continue
        so = np.where(np.isnan(so), np.nanmax(so), so)
        sc[name] = (so, sf, oo, of, ro or rf)
    ab, as_ = F["okb"].mean(), F["oks"].mean(); cb, cs = F["Cb"].mean() * 100, F["Cs"].mean() * 100
    rng = np.random.RandomState(0); idx = rng.randint(0, len(F["ids"]), (a.boot, len(F["ids"])))
    rows, per = [], {}
    for mg in [float(x) for x in a.margins.split(",")]:
        for name, (so, sf, oo, of, reuse) in sc.items():
            t = pick(O, so, oo, reuse, (a.target + mg) * O["oks"].mean())
            esc = np.nan_to_num(sf, nan=np.inf) > t
            acc = np.where(esc, F["oks"], F["okb"]); cost = costs(F, esc, of, reuse)
            def met(ix):
                A, C = acc[ix].mean(0) if ix.ndim > 1 else acc[ix].mean(), cost[ix].mean(0) if ix.ndim > 1 else cost[ix].mean()
                return A, C
            A = acc[idx].mean(1); C = cost[idx].mean(1) * 100; ab_ = F["okb"][idx].mean(1); as_b = F["oks"][idx].mean(1); cb_ = F["Cb"][idx].mean(1) * 100; cs_ = F["Cs"][idx].mean(1) * 100
            lift = 100 * (A - (ab_ + (C - cb_) / (cs_ - cb_) * (as_b - ab_))); sav = 100 * (1 - C / cs_)
            line = ab + (cost.mean() * 100 - cb) / (cs - cb) * (as_ - ab)
            per[(mg, name)] = lift
            rows.append(dict(margin=mg, router=name, escalated=100 * esc.mean(), savings=100 * (1 - cost.sum() / F["Cs"].sum()), sav_lo=np.percentile(sav, 2.5), sav_hi=np.percentile(sav, 97.5),
                             acc_diff_pp=100 * (acc.mean() - as_), retention=100 * acc.mean() / as_, met_target=acc.mean() / as_ >= a.target,
                             lift_pp=100 * (acc.mean() - line), lift_lo=np.percentile(lift, 2.5), lift_hi=np.percentile(lift, 97.5)))
    df = pd.DataFrame(rows).round(2); df.to_csv(os.path.join(a.out, "fresh_summary.csv"), index=False)
    pd.set_option("display.width", 220); print(df.to_string(index=False))
    ref = "selfcons3" if "selfcons3" in sc else None
    if ref:
        print(f"\npaired lift differences vs {ref} (fresh bootstrap)")
        for (mg, name), l in per.items():
            if name != ref and name not in ("random", "oracle_pair"):
                d = per[(mg, ref)] - l; print(f"margin {mg}: {ref} - {name}: {d.mean():+.2f} [{np.percentile(d,2.5):+.2f}, {np.percentile(d,97.5):+.2f}]")
if __name__ == "__main__": main()
