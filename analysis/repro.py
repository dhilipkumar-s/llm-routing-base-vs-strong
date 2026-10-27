"""Reproduce the 8-split table of the paper (95% target, full set) from the cached answers. Run from this folder:
    python repro.py
Expected (approximately): self-consistency 50.6 / -3.4 / 8 of 8 / +3.3; disagreement 48.7 / -4.3 / +2.1; learned combiner 49.0 / -4.1 / +2.4."""
from core import *
import pandas as pd
W = World()
W.add_ext("combo_120b", "combo_120b.jsonl")      # 'Learned combiner' in the paper (gpt-oss-120b + gpt-oss-20b features, no judge)
W.add_ext("combo_nojudge", "combo_nojudge.jsonl")
names = ["length", "heuristic", "tfidf_ridge", "judge", "disagree", "selfcons3", "combo_120b", "combo_sc", "combo_sc_cat", "oracle_pool", "oracle_pair", "random"]
rows = []
for seed in range(1, 9):
    res = run_split(W, seed, names)
    for nm, r in res.items():
        m = summarize(W, r, r["te"]); m.update(router=nm, seed=seed, held=m["retention"] >= 95); rows.append(m)
df = pd.DataFrame(rows)
g = df.groupby("router").agg(sav=("savings", "mean"), sav_sd=("savings", "std"), acc=("acc_diff", "mean"), ret=("retention", "mean"),
                             held=("held", "sum"), lift=("lift", "mean"), lift_sd=("lift", "std")).round(2)
print(g.to_string())
