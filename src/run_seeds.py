"""Repeat simulate.py over several train/test splits and report mean +- sd, so one lucky split doesn't decide.

    python run_seeds.py --seeds 1,2,3,4,5 --outroot seeds_2tier -- --data data_final.jsonl --responses responses_final.jsonl \
        --ladder openai/gpt-oss-120b,anthropic/claude-sonnet-5.5 --scores_file disagree_120b.jsonl,combo_120b.jsonl --credit_reuse

Everything after `--` is passed to simulate.py unchanged (do not pass --seed/--outdir). Writes <outroot>/seed_N/ and
<outroot>/seed_agg.csv. Adds --boot 200 unless you set --boot.
"""
import argparse, os, subprocess, sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="1,2,3,4,5")
ap.add_argument("--outroot", required=True)
a, rest = ap.parse_known_args()
rest = [x for x in rest if x != "--"]
if "--boot" not in rest:
    rest += ["--boot", "200"]
here = os.path.dirname(os.path.abspath(__file__))
frames = []
for sd in a.seeds.split(","):
    out = os.path.join(a.outroot, f"seed_{sd}")
    r = subprocess.run([sys.executable, os.path.join(here, "simulate.py"), *rest, "--seed", sd, "--outdir", out],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"seed {sd} failed:\n{r.stderr[-1500:]}")
    d = pd.read_csv(os.path.join(out, "summary.csv")); d["seed"] = int(sd); frames.append(d)
    print(f"seed {sd} done")
df = pd.concat(frames)
cols = ["accuracy", "usd_per_100", "savings_vs_frontier_pct", "quality_retention_pct", "acc_diff_vs_ref_pp"]
g = df.groupby("strategy")[cols].agg(["mean", "std"]).round(3)
g.columns = ["_".join(c) for c in g.columns]
g["n_seeds"] = df.groupby("strategy").size()
# how often does a 95%/99% operating point actually hold its quality target on TEST?
df["held"] = df.quality_retention_pct >= df.strategy.map(lambda s: 99 if s.endswith("@99%") else 95)
g["target_held_frac"] = df.groupby("strategy")["held"].mean().round(2)
os.makedirs(a.outroot, exist_ok=True)
g.to_csv(os.path.join(a.outroot, "seed_agg.csv"))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(g[g.index.str.contains("@|oracle_router|always")].to_string())
print(f"\nwrote {os.path.join(a.outroot, 'seed_agg.csv')}")
