"""Repeat the routing study for several (base, strong) pairs using ONLY answers you already cached (no API calls).

    python pair_study.py --data data_final.jsonl --responses responses_final.jsonl --outroot pairs --seeds 1,2,3,4,5,6,7,8 \
        --selfcons selfcons_k2.jsonl

For each pair it
  1. builds a 'disagreement' score from the base model and a partner model (cached answers; overhead charged),
  2. runs simulate.py over the seeds with the built-in scorers (length, heuristic features, TF-IDF, random, oracle),
  3. adds self-consistency when the base is openai/gpt-oss-120b and --selfcons is given (samples exist only for that model),
  4. writes <outroot>/pairs_summary.csv and pairs_summary.md with savings, accuracy difference, retention,
     splits at target and lift over a random mix, at the 95% and 99% targets.

Lift = accuracy above the straight line between always-base and always-strong at the router's own cost, per split, then averaged.
"""
import argparse
import glob
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M = {
    "oss120": "openai/gpt-oss-120b",
    "oss20": "openai/gpt-oss-20b",
    "llama70": "meta-llama/llama-3.3-70b-instruct",
    "llama8": "meta-llama/llama-3.1-8b-instruct",
    "qwen235": "qwen/qwen3-235b-a22b-2507",
    "sonnet": "anthropic/claude-sonnet-5.5",
    "opus": "anthropic/claude-opus-5.5",
}
# (name, base, strong, partner for the disagreement probe)
PAIRS = [
    ("oss120->sonnet", "oss120", "sonnet", "oss20"),   # the pair used in the paper
    ("oss120->opus", "oss120", "opus", "oss20"),
    ("llama70->sonnet", "llama70", "sonnet", "llama8"),
    ("qwen235->sonnet", "qwen235", "sonnet", "oss120"),
    ("qwen235->opus", "qwen235", "opus", "oss120"),
    ("oss20->sonnet", "oss20", "sonnet", "llama8"),
]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("failed: " + " ".join(cmd) + "\n" + (r.stdout + r.stderr)[-1500:])
    return r.stdout


def lift_rows(outdir, base, strong):
    out = []
    for f in sorted(glob.glob(os.path.join(outdir, "seed_*", "summary.csv"))):
        d = pd.read_csv(f).set_index("strategy")
        ab, cb = d.loc[f"always:{base}", ["accuracy", "usd_per_100"]]
        as_, cs = d.loc[f"always:{strong}", ["accuracy", "usd_per_100"]]
        for s, r in d.iterrows():
            if "@" not in s:
                continue
            line = ab + (r.usd_per_100 - cb) / (cs - cb) * (as_ - ab)
            tgt = 99 if s.endswith("@99%") else 95
            out.append({"router": s.split("@")[0].replace("ext:", ""), "target": tgt,
                        "savings": r.savings_vs_frontier_pct, "acc_diff_pp": r.acc_diff_vs_ref_pp,
                        "retention": r.quality_retention_pct, "held": r.quality_retention_pct >= tgt,
                        "lift_pp": 100 * (r.accuracy - line), "seed": os.path.basename(os.path.dirname(f))})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--outroot", default="pairs")
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--selfcons", default=None, help="selfcons JSONL for gpt-oss-120b (3 votes), optional")
    ap.add_argument("--pairs", default=None, help="comma-separated pair names to run (default: all)")
    a = ap.parse_args()
    os.makedirs(a.outroot, exist_ok=True)
    keep = set(a.pairs.split(",")) if a.pairs else None
    frames = []
    for name, b, s, p in PAIRS:
        if keep and name not in keep:
            continue
        tag = name.replace("->", "_to_")
        dis = os.path.join(a.outroot, f"disagree_{tag}.jsonl")
        run([sys.executable, os.path.join(HERE, "make_signal_scores.py"), "--data", a.data, "--responses", a.responses,
             "--signal", "disagree", "--models", f"{M[b]},{M[p]}", "--out", dis])
        scores = [dis]
        if a.selfcons and b == "oss120":
            scores.append(a.selfcons)
        out = os.path.join(a.outroot, tag)
        run([sys.executable, os.path.join(HERE, "run_seeds.py"), "--seeds", a.seeds, "--outroot", out, "--",
             "--data", a.data, "--responses", a.responses, "--ladder", f"{M[b]},{M[s]}",
             "--scores_file", ",".join(scores), "--credit_reuse"])
        d = lift_rows(out, M[b], M[s])
        d["pair"] = name
        frames.append(d)
        print("done", name)
    df = pd.concat(frames)
    g = df.groupby(["pair", "target", "router"]).agg(
        savings=("savings", "mean"), savings_sd=("savings", "std"), acc_diff_pp=("acc_diff_pp", "mean"),
        retention=("retention", "mean"), splits_at_target=("held", "sum"), n=("held", "size"),
        lift_pp=("lift_pp", "mean"), lift_sd=("lift_pp", "std")).round(2).reset_index()
    g.to_csv(os.path.join(a.outroot, "pairs_summary.csv"), index=False)
    with open(os.path.join(a.outroot, "pairs_summary.md"), "w") as f:
        for t in (95, 99):
            f.write(f"\n## {t}% target\n\n")
            f.write(g[g.target == t].drop(columns="target").to_markdown(index=False) + "\n")
    print(g[g.target == 95].to_string(index=False))


if __name__ == "__main__":
    main()
