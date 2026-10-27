"""Build a FRESH prompt set that shares no prompt text with data_final.jsonl (needs internet + `pip install datasets`).

    python build_fresh.py --old data_final.jsonl --out data_fresh.jsonl --mmlu_pro 300 --gsm8k 100 --seed 2026

MATH-500 numeric items and AIME problems not used before are taken in full (there are not many); MMLU-Pro and GSM8K are drawn
at random from the unused rest. Ids get an 'f-' prefix so they never collide with the old ones. Reuses make_dataset.py from the
main folder (put this file next to it). Prints the mix; expect roughly 300 MMLU-Pro, 100 GSM8K, ~100-150 MATH, ~30 AIME.
"""
import argparse, json, random
from collections import Counter
import make_dataset as M

ap = argparse.ArgumentParser()
ap.add_argument("--old", required=True); ap.add_argument("--out", default="data_fresh.jsonl")
ap.add_argument("--mmlu_pro", type=int, default=300); ap.add_argument("--gsm8k", type=int, default=100)
ap.add_argument("--seed", type=int, default=2026)
a = ap.parse_args()
old = {json.loads(l)["prompt"].strip() for l in open(a.old, encoding="utf-8") if l.strip()}
rng = random.Random(a.seed)
pool = M.gsm8k(2000, rng) + M.math500(None, rng) + M.mmlu_pro(4000, rng) + M.aime(rng)   # large pools, then remove used prompts
fresh = [it for it in pool if it["prompt"].strip() not in old]
by = {}
for it in fresh: by.setdefault(it["source"], []).append(it)
want = {"mmlu_pro": a.mmlu_pro, "gsm8k": a.gsm8k, "math500": 10**9, "aime": 10**9}
out = []
for s, l in by.items():
    rng.shuffle(l); out += l[:want.get(s, 0)]
rng.shuffle(out)
for k, it in enumerate(out): it["id"] = f"f-{it['source']}-{k}"
with open(a.out, "w", encoding="utf-8") as f:
    for it in out: f.write(json.dumps(it) + "\n")
print(f"wrote {len(out)} fresh prompts ({len(pool) - len(fresh)} pool prompts removed as already used)")
print(Counter(it["source"] for it in out))
