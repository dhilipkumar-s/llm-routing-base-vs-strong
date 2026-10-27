"""Add the MMLU-Pro subject ("category") to your existing data file, matching on id. Needs `pip install datasets`.

    python add_category.py --data data_final.jsonl --out data_final_cat.jsonl
Then use data_final_cat.jsonl with combo_scores.py --category (ids are unchanged, so responses files still match).
"""
import argparse, json
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
cat = {f"mmlupro-{r['question_id']}": r["category"] for r in ds}
n = 0
with open(a.out, "w", encoding="utf-8") as f:
    for l in open(a.data, encoding="utf-8"):
        if not l.strip():
            continue
        it = json.loads(l)
        if it["id"] in cat:
            it["category"] = cat[it["id"]]; n += 1
        f.write(json.dumps(it) + "\n")
print(f"added category to {n} items -> {a.out}")
