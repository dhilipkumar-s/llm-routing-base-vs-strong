"""Embedding-based learned router signal (in the spirit of RouteLLM's classifier routers, NOT RouteLLM itself).

Embeds each prompt with a small sentence-transformer, fits a logistic regression that predicts 'gpt-oss-120b answers wrongly' on the
OLD prompt set (out-of-fold scores for old prompts, final model for fresh prompts) and writes score files.

    pip install sentence-transformers scikit-learn
    python embed_scores.py --old data_final.jsonl --old_responses responses_final.jsonl --fresh data_fresh.jsonl \
        --out_old embed_old.jsonl --out_fresh embed_fresh.jsonl
"""
import argparse, json, numpy as np
from simulate import load

ap = argparse.ArgumentParser()
ap.add_argument("--old", required=True); ap.add_argument("--old_responses", required=True); ap.add_argument("--fresh", required=True)
ap.add_argument("--out_old", required=True); ap.add_argument("--out_fresh", required=True); ap.add_argument("--model", default="all-MiniLM-L6-v2")
a = ap.parse_args()
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
items, ids, names, OK, USD, CN = load(a.old, a.old_responses, 0.9)
y = 1 - OK[:, names.index("openai/gpt-oss-120b")]
enc = SentenceTransformer(a.model)
Xo = enc.encode([items[i]["prompt"] for i in ids], batch_size=32, normalize_embeddings=True)
fr = [json.loads(l) for l in open(a.fresh, encoding="utf-8") if l.strip()]
Xf = enc.encode([r["prompt"] for r in fr], batch_size=32, normalize_embeddings=True)
oof = np.zeros(len(ids))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Xo, y):
    oof[va] = LogisticRegression(C=1.0, max_iter=2000).fit(Xo[tr], y[tr]).predict_proba(Xo[va])[:, 1]
sf = LogisticRegression(C=1.0, max_iter=2000).fit(Xo, y).predict_proba(Xf)[:, 1]
with open(a.out_old, "w") as f:
    for i, s in zip(ids, oof): f.write(json.dumps({"id": i, "score": float(s), "cost": 0.0}) + "\n")
with open(a.out_fresh, "w") as f:
    for r, s in zip(fr, sf): f.write(json.dumps({"id": r["id"], "score": float(s), "cost": 0.0}) + "\n")
print("wrote", a.out_old, a.out_fresh)
