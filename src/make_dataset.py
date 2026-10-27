"""Build a mixed-difficulty data.jsonl for complexity-based routing experiments.

Sources (all public):
  - GSM8K test            (grade-school math, easy)           -> numeric
  - MATH-500 (PRM800K)    (competition math, levels 1-5)      -> numeric answers only
  - MMLU-Pro (optional)   (10-option multiple choice, mixed)  -> mcq   [needs `pip install datasets`
                                                                       and internet access to Hugging Face]
  - AIME 2024 + 2025 (optional, --aime)  (hard competition math, integer answers 0-999) -> numeric  [HF]
  - GPQA Diamond (optional, --gpqa)      (graduate-level science, 4-option mcq)          -> mcq      [HF, gated:
        accept the terms at huggingface.co/datasets/Idavidrein/gpqa, then set HF_TOKEN=<your token>]

Each line: {"id","prompt","answer","type","source","level"}
`level` is the dataset's own difficulty label (GSM8K=0, MATH 1-5, MMLU-Pro=-1 (unlabeled)). It is NOT used by
routers; it is only for sanity checks and for plotting results by bucket.

Usage:
    python make_dataset.py --out data.jsonl --gsm8k 150 --math all --mmlu_pro 150
    # harder mixed set (easy -> very hard):
    python make_dataset.py --out data_hard.jsonl --gsm8k 50 --math all --mmlu_pro 200 --aime --gpqa
"""
import argparse
import json
import os
import random
import re
import urllib.request

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
MATH500_URL = "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/math_splits/test.jsonl"
NUM = re.compile(r"^-?\d+(\.\d+)?$")


def fetch_jsonl(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return [json.loads(l) for l in r.read().decode("utf-8").splitlines() if l.strip()]


def gsm8k(n, rng):
    rows = fetch_jsonl(GSM8K_URL)
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows[:n]):
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k-{i}", "prompt": r["question"], "answer": ans,
                    "type": "numeric", "source": "gsm8k", "level": 0})
    return out


def math500(n, rng):
    rows = fetch_jsonl(MATH500_URL)
    out = []
    for r in rows:
        a = str(r["answer"]).replace(",", "").replace("$", "").strip()
        if NUM.match(a):  # keep only plain numeric answers so scoring is exact and unambiguous
            out.append({"id": f"math-{r['unique_id'].replace('/', '_')}", "prompt": r["problem"],
                        "answer": a, "type": "numeric", "source": "math500", "level": int(r["level"])})
    rng.shuffle(out)
    return out if n is None else out[:n]


def mmlu_pro(n, rng):
    try:
        from datasets import load_dataset
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    except Exception as e:  # noqa: BLE001
        print(f"[skip] MMLU-Pro unavailable ({type(e).__name__}: {e})")
        return []
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx[:n]:
        r = ds[i]
        letters = "ABCDEFGHIJ"
        opts = "\n".join(f"{letters[k]}. {o}" for k, o in enumerate(r["options"]))
        out.append({"id": f"mmlupro-{r['question_id']}", "prompt": f"{r['question']}\n\n{opts}\n\nAnswer with the letter only.",
                    "answer": r["answer"], "type": "mcq", "source": "mmlu_pro", "level": -1})
    return out


def _first(r, keys):
    for k in keys:
        if k in r and r[k] is not None:
            return r[k]
    return None


def aime(rng):
    try:
        from datasets import load_dataset
    except ImportError:
        print("[skip] AIME needs `pip install datasets`")
        return []
    specs = [("Maxwell-Jia/AIME_2024", None, "aime24"),
             ("opencompass/AIME2025", "AIME2025-I", "aime25i"),
             ("opencompass/AIME2025", "AIME2025-II", "aime25ii")]
    out = []
    for repo, cfg, tag in specs:
        try:
            ds = None
            for split in ("train", "test"):
                try:
                    ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
                    break
                except Exception:  # noqa: BLE001
                    ds = None
            if ds is None:
                raise RuntimeError("no train/test split found")
            for k, r in enumerate(ds):
                q = _first(r, ["Problem", "problem", "question"])
                ans = re.sub(r"[^\d.\-]", "", str(_first(r, ["Answer", "answer"])))
                if q and NUM.match(ans):
                    out.append({"id": f"aime-{tag}-{k}", "prompt": q, "answer": ans, "type": "numeric",
                                "source": "aime", "level": 0})
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {repo} unavailable ({type(e).__name__}: {e})")
    rng.shuffle(out)
    return out


def gpqa(rng):
    try:
        from datasets import load_dataset
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=os.environ.get("HF_TOKEN"))
    except Exception as e:  # noqa: BLE001
        print(f"[skip] GPQA unavailable ({type(e).__name__}: {e}). It is gated: accept the terms on its Hugging Face "
              f"page and set HF_TOKEN.")
        return []
    letters, out = "ABCD", []
    for k, r in enumerate(ds):
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        text = "\n".join(f"{letters[j]}. {opts[o]}" for j, o in enumerate(order))
        out.append({"id": f"gpqa-{k}", "prompt": f"{r['Question']}\n\n{text}\n\nAnswer with the letter only.",
                    "answer": letters[order.index(0)], "type": "mcq", "source": "gpqa", "level": 0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data.jsonl")
    ap.add_argument("--gsm8k", type=int, default=150)
    ap.add_argument("--math", default="all", help="number of MATH-500 numeric items, or 'all'")
    ap.add_argument("--mmlu_pro", type=int, default=0)
    ap.add_argument("--aime", nargs="?", type=int, const=-1, default=0,
                    help="add AIME 2024+2025 (needs HF `datasets`): number of items, or bare flag for all")
    ap.add_argument("--gpqa", nargs="?", type=int, const=-1, default=0,
                    help="add GPQA Diamond (gated; needs HF_TOKEN): number of items, or bare flag for all")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    items = gsm8k(a.gsm8k, rng) if a.gsm8k else []
    items += math500(None if a.math == "all" else int(a.math), rng)
    if a.mmlu_pro:
        items += mmlu_pro(a.mmlu_pro, rng)
    if a.aime:
        x = aime(rng)
        items += x if a.aime < 0 else x[:a.aime]
    if a.gpqa:
        x = gpqa(rng)
        rng.shuffle(x)
        items += x if a.gpqa < 0 else x[:a.gpqa]
    rng.shuffle(items)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    from collections import Counter
    print(f"wrote {len(items)} items to {a.out}")
    print(Counter((it["source"], it["level"]) for it in items))


if __name__ == "__main__":
    main()
