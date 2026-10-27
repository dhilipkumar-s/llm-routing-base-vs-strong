"""Offline evaluation of complexity-based LLM routers on cached responses (no new API calls).

    python simulate.py --data data.jsonl --responses responses.jsonl --outdir results

What it does
------------
1. Loads every model's cached answer to every prompt, scores correctness, and uses the cost OpenRouter reported
   (falls back to token counts x the prices in collect.py).
2. Defines a 4-tier "ladder" (one model per tier: cheap -> frontier). Override with --ladder a,b,c,d.
3. Splits prompts into TRAIN / TEST (stratified by dataset bucket). Everything that is fitted or tuned
   (scorers, threshold choice) uses TRAIN only. All reported numbers are on TEST.
4. Complexity scorers map a prompt to a number (higher = more complex):
      length            prompt character count
      heuristic         hand-set rules (length, math symbols, keywords) - NOT tuned
      tfidf_ridge       learned: TF-IDF + ridge regression on empirical difficulty (needs scikit-learn)
      oracle_difficulty upper bound: uses the true empirical difficulty (cheating on purpose)
      random            lower bound: same tier mix, no information
      ext:<name>        scores you computed elsewhere (e.g. an LLM judge): --scores_file my_scores.jsonl
                        lines: {"id": ..., "score": 0.7, "cost": 0.00002 (optional router overhead, USD)}
5. A score is turned into a tier with 3 thresholds on its train percentile. All threshold triples on a grid are
   evaluated; the train-Pareto set forms the router's cost-quality curve (shown with its TEST performance).
   Operating points ("cheapest config keeping >=95% / 99% of frontier-model accuracy on TRAIN") are reported
   on TEST with 95% bootstrap CIs.

Outputs (in --outdir): summary.csv, quality.csv, frontier.csv, buckets.csv, cost_quality.png, bucket_mix.png
"""
import argparse
import csv
import itertools
import json
import math
import os
import re
import time
from collections import Counter, OrderedDict, defaultdict

import numpy as np

from collect import MODELS

PRICE = {m["slug"]: (m["in"], m["out"]) for m in MODELS}
TIER_OF = {m["slug"]: m["tier"] for m in MODELS}
DEFAULT_LADDER = ["meta-llama/llama-3.1-8b-instruct", "openai/gpt-oss-120b",
                  "anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5"]
GRID = np.round(np.linspace(0, 1, 11), 2)


# ======================= correctness =======================
def extract(text):
    t = text or ""
    m = re.findall(r"final answer:\s*(.+)", t, flags=re.I)
    if m:
        return m[-1].strip()
    b = re.findall(r"\\boxed\{([^{}]*)\}", t)
    if b:
        return b[-1].strip()
    lines = [l for l in t.strip().split("\n") if l.strip()]
    return lines[-1].strip() if lines else ""


def parse_num(s):
    s = s.replace(",", "").replace("$", "").replace("\\!", "").replace("\\,", "")
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    m = re.search(r"\\d?frac\{(-?\d+\.?\d*)\}\{(-?\d+\.?\d*)\}", s) or re.search(r"(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*)", s)
    if m:
        d = float(m.group(2))
        return float(m.group(1)) / d if d else None
    n = re.findall(r"-?\d+\.?\d*", s)
    return float(n[-1]) if n else None


def mcq_letter(text):
    """Letter chosen by the model. Priority: 'Final answer: X' > 'answer is X' > \\boxed{X} > a last line that is
    just the letter. (No generic 'first capital letter' search: it would misread 'I think ...' as option I.)"""
    t = text or ""
    for p in (r"(?i:final answer)\s*:\s*\(?\**\s*([A-J])\b",
              r"(?i:answer)\s*(?:is|:)\s*\(?\**\s*([A-J])\b",
              r"\\boxed\{\s*\(?(?:\\text\{)?([A-J])\b"):
        m = re.findall(p, t)
        if m:
            return m[-1]
    lines = [l for l in t.strip().split("\n") if l.strip()]
    if lines:
        m = re.match(r"^\W*([A-Ja-j])\W*$", lines[-1]) or re.match(r"^\W*([A-J])[.)]\s", lines[-1])
        if m:
            return m.group(1).upper()
    return None


def canon(item, text):
    """Canonical answer (used for correctness and for cascade agreement)."""
    if item.get("type") == "mcq":
        return mcq_letter(text)
    v = parse_num(extract(text))
    return None if v is None else round(v, 6)


def is_correct(item, text):
    c = canon(item, text)
    if c is None:
        return 0.0
    if item.get("type") == "mcq":
        return float(c == item["answer"].strip().upper())
    a = parse_num(str(item["answer"]))
    return float(a is not None and math.isclose(c, a, rel_tol=1e-6, abs_tol=1e-6))


def bucket(it):
    src, lvl = it.get("source", "data"), it.get("level")
    return f"{src}-L{lvl}" if isinstance(lvl, int) and lvl > 0 else src


# ======================= loading =======================
def load(data_path, resp_path, min_cov):
    items = OrderedDict()
    for l in open(data_path, encoding="utf-8"):
        if l.strip():
            it = json.loads(l)
            items[it["id"]] = it
    recs = defaultdict(dict)
    for l in open(resp_path, encoding="utf-8"):
        r = json.loads(l)
        if r.get("error") or r["id"] not in items or r["model"] not in PRICE:
            continue
        recs[r["model"]][r["id"]] = r
    names = []
    for m in MODELS:
        cov = len(recs[m["slug"]]) / max(1, len(items))
        if cov >= min_cov:
            names.append(m["slug"])
        else:
            print(f"[warn] dropping {m['slug']}: only {cov:.0%} of prompts answered (bad slug or failed calls?)")
    ids = [i for i in items if all(i in recs[s] for s in names)]
    if len(ids) < 30:
        raise SystemExit(f"Only {len(ids)} prompts have answers from all kept models. Re-run collect.py first.")
    ok = np.zeros((len(ids), len(names)))
    usd = np.zeros_like(ok)
    cn = np.empty(ok.shape, dtype=object)
    for j, s in enumerate(names):
        for k, i in enumerate(ids):
            r = recs[s][i]
            ok[k, j] = is_correct(items[i], r["text"])
            cn[k, j] = canon(items[i], r["text"])
            usd[k, j] = r["cost"] if r.get("cost") is not None else (
                r["prompt_tokens"] * PRICE[s][0] + r["completion_tokens"] * PRICE[s][1]) / 1e6
    for j, s in enumerate(names):
        if usd[:, j].sum() == 0:
            print(f"[warn] {s} has zero cost everywhere: fill its price in collect.py or cost comparisons are wrong")
    return items, ids, names, ok, usd, cn


def resolve_ladder(names, override):
    if override:
        lad = override.split(",")
        bad = [s for s in lad if s not in names]
        if len(lad) not in (2, 3, 4) or bad:
            raise SystemExit(f"--ladder needs 2-4 comma-separated slugs among {names}; problem: {bad}")
        # Internally there are always 4 tiers. A shorter ladder is padded by repeating its CHEAPEST middle model, so
        # e.g. "A,B" becomes A,A,A,B and "A,B,C" becomes A,B,B,C. Repeated tiers are identical models, so this
        # is exactly a 2-/3-tier router; the mix_t* columns of repeated tiers just add up. cascade_agree is
        # meaningless with repeated tiers (adjacent identical models always agree): ignore it.
        if len(lad) == 2:
            lad = [lad[0], lad[0], lad[0], lad[1]]
        elif len(lad) == 3:
            lad = [lad[0], lad[1], lad[1], lad[2]]
        return lad
    lad = []
    for t in range(4):
        if DEFAULT_LADDER[t] in names:
            lad.append(DEFAULT_LADDER[t])
            continue
        alt = [m["slug"] for m in MODELS if m["tier"] == t and m["slug"] in names]
        if not alt:
            raise SystemExit(f"No available model for tier {t}. Available: {names}")
        print(f"[warn] default ladder model {DEFAULT_LADDER[t]} unavailable; using {alt[0]} for tier {t}")
        lad.append(alt[0])
    return lad


# ======================= complexity scorers =======================
class Scorer:
    name = "base"
    overhead = None  # optional dict id -> USD

    def fit(self, prompts, d, m):
        pass

    def score(self, ids, prompts):
        raise NotImplementedError


class LengthScorer(Scorer):
    name = "length"

    def score(self, ids, prompts):
        return np.array([len(p) for p in prompts], float)


KW = ["prove", "find all", "determine", "maximum", "minimum", "integer", "polynomial", "probability", "remainder",
      "sum of", "real", "complex", "triangle", "sequence", "how many ways", "expected", "largest", "smallest",
      "modulo", "divisible", "function", "positive", "distinct", "prime"]


class HeuristicScorer(Scorer):
    """Hand-set rule score. Weights are NOT tuned on data; edit freely or replace with your own idea."""
    name = "heuristic"

    def score(self, ids, prompts):
        out = []
        for p in prompts:
            low = p.lower()
            out.append(0.8 * math.log1p(len(p)) + 0.15 * len(re.findall(r"\\[a-zA-Z]+", p))
                       + 0.05 * len(re.findall(r"[\^_=+*/{}]", p)) + 0.4 * sum(k in low for k in KW)
                       + 0.1 * len(re.findall(r"\d+", p)))
        return np.array(out)


class TfidfRidgeScorer(Scorer):
    """Learned complexity: TF-IDF (+log length) -> ridge regression onto empirical difficulty (train only)."""
    name = "tfidf_ridge"
    learned = True

    def _x(self, prompts, fit):
        from scipy.sparse import csr_matrix, hstack
        X = self.v.fit_transform(prompts) if fit else self.v.transform(prompts)
        ln = csr_matrix(np.array([[math.log1p(len(p)) / 10] for p in prompts]))
        return hstack([X, ln]).tocsr()

    def fit(self, prompts, d, m):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        self.v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
        self.r = Ridge(alpha=1.0).fit(self._x(prompts, True), d)

    def score(self, ids, prompts):
        return self.r.predict(self._x(prompts, False))


class OracleDifficulty(Scorer):
    name = "oracle_difficulty"

    def __init__(self, d_by_id):
        self.d = d_by_id

    def score(self, ids, prompts):
        return np.array([self.d[i] for i in ids]) + 1e-9 * np.arange(len(ids))


class BucketLabelScorer(Scorer):
    """Reference, NOT deployable: complexity = the dataset's own difficulty bucket (e.g. MATH level, GSM8K), scored
    by that bucket's mean empirical difficulty on TRAIN (prompt length breaks ties). Shows what coarse
    human-labelled difficulty is worth, between a weak scorer and the oracle."""
    name = "label_bucket"
    learned = True

    def __init__(self, bucket_by_id):
        self.b = bucket_by_id

    def fit(self, prompts, d, m):
        acc = defaultdict(list)
        for i, v in zip(self.train_ids, d):
            acc[self.b[i]].append(v)
        self.mean = {k: float(np.mean(v)) for k, v in acc.items()}
        self.glob = float(np.mean(d))

    def score(self, ids, prompts):
        return np.array([self.mean.get(self.b[i], self.glob) + 1e-6 * math.log1p(len(p)) for i, p in zip(ids, prompts)])


class RandomScorer(Scorer):
    name = "random"

    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def score(self, ids, prompts):
        return self.rng.rand(len(ids))


class ExternalScorer(Scorer):
    def __init__(self, path):
        self.name = "ext:" + os.path.splitext(os.path.basename(path))[0]
        self.s, self.overhead, self.reuse = {}, {}, None
        for l in open(path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                self.reuse = r.get("reuse", self.reuse)
                self.s[r["id"]] = float(r["score"])
                self.overhead[r["id"]] = float(r.get("cost", 0.0))

    def score(self, ids, prompts):
        miss = [i for i in ids if i not in self.s]
        if miss:
            raise SystemExit(f"{self.name}: missing scores for {len(miss)} ids, e.g. {miss[:3]}")
        return np.array([self.s[i] for i in ids])


# ======================= statistics helpers =======================
def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def auc(score, positive):
    pos = np.asarray(positive, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rankdata(score)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pareto(cost, acc):
    order = np.lexsort((-acc, cost))
    keep, best = [], -1.0
    for k in order:
        if acc[k] > best + 1e-12:
            keep.append(k)
            best = acc[k]
    return np.array(keep, int)


def auqc(x, y):
    """Area under the quality-vs-cost step curve. x = normalised cost (0=cheap tier, 1=frontier tier),
    y = performance-gap-recovered. 1.0 = frontier quality at cheap-tier cost; ~0.5 = no better than mixing."""
    x = np.clip(np.concatenate([[0.0], x]), 0, 1)
    y = np.concatenate([[0.0], y])
    o = np.argsort(x, kind="mergesort")
    x, y = x[o], np.maximum.accumulate(y[o])
    x = np.append(x, 1.0)
    return float(np.sum(y * np.diff(x)))


def to_pct(train_scores, scores):
    ts = np.sort(train_scores)
    return (np.searchsorted(ts, scores, side="right") + 1) / (len(ts) + 2)


def make_cfgs():
    return np.array([t for t in itertools.combinations_with_replacement(GRID, 3)])  # a<=b<=c


def assign(pct, T):
    return ((pct[None, :] > T[:, 0:1]).astype(int) + (pct[None, :] > T[:, 1:2]) + (pct[None, :] > T[:, 2:3]))


def cfg_items(tiers, A, C, ov, reuse_mask=None):
    """Cost of a routing config. reuse_mask (bool per tier): the scorer's probe already produced this tier's answer
    and its price is inside the overhead, so the tier's own cost is not charged a second time."""
    n = A.shape[0]
    idx = np.arange(n)[None, :]
    c = C[idx, tiers]
    cost = c + ov[None, :]
    if reuse_mask is not None:
        cost = cost - c * reuse_mask[tiers]
    return A[idx, tiers], cost


# ======================= main =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--responses", default="responses.jsonl")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--test_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--credit_reuse", action="store_true",
                    help="if a scorer's file has a 'reuse' model slug, do not charge that model's answer twice when "
                         "the router keeps the prompt on it (its price is already in the probe overhead)")
    ap.add_argument("--ladder", default=None)
    ap.add_argument("--scores_file", default=None, help="comma-separated JSONL files of external complexity scores")
    ap.add_argument("--targets", default="0.95,0.99")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min_coverage", type=float, default=0.9)
    ap.add_argument("--sources", default=None, help="comma-separated dataset sources to keep, e.g. mmlu_pro")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    targets = [float(t) for t in a.targets.split(",")]

    items, ids, names, OK, USD, CN = load(a.data, a.responses, a.min_coverage)
    if a.sources:  # evaluate within chosen dataset sources only (removes "which dataset is this?" as a shortcut)
        want = set(a.sources.split(","))
        keep = [k for k, i in enumerate(ids) if items[i].get("source") in want]
        if len(keep) < 30:
            raise SystemExit(f"--sources {a.sources}: only {len(keep)} prompts match. Sources present: "
                             f"{sorted({it.get('source') for it in items.values()})}")
        ids, OK, USD, CN = [ids[k] for k in keep], OK[keep], USD[keep], CN[keep]
    lad = resolve_ladder(names, a.ladder)
    li = [names.index(s) for s in lad]
    A_all, C_all = OK[:, li], USD[:, li]
    n = len(ids)
    d_all = 1 - OK.mean(1)  # empirical difficulty: share of pool models that fail
    m_all = np.array([next((t for t in range(4) if A_all[k, t] == 1), 4) for k in range(n)])  # min sufficient tier
    print(f"{n} prompts, {len(names)} models. Ladder: " + " < ".join(lad))

    # ---- stratified train/test split
    rng = np.random.RandomState(a.seed)
    strata = defaultdict(list)
    for k, i in enumerate(ids):
        strata[bucket(items[i])].append(k)
    tr, te = [], []
    for b in sorted(strata):
        idx = list(strata[b])
        rng.shuffle(idx)
        nt = int(round(len(idx) * a.test_frac))
        te += idx[:nt]
        tr += idx[nt:]
    tr, te = np.array(sorted(tr)), np.array(sorted(te))
    ids_tr, ids_te = [ids[k] for k in tr], [ids[k] for k in te]
    P_tr, P_te = [items[i]["prompt"] for i in ids_tr], [items[i]["prompt"] for i in ids_te]
    A_tr, C_tr, A_te, C_te = A_all[tr], C_all[tr], A_all[te], C_all[te]
    nte = len(te)
    print(f"train {len(tr)} / test {nte}")
    if A_tr[:, 3].mean() <= A_tr[:, 0].mean():
        print("[warn] frontier tier is not more accurate than the cheap tier on train: routing has nothing to gain "
              "on this data/pool (see report).")

    # ---- scorers
    scorers = [LengthScorer(), HeuristicScorer()]
    try:
        import sklearn, scipy  # noqa: F401
        scorers.append(TfidfRidgeScorer())
    except ImportError:
        print("[note] scikit-learn not installed: skipping tfidf_ridge (pip install scikit-learn)")
    for p in (a.scores_file.split(",") if a.scores_file else []):
        scorers.append(ExternalScorer(p))
    scorers += [BucketLabelScorer({i: bucket(items[i]) for i in ids}),
                OracleDifficulty(dict(zip(ids, d_all))), RandomScorer(a.seed + 1)]

    T = make_cfgs()
    d_tr, m_tr = d_all[tr], m_all[tr]
    R = OrderedDict()
    import copy
    fold_of = np.random.RandomState(a.seed + 3).permutation(len(tr)) % 5
    for sc in scorers:
        if getattr(sc, "learned", False):
            # Learned scorers: thresholds are chosen on OUT-OF-FOLD train scores. Scoring train prompts with a model
            # fitted on those same prompts is over-optimistic and makes thresholds miss their target on test.
            s_tr = np.zeros(len(tr))
            for f in range(5):
                va, trn = np.where(fold_of == f)[0], np.where(fold_of != f)[0]
                c = copy.deepcopy(sc)
                c.train_ids = [ids_tr[k] for k in trn]
                c.fit([P_tr[k] for k in trn], d_tr[trn], m_tr[trn])
                s_tr[va] = c.score([ids_tr[k] for k in va], [P_tr[k] for k in va])
            sc.train_ids = ids_tr
            sc.fit(P_tr, d_tr, m_tr)  # final model (all train) scores the test prompts
        else:
            sc.train_ids = ids_tr
            sc.fit(P_tr, d_tr, m_tr)
            s_tr = sc.score(ids_tr, P_tr)
        t0 = time.perf_counter()
        s_te = sc.score(ids_te, P_te)
        ms = (time.perf_counter() - t0) / nte * 1000
        ov_tr = np.array([sc.overhead.get(i, 0.0) for i in ids_tr]) if sc.overhead else np.zeros(len(tr))
        ov_te = np.array([sc.overhead.get(i, 0.0) for i in ids_te]) if sc.overhead else np.zeros(nte)
        pct_tr, pct_te = to_pct(s_tr, s_tr), to_pct(s_tr, s_te)
        tiers_tr, tiers_te = assign(pct_tr, T), assign(pct_te, T)
        rm = None
        if a.credit_reuse and getattr(sc, "reuse", None):
            rm = np.array([lad[t] == sc.reuse for t in range(4)])
            print(f"[info] {sc.name}: crediting reused answers of {sc.reuse} (tiers {list(np.where(rm)[0])})")
        a_tr, c_tr = cfg_items(tiers_tr, A_tr, C_tr, ov_tr, rm)
        a_te, c_te = cfg_items(tiers_te, A_te, C_te, ov_te, rm)
        R[sc.name] = dict(s_te=s_te, ms=ms, tiers_te=tiers_te, a_te=a_te, c_te=c_te,
                          tr_acc=a_tr.mean(1), tr_cost=c_tr.sum(1), te_acc=a_te.mean(1), te_cost=c_te.sum(1))

    acc_front_tr = A_tr[:, 3].mean()
    for name, r in R.items():
        r["par"] = pareto(r["tr_cost"], r["tr_acc"])
        r["ops"] = {}
        for q in targets:
            feas = np.where(r["tr_acc"] >= q * acc_front_tr - 1e-12)[0]
            r["ops"][q] = feas[np.lexsort((-r["tr_acc"][feas], r["tr_cost"][feas]))[0]]

    # ---- best single model per quality target (chosen on train): the baseline a router must beat
    ref = {}
    for q in targets:
        feas = [j for j in range(len(names)) if OK[tr, j].mean() >= q * acc_front_tr - 1e-12]
        ref[q] = min(feas, key=lambda j: (USD[tr, j].sum(), -OK[tr, j].mean())) if feas else li[3]

    # ---- baselines on test
    base = {}
    for j, s in enumerate(names):
        base[f"always:{s}"] = (OK[te, j], USD[te, j], None)
    base["oracle_router"] = None
    best = np.array([min((t for t in range(4) if A_te[k, t] == 1), key=lambda t: C_te[k, t], default=0)
                     for k in range(nte)])
    base["oracle_router"] = (A_te[np.arange(nte), best], C_te[np.arange(nte), best], best)
    # cascade: cheap-first, stop when two consecutive tiers agree; pays for every call
    CNl = CN[te][:, li]
    fin, cc = np.zeros(nte, int), np.zeros(nte)
    for k in range(nte):
        cc[k] = C_te[k, 0]
        fin[k] = 3
        for t in range(1, 4):
            cc[k] += C_te[k, t]
            if CNl[k, t] is not None and CNl[k, t] == CNl[k, t - 1]:
                fin[k] = t
                break
    base["cascade_agree"] = (A_te[np.arange(nte), fin], cc, fin)

    # ---- reported strategies (per-item acc/cost/tier on test)
    strat = OrderedDict()
    for k_, v in base.items():
        strat[k_] = dict(acc=v[0], cost=v[1], tier=v[2], scorer=None, q=None)
    for name, r in R.items():
        for q, idx in r["ops"].items():
            strat[f"{name}@{int(q * 100)}%"] = dict(acc=r["a_te"][idx], cost=r["c_te"][idx],
                                                    tier=r["tiers_te"][idx], scorer=name, q=q)

    # ---- bootstrap
    brng = np.random.RandomState(a.seed + 7)
    cnt = brng.multinomial(nte, [1 / nte] * nte, size=a.boot).astype(float)
    accF, costF = A_te[:, 3], C_te[:, 3]
    accC, costC = A_te[:, 0], C_te[:, 0]
    aF, cF, aC, cC = cnt @ accF / nte, cnt @ costF, cnt @ accC / nte, cnt @ costC

    def ci(v):
        return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))

    rows, bucket_rows = [], []
    te_buckets = np.array([bucket(items[i]) for i in ids_te])
    mix = lambda t: [float(np.mean(t == k)) for k in range(4)] if t is not None else [float("nan")] * 4
    for lab, s in strat.items():
        ab, cb = cnt @ s["acc"] / nte, cnt @ s["cost"]
        sav, ret = 1 - cb / cF, ab / aF
        under = over = float("nan")
        if s["tier"] is not None:
            ok_ = m_all[te] < 4
            under = float(np.mean(s["tier"][ok_] < m_all[te][ok_])) if ok_.any() else float("nan")
            over = float(np.mean(s["tier"][ok_] > m_all[te][ok_])) if ok_.any() else float("nan")
        if s["q"] is not None:  # compare with the cheapest single model meeting the same target
            j = ref[s["q"]]
            rc, ra = cnt @ USD[te, j], cnt @ OK[te, j] / nte
            extra = [names[j], 100 * (1 - s["cost"].sum() / USD[te, j].sum()), *(100 * x for x in ci(1 - cb / rc)),
                     100 * (s["acc"].mean() - OK[te, j].mean()), *(100 * x for x in ci(ab - ra))]
        else:
            extra = ["", *([float("nan")] * 6)]
        rows.append([lab, nte, s["acc"].mean(), *ci(ab), s["cost"].mean() * 100, *(x * 100 / nte for x in ci(cb)),
                     100 * (1 - s["cost"].sum() / costF.sum()), *(100 * x for x in ci(sav)),
                     100 * s["acc"].mean() / accF.mean(), *(100 * x for x in ci(ret)), *mix(s["tier"]), under, over,
                     *extra])
        if lab.startswith(("always:", "oracle", "cascade")) or s["q"] == targets[0]:
            for b in list(sorted(set(te_buckets))) + ["ALL"]:
                mk = np.ones(nte, bool) if b == "ALL" else te_buckets == b
                bucket_rows.append([lab, b, int(mk.sum()), s["acc"][mk].mean(), s["cost"][mk].mean() * 100,
                                    *mix(s["tier"][mk] if s["tier"] is not None else None)])

    # ---- scorer quality + AUQC
    qual = []
    for name, r in R.items():
        P = r["par"]
        aP, cP = r["a_te"][P], r["c_te"][P]
        aPb, cPb = cnt @ aP.T / nte, cnt @ cP.T
        vals = []
        for b in range(a.boot):
            den_c, den_a = cF[b] - cC[b], aF[b] - aC[b]
            if den_c <= 0 or den_a <= 0:
                vals.append(float("nan"))
                continue
            vals.append(auqc((cPb[b] - cC[b]) / den_c, (aPb[b] - aC[b]) / den_a))
        den_c, den_a = costF.sum() - costC.sum(), accF.mean() - accC.mean()
        point = auqc((r["te_cost"][P] - costC.sum()) / den_c,
                     (r["te_acc"][P] - accC.mean()) / den_a) if den_c > 0 and den_a > 0 else float("nan")
        lo, hi = ci(np.array(vals)) if not np.all(np.isnan(vals)) else (float("nan"),) * 2
        qual.append([name, spearman(r["s_te"], d_all[te]), auc(r["s_te"], A_te[:, 0] == 0),
                     auc(r["s_te"], m_all[te] >= 2), point, lo, hi, r["ms"]])

    # ---- write CSVs
    def w(fn, header, data):
        with open(os.path.join(a.outdir, fn), "w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            cw.writerow(header)
            cw.writerows(data)

    w("summary.csv", ["strategy", "n_test", "accuracy", "acc_lo", "acc_hi", "usd_per_100", "usd_lo", "usd_hi",
                      "savings_vs_frontier_pct", "sav_lo", "sav_hi", "quality_retention_pct", "ret_lo", "ret_hi",
                      "mix_t0", "mix_t1", "mix_t2", "mix_t3", "under_route_rate", "over_route_rate",
                      "ref_best_single_model", "sav_vs_ref_pct", "sav_ref_lo", "sav_ref_hi",
                      "acc_diff_vs_ref_pp", "accdiff_lo", "accdiff_hi"], rows)
    w("quality.csv", ["scorer", "spearman_vs_difficulty", "auc_cheap_fails", "auc_needs_tier2plus", "auqc",
                      "auqc_lo", "auqc_hi", "score_ms_per_prompt"], qual)
    w("buckets.csv", ["strategy", "bucket", "n", "accuracy", "usd_per_100", "mix_t0", "mix_t1", "mix_t2", "mix_t3"],
      bucket_rows)
    fr = []
    for name, r in R.items():
        for k in r["par"]:
            fr.append([name, *T[k], r["tr_acc"][k], r["tr_cost"][k], r["te_acc"][k], r["te_cost"][k]])
    w("frontier.csv", ["scorer", "t1", "t2", "t3", "train_acc", "train_usd", "test_acc", "test_usd"], fr)

    # ---- terminal report
    print(f"\nTEST (n={nte}); frontier tier acc={accF.mean():.3f}, cost/100=${costF.mean() * 100:.4f}")
    print(f"{'strategy':44s} {'acc':>6s} {'$/100':>9s} {'save%':>7s} {'retain%':>8s}  tier mix")
    for r in rows:
        print(f"{r[0]:44s} {r[2]:6.3f} {r[5]:9.4f} {r[8]:7.1f} {r[11]:8.1f}  "
              + "/".join(f"{x:.2f}" for x in r[14:18]))
    print("\nVs the cheapest SINGLE model meeting the same target (chosen on train). Positive = router is cheaper / more accurate:")
    for r in rows:
        if r[20]:
            print(f"  {r[0]:26s} ref={r[20].split('/')[-1]:22s} cost saved {r[21]:6.1f}% [{r[22]:.1f},{r[23]:.1f}]"
                  f"   acc diff {r[24]:+5.1f}pp [{r[25]:.1f},{r[26]:.1f}]")
    print("\nScorer quality (TEST): spearman vs difficulty | AUC cheap-fails | AUQC [95% CI]")
    for q_ in qual:
        print(f"  {q_[0]:20s} {q_[1]:7.3f} {q_[2]:7.3f}   {q_[4]:.3f} [{q_[5]:.3f}, {q_[6]:.3f}]")

    # ---- plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not installed, so no plots were saved. Run: python -m pip install matplotlib")
        return
    SURF, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
    col = {"tfidf_ridge": SLOT[0], "heuristic": SLOT[1], "length": SLOT[2], "cascade_agree": SLOT[6]}
    ext = [k for k in R if k.startswith("ext:")]
    for j, k in enumerate(ext):
        col[k] = SLOT[3 + j % 3]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRIDC, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7.4, 6.0), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.grid(color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # always-X models: grey markers with direct labels
    for j, s in enumerate(names):
        x, y = USD[te, j].sum(), OK[te, j].mean()
        ax.plot(x, y, "o", ms=6, color="#8a8985", mec=SURF, mew=1.5, zorder=3)
        ax.annotate(s.split("/")[-1], (x, y), xytext=(5, -9 if j % 2 else 5), textcoords="offset points", fontsize=7, color=INK2)
    MK = dict(zip(targets, ["o", "D", "^"]))
    cfor = lambda nm: "#8a8985" if nm == "random" else INK if nm == "oracle_difficulty" else col.get(nm, SLOT[7])
    for name, r in R.items():
        P, c = r["par"], cfor(name)
        ax.plot(r["te_cost"][P], r["te_acc"][P], "o", ms=3, alpha=0.3, color=c, zorder=2)  # train-selected frontier configs
        for q, idx in r["ops"].items():  # operating points chosen on train
            ax.plot(r["te_cost"][idx], r["te_acc"][idx], MK.get(q, "o"), ms=8, color=c, mec=SURF, mew=1.5, zorder=5)
    ax.plot(base["cascade_agree"][1].sum(), base["cascade_agree"][0].mean(), "s", ms=8, color=col["cascade_agree"],
            mec=SURF, mew=1.5, zorder=5)
    ax.plot(base["oracle_router"][1].sum(), base["oracle_router"][0].mean(), "*", ms=12, color=INK, mec=SURF, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Total cost on test set (USD, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cost vs accuracy on held-out prompts (up and left is better)", loc="left", fontsize=10, color=INK)
    handles = [Line2D([], [], color="#8a8985", marker="o", ls="", ms=6, label="single model (always-X)")]
    for name in R:
        handles.append(Line2D([], [], color=cfor(name), marker="o", ls="", ms=7, label=name))
    handles += [Line2D([], [], color=col["cascade_agree"], marker="s", ls="", ms=8, label="cascade (agreement)"),
                Line2D([], [], color=INK, marker="*", ls="", ms=11, label="oracle router (cheapest correct)")]
    handles += [Line2D([], [], color=INK2, marker=MK.get(q, "o"), ls="", ms=7, label=f"{int(q * 100)}% quality target (chosen on train)")
                for q in targets]
    handles.append(Line2D([], [], color=INK2, marker="o", ls="", ms=3, alpha=0.4, label="other train-Pareto configs"))
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "cost_quality.png"), dpi=200, facecolor=SURF)
    plt.close(fig)

    # tier mix by bucket for the main learned/heuristic scorer at the first target
    main_sc = "tfidf_ridge" if "tfidf_ridge" in R else "heuristic"
    lab = f"{main_sc}@{int(targets[0] * 100)}%"
    tiers_ = strat[lab]["tier"]
    bk = sorted(set(te_buckets))
    mixes = np.array([[np.mean(tiers_[te_buckets == b] == k) for k in range(4)] for b in bk])
    ramp = ["#c9dcf5", "#8fb8ec", "#4f8fdd", "#1f5aa8"]  # one hue, light to dark = cheap to frontier
    fig, ax = plt.subplots(figsize=(7.4, 0.6 * len(bk) + 1.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    left = np.zeros(len(bk))
    for k in range(4):
        ax.barh(bk, mixes[:, k], left=left, color=ramp[k], edgecolor=SURF, linewidth=1.5, label=f"tier {k}: {lad[k].split('/')[-1]}")
        left += mixes[:, k]
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Share of prompts routed to each tier")
    ax.set_title(f"Where {main_sc} sends each prompt type ({int(targets[0] * 100)}% quality target)", loc="left",
                 fontsize=10, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "bucket_mix.png"), dpi=200, facecolor=SURF)
    plt.close(fig)
    print(f"\nSaved to {a.outdir}: summary.csv quality.csv frontier.csv buckets.csv cost_quality.png bucket_mix.png")


if __name__ == "__main__":
    main()
