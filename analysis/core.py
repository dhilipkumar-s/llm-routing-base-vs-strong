"""Shared code for the reviewer-response analyses. Two-tier router (gpt-oss-120b -> Sonnet 5.5) on cached answers.
Reuses load/scorers/splits from simulate.py so numbers can be checked against the paper."""
import copy, json, math, os, sys
from pathlib import Path
from collections import OrderedDict, defaultdict
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def P(name):
    """Locate a data / cache / score file by name inside the repository."""
    for sub in ("data", "cache", "scores"):
        p = ROOT / sub / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(name)

import simulate as S  # noqa: E402  (src/ is on the path above)
BASE, STRONG = "openai/gpt-oss-120b", "anthropic/claude-sonnet-5.5"
GRID11 = np.round(np.linspace(0, 1, 11), 2)
GRID101 = np.round(np.linspace(0, 1, 101), 2)


class World:
    def __init__(self):
        self.items, self.ids, self.names, self.OK, self.USD, self.CN = S.load(P("data_final.jsonl"), P("responses_final.jsonl"), 0.9)
        self.n = len(self.ids)
        b, s = self.names.index(BASE), self.names.index(STRONG)
        self.b, self.s = b, s
        self.okb, self.oks = self.OK[:, b], self.OK[:, s]
        self.Cb, self.Cs = self.USD[:, b], self.USD[:, s]
        self.d_pool = 1 - self.OK.mean(1)                               # paper's "oracle difficulty"
        self.gain = ((self.okb == 0) & (self.oks == 1)).astype(float)   # pair-specific: base wrong, strong right
        self.src = np.array([self.items[i].get("source") for i in self.ids])
        self.bucket = np.array([S.bucket(self.items[i]) for i in self.ids])
        self.prompts = [self.items[i]["prompt"] for i in self.ids]
        self.ext = {}
        self.add_ext("judge", "judge.jsonl"); self.add_ext("disagree", "disagree_120b.jsonl"); self.add_ext("selfcons3", "selfcons_k2.jsonl")
        for k in ("combo", "combo_nojudge", "combo_sc", "combo_sc_cat"):
            self.add_ext(k, k + ".jsonl")

    def add_ext(self, name, path_or_dict):
        if isinstance(path_or_dict, str):
            rows = {}
            for l in open(P(path_or_dict)):
                if l.strip():
                    r = json.loads(l); rows[r["id"]] = r
        else:
            rows = path_or_dict
        sc = np.array([rows[i]["score"] for i in self.ids], float)
        ov = np.array([rows[i].get("cost", 0.0) for i in self.ids], float)
        reuse = any(r.get("reuse") == BASE for r in rows.values())
        self.ext[name] = (sc, ov, reuse)


def split(W, seed, frac=0.5, idx=None):
    """Same stratified split as simulate.py (stratified by dataset bucket). idx restricts to a subset of prompts."""
    rng = np.random.RandomState(seed)
    idx = np.arange(W.n) if idx is None else np.asarray(idx)
    strata = defaultdict(list)
    for k in idx:
        strata[W.bucket[k]].append(k)
    tr, te = [], []
    for b in sorted(strata):
        l = list(strata[b]); rng.shuffle(l)
        nt = int(round(len(l) * frac))
        te += l[:nt]; tr += l[nt:]
    return np.array(sorted(tr)), np.array(sorted(te))


def get_scores(W, tr, te, seed, names, learned_cache=None):
    """Return {name: (s_tr, s_te, ov_tr, ov_te, reuse)} for train/test index arrays."""
    out = {}
    ids_tr, ids_te = [W.ids[k] for k in tr], [W.ids[k] for k in te]
    P_tr, P_te = [W.prompts[k] for k in tr], [W.prompts[k] for k in te]
    z = lambda m: np.zeros(m)
    for nm in names:
        if nm == "length":
            out[nm] = (np.array([len(p) for p in P_tr], float), np.array([len(p) for p in P_te], float), z(len(tr)), z(len(te)), False)
        elif nm == "heuristic":
            h = S.HeuristicScorer()
            out[nm] = (h.score(ids_tr, P_tr), h.score(ids_te, P_te), z(len(tr)), z(len(te)), False)
        elif nm == "tfidf_ridge":
            d_tr = W.d_pool[tr]
            fold = np.random.RandomState(seed + 3).permutation(len(tr)) % 5
            s_tr = np.zeros(len(tr))
            for f in range(5):
                va, trn = np.where(fold == f)[0], np.where(fold != f)[0]
                m = S.TfidfRidgeScorer(); m.fit([P_tr[k] for k in trn], d_tr[trn], None)
                s_tr[va] = m.score(None, [P_tr[k] for k in va])
            m = S.TfidfRidgeScorer(); m.fit(P_tr, d_tr, None)
            out[nm] = (s_tr, m.score(None, P_te), z(len(tr)), z(len(te)), False)
        elif nm == "oracle_pool":
            out[nm] = (W.d_pool[tr] + 1e-9 * np.arange(len(tr)), W.d_pool[te] + 1e-9 * np.arange(len(te)), z(len(tr)), z(len(te)), False)
        elif nm == "oracle_pair":
            out[nm] = (W.gain[tr] + 1e-9 * np.arange(len(tr)), W.gain[te] + 1e-9 * np.arange(len(te)), z(len(tr)), z(len(te)), False)
        elif nm == "random":
            r = np.random.RandomState(seed + 1)
            out[nm] = (r.rand(len(tr)), r.rand(len(te)), z(len(tr)), z(len(te)), False)
        else:
            sc, ov, reuse = W.ext[nm]
            out[nm] = (sc[tr], sc[te], ov[tr], ov[te], reuse)
    return out


def route_cost(W, idx, esc, ov, reuse):
    Cb, Cs = W.Cb[idx], W.Cs[idx]
    base_c = np.where(reuse, 0.0, Cb) if np.isscalar(reuse) or isinstance(reuse, bool) else Cb
    return np.where(esc, Cs, base_c) + ov


def pct(s_tr, s):
    return S.to_pct(s_tr, s)


def curve(W, idx, s_pct, ov, reuse, grid):
    """For each grid threshold c (escalate iff pct>c) return per-config acc (mean) and total cost, plus escalation matrix."""
    E = s_pct[None, :] > grid[:, None]                                  # (G, n)
    base_c = 0.0 if reuse else W.Cb[idx]
    cost = np.where(E, W.Cs[idx][None, :], base_c if np.isscalar(base_c) else base_c[None, :]) + ov[None, :]
    acc = np.where(E, W.oks[idx][None, :], W.okb[idx][None, :])
    return E, acc, cost


def pick(acc_tr, cost_tr, thr):
    feas = np.where(acc_tr.mean(1) >= thr - 1e-12)[0]
    return feas[np.lexsort((-acc_tr.mean(1)[feas], cost_tr.sum(1)[feas]))[0]]


def run_split(W, seed, names, grid=GRID11, q=0.95, margin=0.0, rule="plain", z=1.645, tr=None, te=None, scores=None):
    """One split. rule: 'plain' = cheapest config whose train accuracy >= (q+margin)*train strong acc;
    'ucb' = cheapest config whose upper confidence bound on the train retention shortfall is <= (1-q)."""
    if tr is None:
        tr, te = split(W, seed)
    sc = scores or get_scores(W, tr, te, seed, names)
    accF_tr = W.oks[tr].mean()
    res = {}
    for nm, (s_tr, s_te, ov_tr, ov_te, reuse) in sc.items():
        p_tr, p_te = pct(s_tr, s_tr), pct(s_tr, s_te)
        E_tr, a_tr, c_tr = curve(W, tr, p_tr, ov_tr, reuse, grid)
        E_te, a_te, c_te = curve(W, te, p_te, ov_te, reuse, grid)
        if rule == "plain":
            j = pick(a_tr, c_tr, (q + margin) * accF_tr)
        else:  # ucb on mean shortfall (strong - router) relative to the allowed (1-q)*accF
            short = W.oks[tr][None, :] - a_tr                      # per-prompt shortfall
            ucb = short.mean(1) + z * short.std(1, ddof=1) / math.sqrt(len(tr))
            feas = np.where(ucb <= (1 - q) * accF_tr + 1e-12)[0]
            if len(feas) == 0:
                feas = np.array([0])
            j = feas[np.lexsort((-a_tr.mean(1)[feas], c_tr.sum(1)[feas]))[0]]
        res[nm] = dict(j=j, esc=E_te[j], acc=a_te[j], cost=c_te[j], te=te, c=grid[j], p_te=p_te, s_te=s_te)
    return res


def summarize(W, r, te):
    """Per-split metrics for one router result on test indices te."""
    accF, costF = W.oks[te].mean(), W.Cs[te].sum()
    accB, costB = W.okb[te].mean(), W.Cb[te].mean() * 100
    acc, cost = r["acc"].mean(), r["cost"].mean() * 100
    line = accB + (cost - costB) / (W.Cs[te].mean() * 100 - costB) * (accF - accB)
    # count-matched random mix (ignores cost): same escalation fraction
    f = r["esc"].mean()
    line_cnt = accB + f * (accF - accB)
    return dict(savings=100 * (1 - r["cost"].sum() / costF), acc_diff=100 * (acc - accF), retention=100 * acc / accF,
                lift=100 * (acc - line), lift_cnt=100 * (acc - line_cnt), esc=100 * f)
