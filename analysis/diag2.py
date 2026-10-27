from core import *
import pandas as pd, pickle, json
from collections import Counter, defaultdict
W=World(); pd.set_option("display.width",200)
allids=[json.loads(l)["id"] for l in open(P("data_final.jsonl"))]
miss=[i for i in allids if i not in set(W.ids)]
print("dropped ids:",miss)
rec=defaultdict(set)
for l in open(P("responses_final.jsonl")):
    r=json.loads(l)
    if not r.get("error"): rec[r["id"]].add(r["model"])
for i in miss: print(i,"has",len(rec[i]),"models; missing",set(W.names)-rec[i])
# --- per-benchmark pooled results
pool=pickle.load(open("pool.pkl","rb"))["paper_grid11"]
print("\nPer-benchmark (paper rule, pooled over 100 splits): escalation %, acc, then always-base/strong")
rows=[]
for s in sorted(set(W.src)):
    m=W.src==s
    for nm in ["length","disagree","selfcons3","oracle_pair","random"]:
        p=pool[nm]; c=np.maximum(p["cnt"],1)
        rows.append(dict(src=s,router=nm,esc=100*(p["esc"]/c)[m].mean(),acc=100*(p["acc"]/c)[m].mean(),base=100*W.okb[m].mean(),strong=100*W.oks[m].mean(),
                         cost100=(p["cost"]/c)[m].mean()*100, strong_cost100=W.Cs[m].mean()*100))
print(pd.DataFrame(rows).round(2).to_string())
# --- parse failures
print("\nParse-failure rate (canon is None) by model and source")
tab=defaultdict(dict)
for j,n in enumerate(W.names):
    for s in sorted(set(W.src)):
        m=W.src==s
        tab[n.split("/")[-1]][s]=np.mean([W.CN[k,j] is None for k in np.where(m)[0]])
print(pd.DataFrame(tab).T.round(3).to_string())
# votes for oss120
samp=defaultdict(dict)
for l in open(P("samples_120b.jsonl")):
    r=json.loads(l)
    if r["model"]==BASE and not r.get("error"): samp[r["id"]][r["k"]]=r
gre={}
for l in open(P("responses_final.jsonl")):
    r=json.loads(l)
    if r["model"]==BASE and not r.get("error"): gre[r["id"]]=r
V=[];
for k,i in enumerate(W.ids):
    it=W.items[i]; vs=[S.canon(it,gre[i]["text"])]+[S.canon(it,samp[i][kk]["text"]) for kk in sorted(samp[i])[:2]]
    V.append(vs)
nun=np.array([sum(v is None for v in vs) for vs in V])
print("\nprompts with >=1 unparsed vote among 3:",(nun>0).sum(),"of",W.n,"; all 3 unparsed:",(nun==3).sum())
def votescore(vs,mode):
    if mode=="distinct": vs=[v if v is not None else f"u{k}" for k,v in enumerate(vs)]
    elif mode=="single": vs=[v if v is not None else "UNP" for v in vs]
    elif mode=="drop": vs=[v for v in vs if v is not None] or ["x"]
    top=Counter(vs).most_common(1)[0][1]; return (1-top/len(vs))+0.01*len(set(vs))
# unparsed -> effect on non-unanimity
for mode in ["distinct","single","drop"]:
    sc=np.array([votescore(v,mode) for v in V]); nonun=sc>0.02
    print(mode,"non-unanimous share",nonun.mean().round(3),"| precision for base-fail",W.okb[nonun].mean().round(3),"-> fail rate among flagged",(1-W.okb[nonun]).mean().round(3),"recall of base failures",nonun[W.okb==0].mean().round(3))
# parse-failure-only share: is signal from parse failures?
pf=nun>0
print("base fail rate when some vote unparsed:",(1-W.okb[pf]).mean().round(3),"n=",pf.sum(), "| otherwise",(1-W.okb[~pf]).mean().round(3))
# --- unanimous errors
sc=np.array([votescore(v,"distinct") for v in V]); unan=sc<0.02
err=W.okb==0
print("\nbase errors:",err.sum(),"=",err.mean().round(3),"of prompts. Of errors, unanimous across 3 votes:",unan[err].mean().round(3),"(consistent-but-wrong =",(unan&err).mean().round(3),"of all prompts)")
print("non-unanimous:",(~unan).mean().round(3),"of prompts; of those, base wrong:",err[~unan].mean().round(3))
print("gain prompts (base wrong, strong right):",int(W.gain.sum()),"; of those unanimous-wrong:",unan[W.gain==1].mean().round(3))
for s in sorted(set(W.src)):
    m=W.src==s; print(f" {s:9s} base err {err[m].mean():.3f}  unanimous share of errors {unan[m&err].mean() if (m&err).any() else float('nan'):.3f}  non-unanimous rate {(~unan[m]).mean():.3f}")
