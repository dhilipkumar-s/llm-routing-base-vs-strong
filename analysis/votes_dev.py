from core import *
import pandas as pd, json
from collections import Counter, defaultdict
W=World()
samp=defaultdict(dict); gre={}
for l in open(P("samples_120b.jsonl")):
    r=json.loads(l)
    if r["model"]==BASE and not r.get("error"): samp[r["id"]][r["k"]]=r
for l in open(P("responses_final.jsonl")):
    r=json.loads(l)
    if r["model"]==BASE and not r.get("error"): gre[r["id"]]=r
print("samples per prompt:",Counter(len(samp[i]) for i in W.ids), "k values:",Counter(k for i in W.ids for k in samp[i]))
def cost_of(r): return float(r["cost"]) if r.get("cost") is not None else 0
def build(nv):
    rows={}
    for k,i in enumerate(W.ids):
        it=W.items[i]; rs=[gre[i]]+[samp[i][kk] for kk in sorted(samp[i])[:nv-1]]
        vs=[S.canon(it,r["text"]) for r in rs]; vs=[v if v is not None else f"u{j}" for j,v in enumerate(vs)]
        top=Counter(vs).most_common(1)[0][1]
        rows[i]=dict(score=(1-top/len(vs))+0.01*len(set(vs)),cost=sum(cost_of(r) for r in rs),reuse=BASE)
    return rows
for nv in (2,3,4,5): W.add_ext(f"sc{nv}",build(nv))
# sanity: sc3 equals the provided file?
print("sc3 vs file corr:",np.corrcoef(W.ext["sc3"][0],W.ext["selfcons3"][0])[0,1].round(4))
names=["sc2","sc3","sc4","sc5"]
# (1) all 100 splits, paper rule: lift per vote count
rows=[]
for seed in range(1,101):
    res=run_split(W,seed,names,grid=GRID11)
    for nm,r in res.items():
        m=summarize(W,r,r["te"]); m.update(router=nm,seed=seed); rows.append(m)
df=pd.DataFrame(rows); print(df.groupby("router")[["savings","retention","lift"]].agg(["mean","std"]).round(2).to_string())
# (2) honest dev/test: random 50/50 partition; on dev (repeated inner splits) pick best vote count by lift; then fit thresholds on dev, evaluate on test half
sel=[];test=defaultdict(list)
for rep in range(60):
    dev,tst=split(W,1000+rep)
    inner=defaultdict(list)
    for sd in range(10):
        tr,te=split(W,sd,idx=dev)
        res=run_split(W,sd,names,grid=GRID11,tr=tr,te=te)
        for nm,r in res.items(): inner[nm].append(summarize(W,r,te)["lift"])
    best=max(names,key=lambda n:np.mean(inner[n])); sel.append(best)
    # evaluate each vote count with thresholds picked on all dev, tested on tst
    res=run_split(W,rep,names,grid=GRID11,tr=dev,te=tst)
    for nm,r in res.items(): m=summarize(W,r,tst); test[nm].append((m["lift"],m["savings"],m["retention"]))
print("vote count selected on dev:",Counter(sel))
for nm in names:
    a=np.array(test[nm]); print(nm,"test-half lift %.2f (sd %.2f) savings %.1f retention %.1f held %.2f"%(a[:,0].mean(),a[:,0].std(),a[:,1].mean(),a[:,2].mean(),(a[:,2]>=95).mean()))
selected=np.array([test[s][i][0] for i,s in enumerate(sel)])
print("lift on test half of the dev-selected vote count: %.2f (sd %.2f)"%(selected.mean(),selected.std()))
