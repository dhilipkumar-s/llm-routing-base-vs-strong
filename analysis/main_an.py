from core import *
import pandas as pd, pickle, time
W=World()
names=["length","heuristic","tfidf_ridge","judge","disagree","selfcons3","combo_sc_cat","oracle_pool","oracle_pair","random"]
RULES={"paper_grid11":dict(grid=GRID11),"fine_grid":dict(grid=GRID101),"margin+2":dict(grid=GRID101,margin=0.02),
       "margin+4":dict(grid=GRID101,margin=0.04),"ucb":dict(grid=GRID101,rule="ucb")}
NS=100
rows=[]; pool={r:{nm:dict(acc=np.zeros(W.n),cost=np.zeros(W.n),esc=np.zeros(W.n),cnt=np.zeros(W.n)) for nm in names} for r in RULES}
t0=time.time()
for seed in range(1,NS+1):
    tr,te=split(W,seed)
    sc=get_scores(W,tr,te,seed,names)
    for rn,kw in RULES.items():
        res=run_split(W,seed,names,tr=tr,te=te,scores=sc,**kw)
        for nm,r in res.items():
            m=summarize(W,r,te); m.update(rule=rn,router=nm,seed=seed,held=m["retention"]>=95); rows.append(m)
            p=pool[rn][nm]; p["acc"][te]+=r["acc"]; p["cost"][te]+=r["cost"]; p["esc"][te]+=r["esc"]; p["cnt"][te]+=1
    if seed%20==0: print(seed,time.time()-t0,flush=True)
df=pd.DataFrame(rows); df.to_pickle("splits.pkl"); pickle.dump(pool,open("pool.pkl","wb"))
g=df.groupby(["rule","router"]).agg(sav=("savings","mean"),acc=("acc_diff","mean"),ret=("retention","mean"),held=("held","mean"),lift=("lift","mean"),lift_sd=("lift","std")).round(2)
pd.set_option("display.width",200); print(g.to_string())
# summary table used by gen_tables.py and in the paper (Table "rules")
out = df.groupby(["rule", "router"]).agg(savings=("savings", "mean"), savings_sd=("savings", "std"), acc_diff_pp=("acc_diff", "mean"),
                                         retention=("retention", "mean"), held_frac=("held", "mean"), lift_pp=("lift", "mean"), lift_sd=("lift", "std")).round(2).reset_index()
out.to_csv("rules_100splits.csv", index=False)
