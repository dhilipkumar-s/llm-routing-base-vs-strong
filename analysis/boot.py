from core import *
import pickle, pandas as pd
W=World(); pool=pickle.load(open("pool.pkl","rb"))
rng=np.random.RandomState(0); B=4000
idx=rng.randint(0,W.n,(B,W.n))
def stats(p):
    m=p["cnt"]>0; a=p["acc"]/np.maximum(p["cnt"],1); c=p["cost"]/np.maximum(p["cnt"],1)
    return a,c
def lift_sav(rule,nm):
    a,c=stats(pool[rule][nm])
    A=a[idx].mean(1); C=c[idx].mean(1)*100
    ab=W.okb[idx].mean(1); as_=W.oks[idx].mean(1); cb=W.Cb[idx].mean(1)*100; cs=W.Cs[idx].mean(1)*100
    line=ab+(C-cb)/(cs-cb)*(as_-ab)
    lift=100*(A-line); sav=100*(1-C/cs); ret=100*A/as_
    return lift,sav,ret
out=[]
for rule in ["paper_grid11","fine_grid"]:
    L={nm:lift_sav(rule,nm) for nm in pool[rule]}
    for nm,(l,s,r) in L.items():
        out.append(dict(rule=rule,router=nm,lift=l.mean(),lo=np.percentile(l,2.5),hi=np.percentile(l,97.5),sav=s.mean(),slo=np.percentile(s,2.5),shi=np.percentile(s,97.5),ret=r.mean()))
    print("== paired differences in lift,",rule)
    for a_,b_ in [("selfcons3","length"),("selfcons3","disagree"),("selfcons3","combo_sc_cat"),("selfcons3","tfidf_ridge"),("length","disagree"),("selfcons3","oracle_pool"),("selfcons3","oracle_pair"),("length","random"),("disagree","random"),("selfcons3","random")]:
        d=L[a_][0]-L[b_][0]; print(f"{a_:10s}-{b_:12s} {d.mean():5.2f} [{np.percentile(d,2.5):5.2f},{np.percentile(d,97.5):5.2f}]  P(>0)={np.mean(d>0):.3f}")
df=pd.DataFrame(out).round(2); print(df.to_string()); df.to_csv("boot_lift.csv",index=False)
