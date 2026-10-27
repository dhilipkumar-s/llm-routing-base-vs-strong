import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'.')
import eval_fresh as E, simulate as S
d='fd/'
O=E.view(d+'data_final.jsonl',d+'responses_final.jsonl'); F=E.view(d+'data_fresh.jsonl',d+'responses_fresh.jsonl')
src=np.array([F['items'][i].get('source') for i in F['ids']])
print("fresh n",len(F['ids']),pd.Series(src).value_counts().to_dict())
print("fresh base acc %.3f strong acc %.3f ; by source:"%(F['okb'].mean(),F['oks'].mean()))
for s in sorted(set(src)):
    m=src==s; print(f"  {s:9s} n={m.sum():3d} base={F['okb'][m].mean():.3f} strong={F['oks'][m].mean():.3f} gain={((F['okb'][m]==0)&(F['oks'][m]==1)).mean():.3f}")
print("old by source base/strong:"); osrc=np.array([O['items'][i].get('source') for i in O['ids']])
for s in sorted(set(osrc)):
    m=osrc==s; print(f"  {s:9s} n={m.sum():3d} base={O['okb'][m].mean():.3f} strong={O['oks'][m].mean():.3f}")
specs={"selfcons3":("selfcons_k2.jsonl","selfcons_fresh.jsonl"),"disagree":("dis_old.jsonl","dis_fresh.jsonl"),"judge":("judge.jsonl","judge_fresh.jsonl"),
       "verbal":("verbal_old.jsonl","verbal_fresh.jsonl"),"embed":("embed_old.jsonl","embed_fresh.jsonl")}
sc={}
zo,zf=np.zeros(len(O['ids'])),np.zeros(len(F['ids']))
sc["length"]=(np.array([len(p) for p in O['prompts']],float),np.array([len(p) for p in F['prompts']],float),zo,zf,False)
for k,(fo,ff) in specs.items():
    so,oo,ro=E.read_scores(d+fo,O['ids']); sf,of,rf=E.read_scores(d+ff,F['ids']); so=np.where(np.isnan(so),np.nanmax(so),so); sc[k]=(so,sf,oo,of,ro or rf)
gain=lambda V:((V['okb']==0)&(V['oks']==1)).astype(float)+1e-9*np.arange(len(V['ids']))
sc["oracle_pair"]=(gain(O),gain(F),zo,zf,False)
rows=[]
for name,(so,sf,oo,of,reuse) in sc.items():
    t=E.pick(O,so,oo,reuse,0.95*O['oks'].mean()); esc=np.nan_to_num(sf,nan=np.inf)>t
    acc=np.where(esc,F['oks'],F['okb']); cost=E.costs(F,esc,of,reuse)
    for s in ["ALL"]+sorted(set(src)):
        m=np.ones(len(src),bool) if s=="ALL" else src==s
        rows.append(dict(router=name,src=s,n=int(m.sum()),esc=100*esc[m].mean(),acc=100*acc[m].mean(),base=100*F['okb'][m].mean(),strong=100*F['oks'][m].mean(),
                         cost_vs_sonnet=100*cost[m].sum()/F['Cs'][m].sum()))
df=pd.DataFrame(rows).round(1); pd.set_option("display.width",200); print(df.to_string(index=False)); df.to_csv("fresh_by_source.csv",index=False)
# AUC of scores for base failure within each fresh source
print("\nAUC for base failure on fresh prompts, by source")
fail=1-F['okb']
for name,(so,sf,oo,of,reuse) in sc.items():
    if name=="oracle_pair": continue
    out=[]
    for s in ["ALL"]+sorted(set(src)):
        m=np.ones(len(src),bool) if s=="ALL" else src==s
        v=np.nan_to_num(sf[m],nan=np.nanmax(sf)); out.append(f"{s}={S.auc(v,fail[m]==1):.2f}(fails={int(fail[m].sum())})")
    print(f"{name:10s}"," ".join(out))
