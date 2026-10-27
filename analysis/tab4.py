from core import *
import pandas as pd
W=World()
tr=np.arange(W.n); L=np.array([len(p) for p in W.prompts],float)
# OOF tfidf over all prompts (5-fold), same recipe
fold=np.random.RandomState(3).permutation(W.n)%5; tf=np.zeros(W.n)
for f in range(5):
    va,trn=np.where(fold==f)[0],np.where(fold!=f)[0]
    m=S.TfidfRidgeScorer(); m.fit([W.prompts[k] for k in trn],W.d_pool[trn],None); tf[va]=m.score(None,[W.prompts[k] for k in va])
sc={"length":L,"heuristic":S.HeuristicScorer().score(None,W.prompts),"tfidf_ridge":tf,"judge":W.ext["judge"][0],"disagree":W.ext["disagree"][0],
    "selfcons3":W.ext["selfcons3"][0],"combo_sc_cat":W.ext["combo_sc_cat"][0],"oracle_pool":W.d_pool,"oracle_pair":W.gain+1e-9*np.arange(W.n),"random":np.random.RandomState(1).rand(W.n)}
cheapest=1-W.OK[:,0]   # first model in pool = cheapest (Llama-3.1-8B)
print("names:",W.names[0])
base_fail=1-W.okb
rows=[]
for subset,m in [("all",np.ones(W.n,bool)),("mmlu_pro",W.src=="mmlu_pro")]:
    for nm,s in sc.items():
        rows.append(dict(subset=subset,scorer=nm,rho_pool=S.spearman(s[m],W.d_pool[m]),rho_basefail=S.spearman(s[m],base_fail[m]),auc_cheapest8b=S.auc(s[m],cheapest[m]==1),
                         auc_basefail=S.auc(s[m],base_fail[m]==1),auc_gain=S.auc(s[m],W.gain[m]==1)))
df=pd.DataFrame(rows).round(3); print(df.to_string()); df.to_csv("table4_recomputed.csv",index=False)
