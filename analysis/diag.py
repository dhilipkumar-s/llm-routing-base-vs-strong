from core import *
import pandas as pd
from collections import Counter
W=World(); pd.set_option("display.width",200)
L=np.array([len(p) for p in W.prompts],float)
fail=1-W.okb
print("n by source",Counter(W.src))
print("base acc / strong acc / gain rate by source")
for s in sorted(set(W.src)):
    m=W.src==s
    print(f"{s:10s} n={m.sum():4d} base={W.okb[m].mean():.3f} strong={W.oks[m].mean():.3f} gain={W.gain[m].mean():.3f} base_wrong_strong_wrong={((W.okb[m]==0)&(W.oks[m]==0)).mean():.3f} Cs/100={W.Cs[m].mean()*100:.3f} Cb/100={W.Cb[m].mean()*100:.4f}")
print("\nLength vs base failure / gain: Spearman and AUC (higher length -> more failure)")
for s in ["ALL"]+sorted(set(W.src)):
    m=np.ones(W.n,bool) if s=="ALL" else W.src==s
    print(f"{s:10s} spearman(len,fail)={S.spearman(L[m],fail[m]):+.3f} auc_fail={S.auc(L[m],fail[m]==1):.3f} auc_gain={S.auc(L[m],W.gain[m]==1):.3f}  spearman(len,Cs)={S.spearman(L[m],W.Cs[m]):+.3f}")
# direction of the length router: who gets escalated?
tr,te=split(W,1); res=run_split(W,1,["length"],tr=tr,te=te)["length"]
e=res["esc"]; print("\nlength router seed1: escalated share",e.mean().round(3),"mean len esc/non",L[te][e].mean().round(0),L[te][~e].mean().round(0))
print(" sonnet cost per prompt esc/non (x100):",(W.Cs[te][e].mean()*100).round(4),(W.Cs[te][~e].mean()*100).round(4))
print(" by source share escalated:",{s:round(float(e[W.src[te]==s].mean()),2) for s in sorted(set(W.src))})
# is length mostly a dataset detector? within-source-only length router
print("\nLift by benchmark (pooled over 100 splits are in pool.pkl). Quick: within-source length AUC for gain & fail shown above.")
# extra: stratify MMLU-Pro by category if available
cat=[W.items[i].get("category") for i in W.ids]
print("categories present:",Counter(cat).most_common(5))
m=W.src=="mmlu_pro"
print("mmlu rows",m.sum())
# MMLU-Pro: number of options
import re
nopt=np.array([len(re.findall(r"(?m)^[A-J][\.\)] ",p)) for p in W.prompts])
print("option count dist (mmlupro):",Counter(nopt[m]))
for k in sorted(set(nopt[m])):
    mm=m&(nopt==k)
    if mm.sum()>15: print(f" nopt={k} n={mm.sum()} base acc {W.okb[mm].mean():.3f} mean len {L[mm].mean():.0f}")
