import glob, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R="../results/"   # per-seed simulate.py outputs (summary.csv), see README
SRC={"full":{"sc":R+"seeds_sc","k":R+"seeds_k"},"mmlu":{"sc":R+"seeds_sc_mmlu","k":R+"seeds_k_mmlu"}}
def perseed(root):
    out=[]
    for sf in sorted(glob.glob(root+"/seed_*/summary.csv")):
        s=pd.read_csv(sf).set_index("strategy"); A=s.loc["always:openai/gpt-oss-120b"];B=s.loc["always:anthropic/claude-sonnet-5.5"]
        s=s.copy()
        p=(s.usd_per_100-A.usd_per_100)/(B.usd_per_100-A.usd_per_100)
        s["lift"]=100*(s.accuracy-(A.accuracy+p*(B.accuracy-A.accuracy)))
        out.append(s)
    return out
NAMES={"ext:selfcons_120b":"Self-consistency (3 votes)","ext:disagree_120b":"Disagreement (120b vs 20b)","ext:combo_120b":"Learned combiner","ext:combo_sc":"Combiner + samples","ext:combo_sc_cat":"Combiner + samples + subject",
"length":"Prompt length","tfidf_ridge":"TF-IDF ridge","heuristic":"Heuristic features","oracle_difficulty":"Oracle difficulty (ceiling)","random":"Random scores"}
def agg(seeds,key,col):
    v=np.array([s.loc[key,col] for s in seeds]); return v.mean(), v.std(ddof=1)
rows=[]; md=[]; tex=[]
for view in ["full","mmlu"]:
    S=perseed(SRC[view]["sc"])
    for t in ["95","99"]:
        for k,n in NAMES.items():
            key=f"{k}@{t}%"
            r=dict(view=view,target=t,method=n)
            for c,col in [("sav","savings_vs_frontier_pct"),("acc","accuracy"),("ret","quality_retention_pct"),("lift","lift"),("usd","usd_per_100")]:
                r[c],r[c+"_sd"]=agg(S,key,col)
            r["held"]=np.mean([s.loc[key,"quality_retention_pct"]>=float(t) for s in S])
            r["accdiff"]=np.mean([100*(s.loc[key,"accuracy"]-s.loc["always:anthropic/claude-sonnet-5.5","accuracy"]) for s in S])
            rows.append(r)
    for k in ["always:openai/gpt-oss-120b","always:anthropic/claude-sonnet-5.5","always:anthropic/claude-opus-5.5","oracle_router"]:
        r=dict(view=view,target="-",method=k)
        for c,col in [("acc","accuracy"),("usd","usd_per_100"),("ret","quality_retention_pct")]: r[c],r[c+"_sd"]=agg(S,k,col)
        r["accdiff"]=np.mean([100*(s.loc[k,"accuracy"]-s.loc["always:anthropic/claude-sonnet-5.5","accuracy"]) for s in S])
        r["sav"],r["sav_sd"]=agg(S,k,"savings_vs_frontier_pct"); rows.append(r)
D=pd.DataFrame(rows); D.to_csv("main_results_8seeds.csv",index=False)
# K ablation
Kr=[]
for view in ["full","mmlu"]:
    S=perseed(SRC[view]["k"])
    for kk,lab in [("k1","2 votes (K=1)"),("k2","3 votes (K=2)"),("k4","5 votes (K=4)")]:
        for t in ["95","99"]:
            key=f"ext:selfcons_{kk}@{t}%"; r=dict(view=view,target=t,votes=lab)
            for c,col in [("sav","savings_vs_frontier_pct"),("accdiff","acc_diff_vs_ref_pp"),("ret","quality_retention_pct"),("lift","lift"),("usd","usd_per_100")]:
                r[c],r[c+"_sd"]=agg(S,key,col)
            r["held"]=np.mean([s.loc[key,"quality_retention_pct"]>=float(t) for s in S]); Kr.append(r)
K=pd.DataFrame(Kr); K.to_csv("k_ablation_8seeds.csv",index=False)
# ---- tables
def f(m,s,d=1): return f"{m:.{d}f} ({s:.{d}f})"
def main_table(view,t):
    d=D[(D.view==view)&(D.target==t)]
    h="| Router | Savings vs Sonnet % | Accuracy diff (pp) | Retention % | Splits meeting target | Lift over random mix (pp) |\n|---|---|---|---|---|---|\n"
    return h+"\n".join(f"| {r.method} | {f(r.sav,r.sav_sd)} | {r.accdiff:+.1f} | {r.ret:.1f} | {int(round(r.held*8))}/8 | {f(r.lift,r.lift_sd)} |" for r in d.itertuples())
def k_table(view):
    d=K[K.view==view]
    h="| Probe | Target | Savings % | Accuracy diff (pp) | Retention % | Splits meeting target | Lift (pp) |\n|---|---|---|---|---|---|---|\n"
    return h+"\n".join(f"| {r.votes} | {r.target}% | {f(r.sav,r.sav_sd)} | {r.accdiff:+.1f} | {r.ret:.1f} | {int(round(r.held*8))}/8 | {f(r.lift,r.lift_sd)} |" for r in d.itertuples())
ref=lambda view: D[(D.view==view)&(D.target=="-")]
def ref_table(view):
    h="| Policy | Accuracy % | USD per 100 prompts | Savings vs Sonnet % |\n|---|---|---|---|\n"
    nm={"always:openai/gpt-oss-120b":"Always gpt-oss-120b","always:anthropic/claude-sonnet-5.5":"Always Claude Sonnet 5.5","always:anthropic/claude-opus-5.5":"Always Claude Opus 5.5","oracle_router":"Oracle router (cheapest correct of the two)"}
    return h+"\n".join(f"| {nm[r.method]} | {100*r.acc:.1f} | {r.usd:.3f} | {r.sav:.1f} |" for r in ref(view).itertuples())
open("tables.md","w").write("\n\n".join([
 "### Table A. Reference policies (full set, mean of 8 splits)\n"+ref_table("full"),
 "### Table B. Reference policies (MMLU-Pro only)\n"+ref_table("mmlu"),
 "### Table C. Routers at the 95% operating point, full set\n"+main_table("full","95"),
 "### Table D. Routers at the 95% operating point, MMLU-Pro only\n"+main_table("mmlu","95"),
 "### Table E. Routers at the 99% operating point, full set\n"+main_table("full","99"),
 "### Table F. Routers at the 99% operating point, MMLU-Pro only\n"+main_table("mmlu","99"),
 "### Table G. Number of votes, full set\n"+k_table("full"),
 "### Table H. Number of votes, MMLU-Pro only\n"+k_table("mmlu")]))
# ---- figures
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#52514e","axes.labelcolor":"#0b0b0b","xtick.color":"#52514e","ytick.color":"#52514e","figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","axes.grid":True,"grid.color":"#e6e5e1","grid.linewidth":0.6})
col={"Self-consistency (3 votes)":"#2a78d6","Disagreement (120b vs 20b)":"#eb6834","Learned combiner":"#1baf7a","Prompt length":"#eda100","TF-IDF ridge":"#e87ba4","Heuristic features":"#6250d6","Oracle difficulty (ceiling)":"#52514e","Random scores":"#a8a79f"}
mk={"Self-consistency (3 votes)":"o","Disagreement (120b vs 20b)":"s","Learned combiner":"^","Prompt length":"D","TF-IDF ridge":"v","Heuristic features":"P","Oracle difficulty (ceiling)":"*","Random scores":"X"}
fig,ax=plt.subplots(1,2,figsize=(10,4),sharey=False)
for a,view,ttl in [(ax[0],"full","Full mixed set"),(ax[1],"mmlu","MMLU-Pro only")]:
    r=ref(view).set_index("method"); lo=r.loc["always:openai/gpt-oss-120b"]; hi=r.loc["always:anthropic/claude-sonnet-5.5"]
    a.plot([lo.usd,hi.usd],[100*lo.acc,100*hi.acc],"--",color="#8a897f",lw=1,label="Random mix of the two models",zorder=1)
    for m,lab,c in [("always:openai/gpt-oss-120b","gpt-oss-120b","#52514e"),("always:anthropic/claude-sonnet-5.5","Sonnet 5.5","#52514e"),("always:anthropic/claude-opus-5.5","Opus 5.5","#52514e"),("oracle_router","Oracle router","#0b0b0b")]:
        a.plot(r.loc[m].usd,100*r.loc[m].acc,"o",ms=6,color=c,mfc="white" if m=="oracle_router" else c,zorder=3)
        a.annotate(lab,(r.loc[m].usd,100*r.loc[m].acc),xytext=(5,-10 if m!="oracle_router" else 5),textcoords="offset points",fontsize=8,color="#0b0b0b")
    d=D[(D.view==view)&(D.target=="95")]
    for q in d.itertuples():
        if q.method in col:
            a.errorbar(q.usd,100*q.acc,xerr=q.usd_sd,yerr=100*q.acc_sd,fmt=mk[q.method],color=col[q.method],ms=7,lw=0.8,capsize=0,mec="#fcfcfb",mew=1,label=q.method,zorder=4)
    a.set_xlim(0.06,0.36); a.set_ylim(82.5,95 if view=="full" else 94); a.set_xlabel("Cost per 100 prompts (USD)"); a.set_title(ttl,loc="left",fontsize=10,color="#0b0b0b")
ax[0].set_ylabel("Accuracy (%)")
h,l=ax[1].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=4,frameon=False,fontsize=8)
fig.text(0.5,0.955,"95% operating points, mean over 8 splits (bars: 1 sd). gpt-oss-120b alone is off-scale left (79.3% / 76.2%).",ha="center",fontsize=8,color="#52514e")
fig.tight_layout(rect=(0,0.14,1,0.95)); fig.savefig("figs/fig1_cost_accuracy.png",dpi=200); plt.close()
# fig2 lift
fig,ax=plt.subplots(1,2,figsize=(10,4.2),sharex=True)
order=["Oracle difficulty (ceiling)","Self-consistency (3 votes)","Learned combiner","Disagreement (120b vs 20b)","Prompt length","Combiner + samples + subject","Combiner + samples","TF-IDF ridge","Heuristic features","Random scores"]
for a,view,ttl in [(ax[0],"full","Full mixed set"),(ax[1],"mmlu","MMLU-Pro only")]:
    S=perseed(SRC[view]["sc"]); inv={v:k for k,v in NAMES.items()}
    for i,n in enumerate(order):
        v=np.array([s.loc[inv[n]+"@95%","lift"] for s in S]); c=col.get(n,"#8a897f")
        a.scatter(v,np.full(len(v),i)+np.linspace(-0.12,0.12,len(v)),s=14,color=c,alpha=0.55,zorder=2,edgecolor="none")
        a.plot([v.mean()]*2,[i-0.3,i+0.3],color=c,lw=2.5,zorder=3)
    a.axvline(0,color="#0b0b0b",lw=0.8); a.set_yticks(range(len(order))); a.set_yticklabels(order if view=="full" else [],fontsize=8); a.invert_yaxis()
    a.set_xlabel("Lift over random mix at equal cost (pp)"); a.set_title(ttl,loc="left",fontsize=10)
fig.tight_layout(); fig.savefig("figs/fig2_lift_over_random_mix.png",dpi=200); plt.close()
# fig3 K
fig,ax=plt.subplots(1,2,figsize=(8,3.6),sharey=True)
for a,view,ttl in [(ax[0],"full","Full mixed set"),(ax[1],"mmlu","MMLU-Pro only")]:
    S=perseed(SRC[view]["k"])
    for i,(kk,lab) in enumerate([("k1","2 votes"),("k2","3 votes"),("k4","5 votes")]):
        v=np.array([s.loc[f"ext:selfcons_{kk}@95%","lift"] for s in S])
        a.bar(i,v.mean(),color="#2a78d6",width=0.5); a.scatter(np.full(len(v),i)+np.linspace(-0.12,0.12,len(v)),v,s=12,color="#0b0b0b",alpha=0.5,zorder=3,edgecolor="none")
    a.set_xticks(range(3)); a.set_xticklabels(["2 votes","3 votes","5 votes"]); a.axhline(0,color="#0b0b0b",lw=0.8); a.set_title(ttl,loc="left",fontsize=10)
ax[0].set_ylabel("Lift over random mix (pp), 95% point")
fig.tight_layout(); fig.savefig("figs/fig3_votes.png",dpi=200); plt.close()
print(open("tables.md").read())
