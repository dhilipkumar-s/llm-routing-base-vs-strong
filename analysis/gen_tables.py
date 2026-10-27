from core import *
import pandas as pd, pickle
W=World(); W.add_ext("effort","effort.jsonl")
OUT=str(ROOT/"paper"/"tex")+"/"
lab={"selfcons3":"Self-consistency (3 votes)","disagree":"Disagreement (120b vs 20b)","combo_sc_cat":"Combiner + samples + subject","length":"Prompt length",
     "tfidf_ridge":"TF-IDF ridge","heuristic":"Heuristic features","judge":"LLM judge","oracle_pool":"Oracle difficulty (pool share; paper's earlier ceiling)",
     "oracle_pair":"Oracle for this pair (base wrong, Sonnet right)","random":"Random scores"}
f=lambda x,d=1:f"{x:.{d}f}".replace("-","$-$")
# ---- Table CI (paper rule, 100 splits)
b=pd.read_csv("boot_lift.csv"); b=b[b.rule=="paper_grid11"].set_index("router")
r=pd.read_csv("rules_100splits.csv"); r=r[r.rule=="paper_grid11"].set_index("router")
order=["selfcons3","disagree","combo_sc_cat","length","tfidf_ridge","heuristic","judge","oracle_pool","oracle_pair","random"]
L=[r"\begin{table}[!htbp]",r"\centering\small",
r"\caption{Routers at the 95\% target over 100 random 50/50 splits (full set, the paper's threshold rule). Intervals are 95\% prompt-level bootstrap intervals (4000 draws) on each prompt's average result over the splits in which it was in the test half. Held = share of splits in which test retention met 95\%. Savings are relative to always using Sonnet.}\label{tab:ci}",
r"\begin{tabular}{lrrrr}",r"\toprule","Router & Savings (\\%) [95\\% CI] & Lift (pp) [95\\% CI] & Retention (\\%) & Held \\\\",r"\midrule"]
for k in order:
    if k=="oracle_pool": L.append(r"\midrule")
    L.append(f"{lab[k]} & {f(b.loc[k,'sav'])} [{f(b.loc[k,'slo'])}, {f(b.loc[k,'shi'])}] & {f(b.loc[k,'lift'])} [{f(b.loc[k,'lo'])}, {f(b.loc[k,'hi'])}] & {f(r.loc[k,'retention'])} & {100*r.loc[k,'held_frac']:.0f}\\% \\\\")
L+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
open(OUT+"tab_ci.tex","w").write("\n".join(L)+"\n")
# ---- Margin table
R=pd.read_csv("rules_100splits.csv")
rules=[("paper_grid11","Paper rule (11-step grid)"),("fine_grid","Same rule, 101-step grid"),("margin+2","Target + 2 points"),("ucb","Upper confidence bound")]
sub=["selfcons3","disagree","combo_sc_cat","length","oracle_pool","oracle_pair","random"]
L=[r"\begin{table}[!htbp]",r"\centering\small",
r"\caption{Effect of the threshold rule at the 95\% target (100 splits, full set). Each cell is savings (\%) / lift (pp) / share of splits in which the target was met. ``Target + 2 points'' picks the cheapest threshold reaching 97\% of Sonnet's training accuracy; ``upper confidence bound'' requires a one-sided 95\% normal upper bound on the training accuracy shortfall to stay within the allowed 5\%.}\label{tab:rules}",
r"\resizebox{\textwidth}{!}{\begin{tabular}{l"+"c"*len(rules)+"}",r"\toprule","Router & "+" & ".join(n for _,n in rules)+r" \\",r"\midrule"]
for k in sub:
    cells=[]
    for rn,_ in rules:
        x=R[(R.rule==rn)&(R.router==k)].iloc[0]; cells.append(f"{f(x.savings)} / {f(x.lift_pp)} / {100*x.held_frac:.0f}\\%")
    L.append(lab[k].replace(" (pool share; paper's earlier ceiling)"," (pool share)").replace(" (base wrong, Sonnet right)","")+" & "+" & ".join(cells)+r" \\")
L+=[r"\bottomrule",r"\end{tabular}}",r"\end{table}"]
open(OUT+"tab_rules.tex","w").write("\n".join(L)+"\n")
# ---- Table 4 recomputed
L_=np.array([len(p) for p in W.prompts],float); fold=np.random.RandomState(3).permutation(W.n)%5; tf=np.zeros(W.n)
for fo in range(5):
    va,trn=np.where(fold==fo)[0],np.where(fold!=fo)[0]
    m=S.TfidfRidgeScorer(); m.fit([W.prompts[k] for k in trn],W.d_pool[trn],None); tf[va]=m.score(None,[W.prompts[k] for k in va])
sc={"length":L_,"heuristic":S.HeuristicScorer().score(None,W.prompts),"tfidf_ridge":tf,"judge":W.ext["judge"][0],"effort":W.ext["effort"][0],"disagree":W.ext["disagree"][0],
    "selfcons3":W.ext["selfcons3"][0],"combo_sc_cat":W.ext["combo_sc_cat"][0],"oracle_pair":W.gain+1e-9*np.arange(W.n),"random":np.random.RandomState(1).rand(W.n)}
calls={"length":0,"heuristic":0,"tfidf_ridge":0,"judge":1,"effort":1,"disagree":2,"selfcons3":2,"combo_sc_cat":"2--3","oracle_pair":"--","random":0}
lab2=dict(lab); lab2["effort"]="Llama-3.1-8B output length"; lab2["oracle_pair"]="Oracle for this pair (not realisable)"; lab2["disagree"]="Disagreement (gpt-oss-120b vs 20b)"
fail=1-W.okb
def cols(m):
    return lambda s:(S.spearman(s[m],fail[m]),S.auc(s[m],fail[m]==1),S.auc(s[m],W.gain[m]==1))
ma=np.ones(W.n,bool); mm=W.src=="mmlu_pro"
L=[r"\begin{table}[!htbp]",r"\centering\small",
r"\caption{Signal quality of lightweight scorers for the pair used in the routing experiments, computed on all prompts (learned scorers use 5-fold out-of-fold scores; no train/test split). Spearman and the first AUC are against gpt-oss-120b's own failures; the second AUC is for ``needs Sonnet'' (gpt-oss-120b wrong, Sonnet right). Calls = model calls needed before the routing decision.}\label{tab:signal}",
r"\begin{tabular}{lcrrrrrr}",r"\toprule"," & & \\multicolumn{3}{c}{Full set} & \\multicolumn{3}{c}{MMLU-Pro} \\\\",r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
"Scorer & Calls & $\\rho$ & AUC fail & AUC gain & $\\rho$ & AUC fail & AUC gain \\\\",r"\midrule",
r"\multicolumn{8}{l}{\emph{Prompt characteristics (read the prompt only)}} \\"]
def row(k):
    a=cols(ma)(sc[k]); b_=cols(mm)(sc[k]); return f"{lab2[k]} & {calls[k]} & "+" & ".join(f(x,2) for x in a+b_)+r" \\"
for k in ["length","heuristic","tfidf_ridge","judge"]: L.append(row(k))
L+=[r"\midrule",r"\multicolumn{8}{l}{\emph{Model behaviour (needs model answers)}} \\"]
for k in ["effort","disagree","selfcons3","combo_sc_cat"]: L.append(row(k))
L+=[r"\midrule",r"\multicolumn{8}{l}{\emph{References}} \\"]
for k in ["oracle_pair","random"]: L.append(row(k))
L+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
open(OUT+"tab_signal.tex","w").write("\n".join(L)+"\n")
print("\n".join(L))
# ---- per-benchmark table
pool=pickle.load(open("pool.pkl","rb"))["paper_grid11"]
L=[r"\begin{table}[!htbp]",r"\centering\small",
r"\caption{Results by benchmark at the 95\% target (full set, 100 splits, paper threshold rule, prompts pooled by benchmark). Share escalated is the percentage of prompts sent to Sonnet; accuracy is the router's accuracy. Always-base and always-Sonnet accuracy are shown for reference.}\label{tab:bench}",
r"\begin{tabular}{llrrrrr}",r"\toprule","Benchmark (n) & Router & Escalated (\\%) & Accuracy (\\%) & gpt-oss-120b & Sonnet & Cost / Sonnet \\\\",r"\midrule"]
nm_={"aime":"AIME","gsm8k":"GSM8K","math500":"MATH-500","mmlu_pro":"MMLU-Pro"}
for s in ["mmlu_pro","aime","math500","gsm8k"]:
    m=W.src==s; first=True
    for k in ["selfcons3","disagree","length"]:
        p=pool[k]; c=np.maximum(p["cnt"],1)
        e=100*(p["esc"]/c)[m].mean(); a=100*(p["acc"]/c)[m].mean(); cs=(p["cost"]/c)[m].sum()/W.Cs[m].sum()
        L.append((f"{nm_[s]} ({m.sum()})" if first else "")+f" & {lab[k].split(' (')[0] if k!='disagree' else 'Disagreement'} & {e:.0f} & {a:.1f} & {100*W.okb[m].mean():.1f} & {100*W.oks[m].mean():.1f} & {100*cs:.0f}\\% \\\\"); first=False
    L.append(r"\addlinespace")
L+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
open(OUT+"tab_bench.tex","w").write("\n".join(L)+"\n")
