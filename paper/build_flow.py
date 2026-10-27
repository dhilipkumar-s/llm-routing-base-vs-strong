import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
plt.rcParams["font.family"]="DejaVu Sans"
C={"start":("#fcfcfb","#0b0b0b"),"base":("#eaf2fd","#2a78d6"),"step":("#f4f3ef","#8a897f"),"dec":("#fff4d6","#eda100"),"score":("#efe9fb","#6250d6"),"cheap":("#e0f5ec","#1baf7a"),"strong":("#fde9e1","#eb6834"),"data":("#e8f1ee","#3d8b6e")}
def box(ax,x,y,w,h,txt,kind,fs=7,bold_first=True):
    fc,ec=C[kind]; ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.0,rounding_size=0.06",fc=fc,ec=ec,lw=1.3))
    lines=txt.split("\n"); n=len(lines); step=fs*1.55/72
    y0=y+h/2+(n-1)*step/2
    for i,l in enumerate(lines):
        ax.text(x+w/2,y0-i*step,l,ha="center",va="center",fontsize=fs,fontweight="bold" if (i==0 and bold_first) else "normal",color="#0b0b0b")
def arrow(ax,x1,y1,x2,y2,label=None,lx=0,ly=0.08):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",color="#52514e",lw=1.1,shrinkA=0,shrinkB=0,mutation_scale=9))
    if label: ax.text((x1+x2)/2+lx,(y1+y2)/2+ly,label,fontsize=6.5,ha="center",va="bottom",color="#52514e",style="italic")
# ---------- Part 1
fig,ax=plt.subplots(figsize=(7.7,4.6)); ax.set_xlim(0,7.7); ax.set_ylim(0,4.6); ax.axis("off")
box(ax,0.05,1.95,0.8,0.7,"Prompt\narrives","start")
lanes=[(3.55,"A. Prompt-only scorer\nlength, heuristics, TF-IDF\nno model call yet","step"),
       (2.0,"B. LLM judge\nlow-cost judge model\nrates the prompt 1-10\n1 judge call","step"),
       (0.35,"C. Probe the base model\nself-consistency / disagreement\nbase model answers 3 times;\nvotes -> disagreement score","base")]
for y,t,k in lanes:
    h=0.8 if t.count("\n")<3 else 0.95
    yy=y if h==0.8 else y-0.075
    box(ax,1.2,yy,2.0,h,t,k,fs=6.8)
    arrow(ax,0.85,2.3,1.2,yy+h/2); arrow(ax,3.2,yy+h/2,3.45,2.3)
box(ax,3.45,1.95,1.2,0.7,"Complexity score\nhigher = harder","score",fs=6.4)
arrow(ax,4.65,2.3,4.8,2.3)
# threshold diamond
cx,cy=5.25,2.3; ax.add_patch(Polygon([[cx-0.45,cy],[cx,cy+0.55],[cx+0.45,cy],[cx,cy-0.55]],fc=C["dec"][0],ec=C["dec"][1],lw=1.3))
ax.text(cx,cy+0.06,"Score above\nthreshold?",ha="center",va="center",fontsize=6.5,fontweight="bold")
box(ax,6.0,3.05,1.62,1.1,"Base model answers\ngpt-oss-120b\nA, B: 1 call\nC: reuses probe answer","cheap",fs=6.4)
box(ax,6.0,0.45,1.62,1.1,"Strong model answers\nClaude Sonnet 5.5\n(C: probe calls\nalready paid for)","strong",fs=6.6)
arrow(ax,cx+0.28,cy+0.28,6.0,3.6,"No: looks easy",lx=-0.5,ly=0.0)
arrow(ax,cx+0.28,cy-0.28,6.0,1.0,"Yes: looks hard",lx=-0.6,ly=-0.2)
ax.text(3.8,0.05,"Threshold is picked on training data. Cost of a prompt = the scorer's own calls + the model that finally answers.",fontsize=7,ha="center",color="#0b0b0b",style="italic")
fig.savefig("figs/fig_flow_part1_routing_paper.png",dpi=220,bbox_inches="tight",facecolor="#fcfcfb"); plt.close()
# ---------- Part 2
fig,ax=plt.subplots(figsize=(7.4,3.5)); ax.set_xlim(0,7.4); ax.set_ylim(0,3.5); ax.axis("off")
W,H=1.62,1.15; xs=[0.05,1.95,3.85,5.75]
r1=[("1. Prompts\n790 with known answers\nGSM8K 30, MATH-500 100\nMMLU-Pro 600, AIME 60","data"),
    ("2. Collect answers\nevery model answers every\nprompt once (7 models),\nplus extra base-model\nsamples; saved to files","data"),
    ("3. Mark answers\nright or wrong by\ncomparing with the\ngold answer","data"),
    ("4. Split\n50/50 into train and test;\nrepeated for 8 random\nsplits","step")]
for x,(t,k) in zip(xs,r1): box(ax,x,2.1,W,H,t,k,fs=6.5)
for a,b in zip(xs[:-1],xs[1:]): arrow(ax,a+W,2.675,b,2.675)
r2=[("5. On TRAIN\nbuild each scorer (learned\nones use 5-fold out-of-fold\nscores); pick the cheapest\nthreshold keeping 95% or\n99% of Sonnet's accuracy","base"),
    ("6. On TEST\nscore each prompt, apply the\nthreshold, look up the saved\nanswer of the chosen model,\nadd the scorer's own cost","step"),
    ("7. Report\naccuracy, cost, savings vs\nSonnet, lift over a random\nmix at equal cost, how often\nthe target is met on test","score")]
xs2=[0.05,2.65,5.25]; W2=2.1; H2=1.35
for x,(t,k) in zip(xs2,r2): box(ax,x,0.25,W2,H2,t,k,fs=6.5)
for a,b in zip(xs2[:-1],xs2[1:]): arrow(ax,a+W2,0.925,b,0.925)
# connector from split box down, across, and into the TRAIN box (three straight segments)
ax.plot([6.56,6.56],[2.1,1.88],color="#52514e",lw=1.1); ax.plot([6.56,1.1],[1.88,1.88],color="#52514e",lw=1.1)
ax.annotate("",xy=(1.1,0.25+H2),xytext=(1.1,1.88),arrowprops=dict(arrowstyle="-|>",color="#52514e",lw=1.1,mutation_scale=9))
fig.savefig("figs/fig_flow_part2_evaluation_paper.png",dpi=220,bbox_inches="tight",facecolor="#fcfcfb"); plt.close()
