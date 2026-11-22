import pandas as pd, numpy as np, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"axes.grid":True,"grid.alpha":.25})
NAVY,TEAL,ORG,GR="#1b2a41","#2a9d8f","#e76f51","#8d99ae"
R=json.load(open("data/results.json")); ep=pd.read_csv("data/ep_curve.csv"); ps=pd.read_csv("data/punter_seasons.csv"); rc=pd.read_csv("data/leaderboard_recent_2024_2026.csv")
p=pd.read_csv("data/punts_scored.csv.gz")
# 1 pipeline diagram
fig,ax=plt.subplots(figsize=(11,3.3)); ax.axis("off"); ax.set_xlim(0,11); ax.set_ylim(0,3.3)
boxes=[("nflverse\nplay-by-play\n1999–2026",.1),("Punt table\n57k clean punts\nland / end spot",2.3),("EP curve\n1st&10 by field\nposition (380k plays)",4.5),("Baseline model\nE[EP | start, era]\ncross-fitted GBM",6.7),("Scores\nPFPV · KV · Coverage\n+ shrinkage",8.9)]
for i,(t,x) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((x,1.1),1.9,1.5,boxstyle="round,pad=.05",fc=[NAVY,NAVY,TEAL,TEAL,ORG][i],ec="none"))
    ax.text(x+.95,1.85,t,ha="center",va="center",color="white",fontsize=9.5,weight="bold")
    if i<4: ax.annotate("",xy=(x+2.25,1.85),xytext=(x+1.95,1.85),arrowprops=dict(arrowstyle="-|>",color=GR,lw=2))
ax.text(5.5,.45,"Outputs: leaderboard · reliability tests · Monte-Carlo simulation · dashboard",ha="center",color=NAVY,fontsize=10)
fig.savefig("figures/fig1_pipeline.png",bbox_inches="tight"); plt.close()
# 2 EP curve
fig,ax=plt.subplots(figsize=(7.5,4)); own=100-ep.yardline_100; ax.plot(own,ep.ep,color=NAVY,lw=2.5)
f=lambda x:np.interp(x,own[::-1],ep.ep[::-1])
for x,l,c in [(1,"downed at the 1",TEAL),(19,"downed at the 19",ORG)]: ax.scatter([x],[f(x)],s=70,color=c,zorder=5); ax.annotate(f"{l}\nEP = {f(x):+.2f}",(x,f(x)),xytext=(8,1.6) if x==1 else (30,0.0),fontsize=9,color=c,arrowprops=dict(arrowstyle="-",color=c))
ax.set_xlabel("Receiving team's starting yard line (own territory)"); ax.set_ylabel("Expected points (1st & 10)"); ax.set_xlim(0,80)
ax.set_title("Same “inside the 20” box, ~0.8 points apart",loc="left",weight="bold"); fig.savefig("figures/fig2_ep_curve.png",bbox_inches="tight"); plt.close()
# 3 position bias
a=ps[ps.punts>=40]; fig,axs=plt.subplots(1,2,figsize=(9,3.8),sharex=True)
for ax,(c,t,col) in zip(axs,[("net","Net average (yds)",ORG),("pfpv100","PFPV (pts per 100 punts)",TEAL)]):
    ax.scatter(a.avg_start,a[c],s=14,alpha=.45,color=col); m=np.polyfit(a.avg_start,a[c],1); xs=np.linspace(a.avg_start.min(),a.avg_start.max(),50); ax.plot(xs,np.polyval(m,xs),color=NAVY,lw=2)
    ax.set_title(f"{t}   r = {a.avg_start.corr(a[c]):+.2f}",loc="left",fontsize=10,weight="bold"); ax.set_xlabel("Avg punt start (yds from opp. goal; higher = deeper in own end)")
fig.suptitle("Net average rewards punting from deep; PFPV does not",x=.01,ha="left",weight="bold"); fig.savefig("figures/fig3_position_bias.png",bbox_inches="tight"); plt.close()
# 4 reliability
rel=R["reliability"]; order=[("net","Net avg"),("gross","Gross avg"),("kick","Kick Value"),("cov","Coverage"),("unret","Unreturnable %"),("pfpv","PFPV total"),("in20","In-20 rate"),("in10","In-10 rate")]
fig,ax=plt.subplots(figsize=(7.5,3.8)); v=[rel[k] for k,_ in order]; ax.barh([n for _,n in order][::-1],v[::-1],color=[TEAL if n in("Kick Value","PFPV total") else GR for _,n in order][::-1])
for i,x in enumerate(v[::-1]): ax.text(x+.01,i,f"{x:.2f}",va="center",fontsize=9)
ax.set_xlabel("Year-to-year correlation (same punter, consecutive seasons, ≥40 punts)"); ax.set_title("How repeatable is each stat?",loc="left",weight="bold"); fig.savefig("figures/fig4_reliability.png",bbox_inches="tight"); plt.close()
# 5 simulation
S=R["sim"]; nm={"gross":"Gross","net_yards":"Net","in20":"In-20 rate","unreturnable":"Unreturnable %","pfpv":"PFPV total","kick_val":"Kick Value"}
fig,axs=plt.subplots(1,2,figsize=(9,3.6))
for ax,(k,t) in zip(axs,[("mean_corr","Corr. with neutral true talent"),("top1","Picks the true #1 punter")]):
    ks=list(nm); v=[S[x][k] for x in ks]; ax.bar([nm[x] for x in ks],v,color=[TEAL if x=="kick_val" else GR for x in ks]); ax.set_title(t,loc="left",weight="bold",fontsize=10); ax.tick_params(axis="x",rotation=40)
    for i,x in enumerate(v): ax.text(i,x+.01,f"{x:.2f}",ha="center",fontsize=8)
fig.suptitle("Simulation: 3,000 seasons of 70 punts, 30 punters",x=.01,ha="left",weight="bold"); fig.tight_layout(); fig.savefig("figures/fig5_simulation.png",bbox_inches="tight"); plt.close()
# 6 kick vs coverage
fig,ax=plt.subplots(figsize=(7.5,5.2)); ax.scatter(rc.kv100,rc.cov100,s=rc.punts/2.5,color=TEAL,alpha=.6,ec="white"); ax.axhline(0,color=GR,lw=1); ax.axvline(0,color=GR,lw=1)
for _,r in pd.concat([rc.nlargest(5,"kv100"),rc.nsmallest(4,"kv100"),rc.nlargest(3,"cov100"),rc.nsmallest(3,"cov100")]).drop_duplicates("name").iterrows(): ax.annotate(r["name"],(r.kv100,r.cov100),xytext=(4,3),textcoords="offset points",fontsize=8)
ax.set_xlabel("Kick Value (EP pts / 100 punts) — the punter"); ax.set_ylabel("Coverage value (pts / 100 punts) — unit & returner luck"); ax.set_title("Separating the kick from the coverage, 2024–26",loc="left",weight="bold"); fig.savefig("figures/fig6_kick_vs_coverage.png",bbox_inches="tight"); plt.close()
# 7 era drift
y=p.groupby("season")[["gross","net_yards","in20"]].mean(); fig,ax=plt.subplots(figsize=(7.5,3.6)); ax.plot(y.index,y.gross,color=NAVY,lw=2,label="Gross"); ax.plot(y.index,y.net_yards,color=ORG,lw=2,label="Net"); ax.legend(frameon=False); ax.set_ylabel("Yards"); ax.set_title("Punting has changed: era must be controlled",loc="left",weight="bold"); fig.savefig("figures/fig7_era_drift.png",bbox_inches="tight"); plt.close()
print("ok")
