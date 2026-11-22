"""PFPV model, punter tables, reliability tests, and Monte-Carlo simulation."""
import pandas as pd, numpy as np, json
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
rng=np.random.default_rng(7)
p=pd.read_csv("data/punts_clean.csv.gz").dropna(subset=["ep_end","ep_land"]).reset_index(drop=True)
# ---------- 1. Baseline: expected EP for a league-average punt from this spot & era (cross-fitted by game) ----------
X=p[["yardline_100","season"]].values
for tgt in ["ep_end","ep_land"]:
    oof=np.zeros(len(p))
    for tr,te in GroupKFold(5).split(X,groups=p.game_id):
        m=HistGradientBoostingRegressor(max_depth=3,learning_rate=0.05,max_iter=250,min_samples_leaf=300,
                                         monotonic_cst=[0,0]).fit(X[tr],p[tgt].values[tr]); oof[te]=m.predict(X[te])
    p["base_"+tgt]=oof
p["pfpv"]=p.base_ep_end-p.ep_end            # + = punter/coverage pushed opponent's expected points DOWN vs average
p["kick_val"]=p.base_ep_land-p.ep_land      # value at the spot the ball lands / is downed (pre-return)
p["cov_val"]=p.pfpv-p.kick_val              # what happened after landing vs expectation (coverage / returner)
p["pin_tier"]=np.select([p.in5==1,p.in10==1,p.in20==1],[3,2,1],0)
# ---------- 2. Punter-season and career tables ----------
p["season_c"]=p.season
name=p.groupby("punter_player_id").punter_player_name.agg(lambda s:s.mode().iat[0])
def agg(g):
    return pd.Series(dict(punts=len(g),gross=g.gross.mean(),net=g.net_yards.mean(),avg_start=g.yardline_100.mean(),
        in20=g.in20.mean(),in10=g.in10.mean(),in5=g.in5.mean(),unret=g.unreturnable.mean(),touchback=(g.outcome=="touchback").mean(),
        pin_score=g.pin_tier.mean(),pfpv=g.pfpv.mean(),kick=g.kick_val.mean(),cov=g.cov_val.mean(),
        first=g.season_c.min(),last=g.season_c.max(),team=g.posteam.iloc[-1]))
ps=p.groupby(["punter_player_id","season"]).apply(agg).reset_index(); ps["name"]=ps.punter_player_id.map(name)
sw2=p.pfpv.var()
# empirical-Bayes shrinkage of mean PFPV
cp=p.groupby("punter_player_id").apply(agg).reset_index(); cp["name"]=cp.punter_player_id.map(name)
big=cp[cp.punts>=150]
KS={}
for m_,col in [("pfpv","pfpv"),("kick","kick_val"),("cov","cov_val")]:
    s2=p[col].var(); t2=max(big[m_].var()-(s2/big.punts).mean(),1e-6); KS[m_]=s2/t2
k=KS["pfpv"]; tau2=sw2/k
def shrink(df,**_):
    for m_ in KS: df[m_+"_adj"]=df[m_]*df.punts/(df.punts+KS[m_])
    return df
cp=shrink(cp); ps=shrink(ps)
# ---------- 3. Reliability: year-to-year correlation of punter-season stats (>=40 punts both years) ----------
a=ps[ps.punts>=40].copy(); a["prev"]=a.season-1
b=a.merge(a,left_on=["punter_player_id","season"],right_on=["punter_player_id","prev"],suffixes=("","_n"))
rel={s:float(b[s].corr(b[s+"_n"])) for s in ["gross","net","in20","in10","unret","pin_score","pfpv","kick","cov"]}
print("year-to-year r:",{k_:round(v,2) for k_,v in rel.items()}, "pairs",len(b))
# position bias: corr of season stat with avg start spot (yardline_100 higher = punting from own territory)
bias={s:float(a[s].corr(a.avg_start)) for s in ["gross","net","in20","in10","unret","pfpv"]}
print("corr with avg start (position bias):",{k_:round(v,2) for k_,v in bias.items()})
# ---------- 4. Monte-Carlo: how well does a 70-punt season recover true talent? ----------
rec=p[p.season>=2016]; pun=rec.groupby("punter_player_id").filter(lambda g:len(g)>=300)
bins=pd.cut(rec.yardline_100,[0,50,60,70,80,101],labels=False); rec_mix=bins.value_counts(normalize=True).sort_index().values
pun=pun.assign(b=pd.cut(pun.yardline_100,[0,50,60,70,80,101],labels=False))
ids=pun.punter_player_id.unique(); STATS=["gross","net_yards","in20","unreturnable","pfpv","kick_val"]
truth={s:[] for s in STATS}; samp=[]
for i in ids:
    g=pun[pun.punter_player_id==i]; share=g.b.value_counts(normalize=True).reindex(range(5)).fillna(0).values
    w=np.where(share>0,rec_mix/np.where(share>0,share,1),0); w=g.b.map(dict(enumerate(w))).values; w=w/w.sum()
    for s in STATS: truth[s].append(float((g[s].values*w).sum()))       # true talent under a NEUTRAL league-average start-spot mix
    samp.append(g[STATS].values)
N=3000; n=70; res={s:[] for s in STATS}; top1={s:0 for s in STATS}; top5={s:0 for s in STATS}
T={s:np.array(truth[s]) for s in STATS}
sim_corr={s:[] for s in STATS}
for _ in range(N):
    vals=np.array([m[rng.integers(0,len(m),n)].mean(0) for m in samp])
    for j,s in enumerate(STATS):
        sim_corr[s].append(np.corrcoef(vals[:,j],T[s])[0,1])
        top1[s]+=vals[:,j].argmax()==T[s].argmax(); top5[s]+= len(set(np.argsort(-vals[:,j])[:5])&set(np.argsort(-T[s])[:5]))
sim={s:dict(mean_corr=float(np.mean(sim_corr[s])),p10=float(np.percentile(sim_corr[s],10)),p90=float(np.percentile(sim_corr[s],90)),
            top1=top1[s]/N,top5_overlap=top5[s]/N/5) for s in STATS}
print("simulation (70-punt season vs neutral true talent), punters:",len(ids))
for s in STATS: print(f"  {s:13s} corr={sim[s]['mean_corr']:.2f}  top1={sim[s]['top1']:.2f}  top5 overlap={sim[s]['top5_overlap']:.2f}")
# ---------- 5. Leaderboards ----------
for df in (cp,ps):
    df["pfpv100"]=df.pfpv_adj*100; df["raw100"]=df.pfpv*100; df["kv100"]=df.kick_adj*100; df["cov100"]=df.cov_adj*100
cp["rank_kv"]=cp.kick_adj.rank(ascending=False); cp["rank_net"]=cp.net.rank(ascending=False)
career=cp[cp.punts>=300].sort_values("kick_adj",ascending=False)
rc=ps[(ps.season>=2024)].groupby("punter_player_id").apply(lambda g:pd.Series(dict(punts=g.punts.sum(),
    net=np.average(g.net,weights=g.punts),gross=np.average(g.gross,weights=g.punts),unret=np.average(g.unret,weights=g.punts),
    in10=np.average(g.in10,weights=g.punts),pfpv=np.average(g.pfpv,weights=g.punts),kick=np.average(g.kick,weights=g.punts),
    cov=np.average(g["cov"],weights=g.punts),team=g.team.iloc[-1]))).reset_index()
rc["name"]=rc.punter_player_id.map(name); rc=shrink(rc); rc["pfpv100"]=rc.pfpv_adj*100; rc["kv100"]=rc.kick_adj*100; rc["cov100"]=rc.cov_adj*100
rc["rank_kv"]=rc.kick_adj.rank(ascending=False); rc["rank_net"]=rc.net.rank(ascending=False)
recent=rc[rc.punts>=80].sort_values("kick_adj",ascending=False)
career=career.assign(rank_kv=range(1,len(career)+1)); career["rank_net"]=career.net.rank(ascending=False)
recent=recent.assign(rank_kv=range(1,len(recent)+1)); recent["rank_net"]=recent.net.rank(ascending=False)
seasons=ps[ps.punts>=50].sort_values("kick_adj",ascending=False)
cp.to_csv("data/punters_career.csv",index=False); ps.to_csv("data/punter_seasons.csv",index=False)
career.to_csv("data/leaderboard_career.csv",index=False); recent.to_csv("data/leaderboard_recent_2024_2026.csv",index=False)
seasons.head(50).to_csv("data/leaderboard_best_seasons.csv",index=False)
p.to_csv("data/punts_scored.csv.gz",index=False)
json.dump(dict(k=float(k),shrink_k={a:float(b) for a,b in KS.items()},sw2=float(sw2),reliability=rel,bias=bias,sim=sim,n_punts=len(p),n_punters=int(p.punter_player_id.nunique()),
   seasons=[int(p.season.min()),int(p.season.max())]),open("data/results.json","w"),indent=1)
cols=["name","team","punts","net","kv100","pfpv100","cov100","rank_kv","rank_net"]
print("\nshrinkage k (punts):",round(k,1))
print("\nCAREER top 10 (>=300 punts)\n",career.head(10)[["name","punts","net","kv100","pfpv100","rank_kv","rank_net"]].round(2).to_string(index=False))
print("\nRECENT 2024-26 top 10 (>=80)\n",recent.head(10)[cols].round(2).to_string(index=False))
print("\nRECENT bottom 3\n",recent.tail(3)[cols].round(2).to_string(index=False))
