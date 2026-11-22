"""Clean punts -> compute landing/end spots; fit EP curve for receiving team (1st&10) from all plays."""
import pandas as pd, numpy as np, glob, json
from sklearn.isotonic import IsotonicRegression
# ---- EP curve: expected points (next score) for 1st & 10, by yards from opp goal ----
eps=[]
for f in sorted(glob.glob("data/raw/pbp_*.csv.gz")):
    d=pd.read_csv(f,usecols=["down","ydstogo","yardline_100","ep","season_type"],low_memory=False)
    d=d[(d.down==1)&(d.ydstogo==10)&(d.season_type=="REG")].dropna(subset=["ep","yardline_100"])
    eps.append(d[["yardline_100","ep"]])
e=pd.concat(eps); print("1st&10 plays",len(e))
g=e.groupby("yardline_100").ep.agg(["mean","count"]).reset_index()
g=g[(g.yardline_100>=1)&(g.yardline_100<=89)]
# smooth + enforce monotone (closer to goal => more EP)
iso=IsotonicRegression(increasing=False).fit(g.yardline_100,g["mean"],sample_weight=g["count"])
sm=pd.Series(iso.predict(np.arange(1,100)),index=np.arange(1,100)).rolling(5,center=True,min_periods=1).mean()
# extend 90-99 (own goal line area) by linear extrapolation of 80-89 slope
slope=np.polyfit(np.arange(80,90),sm.loc[80:89].values,1)[0]
for y in range(90,100): sm.loc[y]=sm.loc[89]+slope*(y-89)
sm.index.name="yardline_100"; sm.name="ep"
sm.reset_index().to_csv("data/ep_curve.csv",index=False)
EP=lambda own: np.interp(100-np.asarray(own,float),sm.index.values,sm.values)  # own = receiving team's own yard line (1-99)
# ---- clean punts ----
p=pd.read_csv("data/punts_raw.csv.gz",low_memory=False)
n0=len(p)
p=p[(p.season_type=="REG")]
for c in ["punt_blocked","punt_out_of_bounds","punt_downed","punt_fair_catch","touchback","penalty","fumble_lost","safety","punt_in_endzone","punt_inside_twenty"]:
    p[c]=p[c].fillna(0).astype(int)
p["td"]=p.td_team.notna().astype(int)
bad=(p.punt_blocked==1)|(p.penalty==1)|(p.fumble_lost==1)|(p.safety==1)|(p.td==1)|p.kick_distance.isna()|p.yardline_100.isna()|p.punter_player_id.isna()
print("dropped (blocked/penalty/fumble/td/missing):",bad.sum(),"of",len(p)); p=p[~bad].copy()
p["return_yards"]=p.return_yards.fillna(0)
p["land_opp"]=p.yardline_100-p.kick_distance            # yards from opp goal where ball lands
p["land_own"]=p.land_opp.clip(1,99)                      # receiving team's own yard line at landing (= distance from the goal they defend)
def outcome(r):
    if r.touchback==1: return "touchback"
    if r.punt_out_of_bounds==1: return "out_of_bounds"
    if r.punt_fair_catch==1: return "fair_catch"
    if r.punt_downed==1: return "downed"
    return "return"
p["outcome"]=p.apply(outcome,axis=1)
p["end_own"]=np.where(p.outcome=="touchback",20,np.where(p.outcome=="return",p.land_own+p.return_yards,p.land_own)).clip(1,99)
p.loc[(p.outcome=="return")&(p.land_opp<=0),"end_own"]=np.nan  # caught in end zone: ambiguous, drop
p=p.dropna(subset=["end_own"])
p["net_yards"]=p.yardline_100-p.end_own   # true field-position gain (touchback ends at the 20)
p["gross"]=p.kick_distance
p["ep_land"]=EP(p.land_own); p["ep_end"]=EP(p.end_own)
p["unreturnable"]=p.outcome.isin(["fair_catch","out_of_bounds","downed","touchback"]).astype(int)
p["in20"]=((p.end_own<20)&(p.outcome!="touchback")).astype(int)
p["in10"]=((p.end_own<=10)&(p.outcome!="touchback")).astype(int)
p["in5"]=((p.end_own<=5)&(p.outcome!="touchback")).astype(int)
print("check in20 agrees w/ nflverse flag:",(p.in20==p.punt_inside_twenty).mean().round(3))
keep=["game_id","season","week","qtr","posteam","defteam","punter_player_id","punter_player_name","yardline_100","gross","land_own","return_yards","end_own","net_yards","outcome","unreturnable","in20","in10","in5","ep_land","ep_end","score_differential","game_seconds_remaining"]
p[keep].to_csv("data/punts_clean.csv.gz",index=False)
print("clean punts",len(p),"| seasons",p.season.min(),"-",p.season.max(),"| punters",p.punter_player_id.nunique())
print(p.outcome.value_counts(normalize=True).round(3).to_dict())
print("EP at recv own 1/10/20/30/50:",[round(float(EP(x)),2) for x in (1,10,20,30,50)])
