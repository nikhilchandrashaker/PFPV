import pandas as pd, json
d="data/"; R=json.load(open(d+"results.json")); ep=pd.read_csv(d+"ep_curve.csv")
rd=lambda f,c:pd.read_csv(d+f)[c].round(3).where(lambda x:x.notna(),None).to_dict("records")
base=["name","team","punts","net","gross","unret","kv100","pfpv100","cov100"]
D=dict(recent=rd("leaderboard_recent_2024_2026.csv",base+["rank_kv","rank_net"]),career=rd("leaderboard_career.csv",base+["rank_kv","rank_net","first","last"]),
 seasons=rd("leaderboard_best_seasons.csv",base+["season"]),
 ep=[round(x,3) for x in ep.sort_values("yardline_100",ascending=False).ep.tolist()],  # index i -> own yard line i+1
 rel=R["reliability"],sim=R["sim"],bias=R["bias"],n=R["n_punts"],seasonsRange=R["seasons"])
html=open("src/dashboard_template.html").read().replace("/*DATA*/null",json.dumps(D,separators=(",",":")))
open("/mnt/user-data/outputs/punter_dashboard.html","w").write(html); print(len(html)//1024,"KB")
