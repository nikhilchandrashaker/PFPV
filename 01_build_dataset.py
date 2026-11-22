"""Download nflverse play-by-play (1999-present) and extract every punt into one clean table."""
import pandas as pd, numpy as np, urllib.request, os, sys
from concurrent.futures import ThreadPoolExecutor
COLS = ["game_id","season","week","season_type","play_id","posteam","defteam","yardline_100","qtr","game_seconds_remaining",
 "score_differential","play_type","punt_blocked","punt_inside_twenty","punt_in_endzone","punt_out_of_bounds","punt_downed",
 "punt_fair_catch","touchback","kick_distance","return_yards","punter_player_id","punter_player_name","punt_returner_player_id",
 "penalty","fumble_lost","wp","desc","td_team","safety"]
URL="https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.csv.gz"
def get(y):
    f=f"data/raw/pbp_{y}.csv.gz"
    try:
        if not os.path.exists(f): urllib.request.urlretrieve(URL.format(y=y),f)
        d=pd.read_csv(f,usecols=lambda c:c in COLS,low_memory=False)
        return d[d.play_type=="punt"]
    except Exception as e:
        print("skip",y,e,file=sys.stderr); return None
with ThreadPoolExecutor(6) as ex: parts=[p for p in ex.map(get,range(1999,2027)) if p is not None]
p=pd.concat(parts,ignore_index=True)
print("raw punt rows",len(p),"seasons",p.season.min(),p.season.max())
p.to_csv("data/punts_raw.csv.gz",index=False)
