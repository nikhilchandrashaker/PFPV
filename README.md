# PFPV — Punt Field-Position Value

A context-adjusted punter stat built from nflverse play-by-play (regular seasons 1999–2026, 57,127 punts, 249 punters).

| Stat | What it is |
|---|---|
| **PFPV** | Opponent expected points saved at the final ball spot vs. an average punt from the same start spot and era |
| **KV (Kick Value)** | Same, measured at the spot the ball lands or is downed, before the return. The punter-level headline stat |
| **Coverage** | PFPV − KV: what the coverage unit and returner did after the ball landed |
| **Unreturnable rate** | Fair catch + out of bounds + downed + touchback |
| **Pin score** | Tiered credit: 1 inside the 20, 2 inside the 10, 3 inside the 5 |

All values are shrunk toward zero (empirical Bayes) and shown per 100 punts.

## Contents
- `paper.md`: research paper (methods, simulation, diagrams)
- `figures/`: all paper figures
- `punter_dashboard.html`: interactive dashboard (also published as an artifact)
- `data/`: `punts_scored.csv.gz` (every punt with PFPV/KV), `punter_seasons.csv`, `punters_career.csv`, `leaderboard_*.csv`, `ep_curve.csv`, `results.json`
- `src/`: pipeline

## Reproduce
```bash
pip install pandas numpy scikit-learn matplotlib
python src/01_build_dataset.py     # download nflverse pbp, extract punts (~28 files, ~20 MB each)
python src/02_clean_and_ep.py      # clean punts, fit EP curve
python src/03_pfpv_and_sim.py      # PFPV/KV, shrinkage, reliability, Monte-Carlo, leaderboards
python src/04_figures.py           # figures
python src/05_dashboard.py         # dashboard (needs src/dashboard_template.html)
```

## Data dictionary (`punts_scored.csv.gz`)
`yardline_100` start (yards from opp. goal) · `gross` kick distance · `land_own` receiver yard line where ball lands · `end_own` final receiver yard line · `outcome` return / fair_catch / downed / out_of_bounds / touchback · `ep_land`, `ep_end` opponent EP · `base_ep_*` model baseline · `pfpv`, `kick_val`, `cov_val` per-punt scores · `pin_tier` 0–3.

## Known limits
No hang time or tracking data; no weather/altitude controls; EP ignores clock and score. See paper §5. Data © nflverse (CC-BY 4.0); verify before commercial use.
