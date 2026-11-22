# Punt Field-Position Value: A Context-Adjusted Measure of Punter Performance

*Data: nflverse play-by-play, regular seasons 1999–2026 (2026 in progress). 57,127 clean punts, 249 punters.*

## Abstract
Traditional punting statistics (gross yards, net yards, punts inside the 20) ignore where a punt starts and how valuable the ending field position actually is. We build **Punt Field-Position Value (PFPV)**, which converts every punt's ending spot into opponent expected points (EP) and compares it with what a league-average punt from the same spot and era produces. We split PFPV into **Kick Value (KV)**, the value at the spot the ball lands or is downed, and a **Coverage** term for what happens afterwards. PFPV removes the field-position bias in net average (correlation with average start spot: +0.47 for net, +0.01 for PFPV). KV is the best punter-level performer in a Monte-Carlo test of recovering true talent from a 70-punt season, but net and gross averages remain more repeatable year to year. We report both results and the limits of what public data can show.

![Pipeline](figures/fig1_pipeline.png)

## 1. Why the box score misleads
1. **Short-field penalty.** A punt from the opponent's 45 downed at the 4 gains only 41 yards, however perfect.
2. **In-20 blind spot.** A punt downed at the 19 and one at the 1 count identically, yet they are worth about 0.8 expected points apart (Figure 2).
3. **Coverage contamination.** Net yards mix the kick with gunners' tackling and the returner's skill.

![EP curve](figures/fig2_ep_curve.png)

## 2. Data
We downloaded every nflverse play-by-play file from 1999 to 2026 and kept regular-season plays with `play_type = punt`. We dropped blocked punts, penalties, lost fumbles, touchdowns, safeties and rows missing a kicker or distance (7,022 of 64,207), plus punts caught in the end zone and returned (ambiguous start spot) and 21 rows with undefined EP. The final table has 57,127 punts. Our recomputed "inside the 20" flag matches nflverse's in 99.9% of punts, which validated the geometry.

For each punt: **landing spot** = start yard line minus kick distance, from the receiver's goal line; **end spot** = landing spot (+ return yards if returned; 20 for a touchback). Outcome classes: return 42%, fair catch 25%, downed 14%, out of bounds 10%, touchback 9%. Average gross distance rose from 41.7 to 47.3 yards since 1999 (Figure 7), so every comparison below is era-adjusted.

![Era drift](figures/fig7_era_drift.png)

## 3. Method
**Expected points curve.** From 379,907 first-and-10 plays we average nflfastR's `ep` by yard line, impose monotonicity with isotonic regression, and lightly smooth. Receiving-team EP ranges from about −0.39 at its own 1 to +2.0 at midfield.

**Baseline.** A gradient-boosted regressor predicts EP at the end spot (and at the landing spot) from `(start yard line, season)`. Predictions are 5-fold cross-fitted by game so no punt is scored by a model that saw it.

**Metrics (EP points; positive = good for the punting team).**
- `PFPV = baseline_EP_end − EP_end`
- `KV = baseline_EP_land − EP_land` (punter-attributable: distance, placement, directional control)
- `Coverage = PFPV − KV` (what happened after the ball came down relative to expectation)
- `Unreturnable rate` = share of fair catches, out of bounds, downed balls and touchbacks
- `Pin score` = tiered credit: 1 for inside the 20, 2 for the 10, 3 for the 5 (touchbacks 0)

Reported per 100 punts. **Shrinkage.** Because punters have few punts, we use empirical-Bayes shrinkage `x · n/(n+k)`, with k estimated by method of moments: **k = 44 punts for KV, 101 for Coverage, 147 for PFPV**. The much smaller KV constant says that KV contains far more real punter signal per punt than the others.

## 4. Results

### 4.1 Net average rewards punting from deep; PFPV does not
Across punter-seasons (≥40 punts), the correlation between a stat and the punter's average start spot (distance from the opponent goal) is **+0.47 for net**, +0.48 for gross, −0.32 for in-20 rate, and **+0.01 for PFPV**.

![Position bias](figures/fig3_position_bias.png)

### 4.2 Repeatability
Year-to-year correlations (consecutive seasons, both ≥40 punts, 633 pairs): net 0.74, gross 0.70, **KV 0.49**, coverage 0.34, unreturnable rate 0.33, PFPV 0.31, in-20 rate 0.30, in-10 rate 0.24, pin score 0.27.

Two honest takeaways. First, **gross and net are the most stable numbers, and we do not beat them on this test.** Leg strength is a durable talent, and part of net's stability is a persistent field-position effect (punters on weak offenses keep punting from deep), which is exactly the bias in 4.1. Second, the in-20 counts that fans cite are among the *least* repeatable stats, and KV is the most repeatable of the context-adjusted ones.

![Reliability](figures/fig4_reliability.png)

### 4.3 Simulation
For 30 punters with ≥300 punts since 2016, we define "true talent" as each stat computed on that punter's own punts reweighted to a **neutral league-average start-spot mix** (five distance bands). We then simulate 3,000 seasons of 70 punts drawn with replacement from each punter's real (position-biased) punts and ask how well each stat recovers true talent.

| Stat | Corr. with true talent | Picks true #1 | Top-5 overlap |
|---|---|---|---|
| **Kick Value** | **0.72** | **42%** | 0.54 |
| Gross avg | 0.69 | 35% | 0.54 |
| PFPV total | 0.69 | 25% | 0.51 |
| Net avg | 0.66 | 26% | 0.51 |
| Unreturnable % | 0.59 | 14% | 0.48 |
| In-20 rate | 0.50 | 19% | 0.36 |

*Caveat:* the simulation resamples the same punts used to define talent, so absolute levels are optimistic and the gaps between top stats (0.72 vs 0.69) are modest. The robust conclusion is the ordering at the bottom: in-20 rate is clearly the weakest signal.

![Simulation](figures/fig5_simulation.png)

### 4.4 Leaderboards
Top punters by KV per 100 punts (shrunk): **2024–26:** A. Cole (+7.7), R. Rehkow (+7.5), J. Stout (+6.4), M. Araiza (+5.0), L. Cooke (+4.9). **Career (300+ punts, 1999–2026):** S. Lechler (+15.6), T. Sauerbrun (+11.6), A. Cole (+10.3). **Best single season (50+ punts):** Lechler 2009 (+19.9), Sauerbrun 2001 (+17.8), Stonehouse 2022 (+16.6). Several career leaders rank 40th–80th in raw net average only because the league's net average has risen with the era. Full tables and the interactive version are in the dashboard.

![Kick vs coverage](figures/fig6_kick_vs_coverage.png)

## 5. Limitations
- **No hang time or tracking data.** We cannot separate a punter's directional skill from gunner quality; KV is an outcome proxy. NFL Next Gen Stats hang time and gunner tracking would sharpen it.
- **Context is partly missing.** We control for field position and era, but not weather, altitude, wind, dome, score or game state in the baseline; those could be added.
- **EP is an average.** The first-and-10 EP curve ignores clock and score; win-probability-based value would be better for late-game punts.
- **Small samples.** Even with shrinkage, a single season is noisy; the repeatability r of 0.49 for KV means roughly half of a season's differences are noise.
- **Selection.** Excluding returned punts caught in the end zone and blocked punts removes rare but real events.

## 6. Conclusion
Counting punts inside the 20 is a weak signal, and net average is partly a measure of where a team's offense stalls. Valuing the ending field position in expected points fixes the bias, and separating the kick from the coverage gives a punter-level number (Kick Value) that identifies true talent as well as or slightly better than gross and net average, while crediting pinned punts by their real worth. The right next step is adding hang time and gunner data.

## Reproducibility
All code, data tables and figures are in this repository; see `README.md`.
