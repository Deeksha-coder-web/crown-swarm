# crownswarm — crown-shyness-inspired swarm dispersion

Simulator and experiments for the project *"Crown Shyness-Inspired Decentralized
Swarm Spacing"* (BT232TD). Robots are released through one entrance into a
GPS-denied building, tunnel or mine and must spread out to cover it, **without
positions, maps, radio or a leader**. The control rules copy the two proposed
mechanisms behind crown shyness:

* **Rule A: abrasion.** Senses only bumps. A contact damages the agent's "vigour"
  (speed) and triggers a short retreat and turn away. After a quiet period,
  vigour regrows.
* **Rule B: light sensing.** Reads a weak far-red-like signal in 8 body-frame
  sectors. The signal falls off with distance and is blocked by walls and by
  closer agents. The agent steers toward the weakest side, slows as shading
  rises, and stops when shaded on every side.
* **Rule C: hybrid.** B's steering plus A's contact damage and retreat.

These are compared against baselines ordered by how much each one is allowed to sense
(the "sensing-budget ladder", see `PLAN.md`):

| Rung | Controller | Senses |
|---|---|---|
| 0 | Random walk (`RW`) | bump only |
| 1 | Rule A | bump only, plus internal memory |
| 2 | Rule B / Rule C | occluded 8-sector signal (C: plus bump) |
| 3 | Potential field (`PF`), `Boids` (no cohesion) | exact range and bearing (Boids: plus neighbour headings) |
| 4 | `Lloyd` (geodesic) | **privileged**: global positions and map |

A greedy line-of-sight placement gives a static **coverage ceiling** for each plan.

## Quick start

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python -m pytest -q tests                    # sanity tests (~15 s)
./venv/bin/python export_web.py                          # record runs for the web viewer
./venv/bin/python -m http.server 8765 --directory viewer  # then open http://localhost:8765
./venv/bin/python demo.py --controller A B --plan office  # matplotlib animation
./venv/bin/python demo.py --controller B --plan rubble --fail 0.2 --gif b.gif
./venv/bin/python tune.py --all                          # CMA-ES, ~15 min on 10 cores
./venv/bin/python experiments.py all                     # ~30 min; --seeds 5 for a quick run
./venv/bin/python analyze.py                             # results/summary.md + results/figs/*.png
```

`demo.py` and `experiments.py` use the tuned parameters in `results/params/*.json`
when they exist, and the defaults otherwise.

## Layout

| Path | What |
|---|---|
| `crownswarm/floorplans.py` | Occupancy grids at 0.1 m. **Train:** office, hall, mine. **Test (unseen):** apartment, rubble. **Application showcases (unseen):** sky (drones, windy, 3 m camera), forest, orchard, warehouse, cave. Also `open[:WxH]`. |
| `export_web.py`, `viewer/` | Records every controller on each scenario, then plays them back in an interactive web viewer (one data file per scenario, loaded on demand). |
| `crownswarm/world.py` | Numba kernels: collision-free sequential motion, rays, line of sight, occluded sector signal, geodesic Lloyd |
| `crownswarm/controllers.py` | All controllers and their tunable parameter ranges |
| `crownswarm/episode.py` | `run_episode(EpisodeConfig)`: sense, act, move, metrics |
| `crownswarm/metrics.py` | Coverage (2 m line-of-sight footprint), spacing and gaps, coverage ceiling |
| `tune.py` | CMA-ES on randomised training scenarios (common random numbers, validation against defaults) |
| `experiments.py` | `compare`, `sweep --var sway/noise/n/fail`, `ablate`, `biology`, `scaling` |
| `analyze.py` | Bootstrap 95% CIs, paired Wilcoxon tests, figures |

## Model

* Agents are unicycle robots: radius 0.3 m, top speed 0.2 m/step. They start packed
  at the entrance.
* **Wind sway** `sway` is Gaussian position noise per step (m), with 2.5× that
  noise on heading (rad). **Sensor noise** is multiplicative on signals and ranges,
  and additive (rad) on bearings.
* Controllers get **only body-frame data**: bump bearing, 8 wall-range rays,
  visible neighbours (PF and Boids only), and the sector signal (B and C only). A
  test checks that only Lloyd sees world state.
* Motion is resolved one agent at a time in random order. A step that would enter a
  wall or another agent slides or is cancelled, so bodies never overlap. A
  *collision* is counted when a pair comes into touching range (5 cm) after having
  been at least 15 cm apart, so jitter between resting neighbours isn't counted.

### Metrics
Coverage is the fraction of floor points within 2 m **and in line of sight** of an
agent. The other metrics are:
* area explored, as the cumulative coverage;
* spacing CV, the std/mean of nearest-neighbour distance;
* gap width between agent bodies;
* collisions per agent;
* path length per agent;
* time to 90% of peak coverage;
* coverage kept after 20% of agents are removed mid-run, compared with the same
  seed left intact.

### Fitness used for tuning
`coverage(2nd half) + 0.25·explored − 0.1·spacing CV − 0.01·collisions − 0.1·path/(vmax·steps)`,
averaged over 8 random training scenarios per candidate. Scenarios vary the plan,
30–90 agents, sway 0–0.05, noise 0–0.15, and whether 20% of agents fail. Every
tunable controller gets the same budget.

## Key results (30 seeds, unseen test plans; full tables in `results/summary.md`)

| Controller | Coverage | Collisions / agent | Path / agent (m) |
|---|---|---|---|
| Random walk | 0.61 | 8.6 | 66 |
| A: abrasion | 0.60 | 5.9 | 60 |
| **B: light** | **0.89** | **1.05** | 50 |
| C: hybrid | 0.89 | 1.23 | 54 |
| Potential field | 0.81 | 1.17 | 34 |
| Boids | 0.79 | 1.34 | 93 |
| Lloyd (privileged) | 0.70 | 2.01 | 39 |

1. **Rule B is the best controller at 60 agents.** It beats the potential field
   (+0.08 coverage) and Boids (+0.10), both of which get *exact* range and bearing,
   and it beats privileged Lloyd. All paired p < 1e-7. It also generalises best
   (0.93 on training plans vs 0.89 on unseen plans).
2. **Rule A matches random walk on coverage but has 31% fewer collisions**
   (p ≈ 1e-11, same bump sensor). With contact-only sensing, the abrasion memory
   buys safety, not reach. Without regrowth it collapses (coverage 0.53 → 0.30).
3. **The hybrid adds nothing over B.** CMA-ES tuned C's abrasion layer to almost
   zero (damage 0.03).
4. **Crossover points.**
   * Sensor noise: B loses 0.08 coverage from noise 0 to 0.8, while A is flat.
     A is robust to noise, as hypothesised, but never catches up within the tested range.
   * Swarm size: at 120 agents, PF (0.95) and Boids (0.93) overtake B (0.92).
5. **Stop rule.** The docx's literal "total signal" stop rule deadlocks (0.25).
   Removing the stop entirely slightly *raises* coverage (0.85 vs 0.83) but adds
   collisions. Occlusion makes no measurable difference.
6. **Biology check: only partly supported.**
   * Under Rule A, more sway gives wider gaps (Spearman ρ = +0.22), but the effect
     is small (0.34 → 0.36 m).
   * Random walk shows it more strongly (ρ = +0.51), so it is a generic noise
     effect, not specific to abrasion.
   * Under B and C, gaps *narrow* with sway.
   * Report this honestly, not as confirmation.
7. **Scaling.** 1,000 agents run at 6 ms/step (A) to 44 ms/step (C) on one core.

Caveat: the sensing range R and the PF gain tuned to the top of their allowed
ranges (the same limits for every controller). Widening them could make the
baselines slightly stronger.

## Deviations from the docx (and why)

* **Rule B stop condition.** The docx says to stop when the *total* signal exceeds a
  threshold. Implemented literally, a swarm released as a dense cluster deadlocks,
  because every agent is crowded and so every agent stops. The default instead
  stops when the *weakest 90° quadrant* is shaded, which means the crown is boxed
  in on every side. The literal rule is kept as the ablation `B-total-stop`.
* **Walls and the light signal** (an open question in the docx). Walls block the
  neighbour signal (line of sight) and also reflect a weak signal of their own.
  The strength of that reflection, `wall_weight`, is tuned.
* **Lloyd** uses geodesic Voronoi cells with over-relaxation, because Euclidean
  cells cut through walls. It still gets stuck in local optima in multi-room plans,
  so the greedy coverage ceiling is reported as the real upper bound.
* The optional neural-network policy is deferred.
