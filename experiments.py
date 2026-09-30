"""Experiments from the project plan (docx section 6.6). Results go to results/*.csv.

    python experiments.py compare             # all controllers, unseen test plans, 30 seeds
    python experiments.py sweep --var sway    # also: noise, n, fail
    python experiments.py ablate
    python experiments.py biology             # does more sway give wider gaps?
    python experiments.py scaling             # 10 -> 1000 agents
    python experiments.py all
Add --seeds 5 for a quick run.
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from crownswarm import CONTROLLERS, TEST_PLANS, TRAIN_PLANS, EpisodeConfig, run_episode
from demo import load_params

OUT = Path("results")
MAIN = ["RW", "A", "B", "C", "PF", "Boids", "Lloyd"]

# name -> (controller, param overrides, flags)
VARIANTS = {name: (name, {}, {}) for name in CONTROLLERS}
VARIANTS.update({
    "A-no-regrowth": ("A", {"regrow_rate": 0.0}, {}),
    "A-no-damage": ("A", {"damage": 0.0}, {}),
    "B-total-stop": ("B", {}, {"stop_on": "total"}),
    "B-no-occlusion": ("B", {}, {"occlusion": False}),
    "B-no-stop": ("B", {"stop_threshold": 1e9}, {}),
})


def _run(job):
    variant, kw = job
    ctrl, over, flags = VARIANTS[variant]
    params = {**(load_params(ctrl) or {}), **over}
    t = time.perf_counter()
    m = run_episode(EpisodeConfig(controller=ctrl, params=params or None, flags=flags, **kw))
    m["variant"] = variant
    m["sec_per_step"] = (time.perf_counter() - t) / kw.get("steps", 800)
    return m


def run(jobs, name, workers):
    t0 = time.time()
    with Pool(workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_run, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % max(1, len(jobs) // 10) == 0:
                print(f"  {name}: {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"wrote {OUT / name}.csv ({len(df)} rows, {time.time() - t0:.0f}s)")
    return df


BASE = dict(n_agents=60, steps=800, sway=0.02, sensor_noise=0.05, fail_frac=0.2)


def compare(seeds, workers):
    """Main table: every controller on train and unseen test plans, each seed run
    twice - intact, and with 20% of agents removed mid-run - so failure recovery
    is measured against the same run without the failure."""
    jobs = [(v, dict(BASE, plan=p, seed=s, fail_frac=f)) for v in MAIN for p in TRAIN_PLANS + TEST_PLANS
            for s in range(seeds) for f in (0.0, 0.2)]
    return run(jobs, "compare", workers)


SWEEPS = {
    "sway": [0.0, 0.02, 0.04, 0.07, 0.1],
    "noise": [0.0, 0.1, 0.2, 0.4, 0.8],
    "n": [10, 30, 60, 90, 120],
    "fail": [0.0, 0.2, 0.4, 0.6],
}
FIELD = {"sway": "sway", "noise": "sensor_noise", "n": "n_agents", "fail": "fail_frac"}


def sweep(var, seeds, workers):
    jobs = [(v, dict(BASE, plan=p, seed=s, **{FIELD[var]: x}))
            for v in MAIN for p in TEST_PLANS for x in SWEEPS[var] for s in range(seeds)]
    return run(jobs, f"sweep_{var}", workers)


def ablate(seeds, workers):
    vs = ["A", "A-no-regrowth", "A-no-damage", "B", "B-total-stop", "B-no-occlusion", "B-no-stop", "RW"]
    jobs = [(v, dict(BASE, plan=p, seed=s)) for v in vs for p in TEST_PLANS for s in range(seeds)]
    return run(jobs, "ablate", workers)


def biology(seeds, workers):
    """Dense 'stand' in an open arena, no failures: gap width vs sway strength."""
    sways = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
    jobs = [(v, dict(plan="open", n_agents=150, steps=1500, sway=w, sensor_noise=0.05, seed=s))
            for v in ["A", "B", "C", "RW"] for w in sways for s in range(seeds)]
    return run(jobs, "biology", workers)


def scaling(seeds, workers):
    """Constant density (6.4 m^2 per agent), open arena, released from the centre."""
    jobs = []
    for n in [10, 30, 100, 300, 1000]:
        side = np.sqrt(n * 6.4 / 1.5)
        plan = f"open:{np.ceil(1.5 * side):g}x{np.ceil(side):g}"
        for v in ["A", "B", "C", "PF", "Boids", "RW"]:
            for s in range(seeds):
                jobs.append((v, dict(plan=plan, n_agents=n, steps=1200, sway=0.02,
                                     sensor_noise=0.05, seed=s)))
    return run(jobs, "scaling", workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["compare", "sweep", "ablate", "biology", "scaling", "all"])
    ap.add_argument("--var", choices=list(SWEEPS), help="for sweep; default: all sweeps")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if a.what in ("compare", "all"):
        compare(a.seeds, a.workers)
    if a.what in ("sweep", "all"):
        for var in [a.var] if a.var else list(SWEEPS):
            sweep(var, a.seeds, a.workers)
    if a.what in ("ablate", "all"):
        ablate(a.seeds, a.workers)
    if a.what in ("biology", "all"):
        biology(a.seeds, a.workers)
    if a.what in ("scaling", "all"):
        scaling(max(3, a.seeds // 3), a.workers)


if __name__ == "__main__":
    main()
