"""CMA-ES tuning of controller parameters on randomised TRAINING scenarios.

Every controller gets the same budget, so baselines are as strong as the
crown-shyness rules. Within a generation all candidates are scored on the
same scenarios (common random numbers) to reduce noise.

    python tune.py --controller A            # one controller
    python tune.py --all                     # every tunable controller
    python tune.py --all --gens 8 --pop 8 --episodes 4   # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cma
import numpy as np

from crownswarm import CONTROLLERS, TRAIN_PLANS, TUNABLE, EpisodeConfig, run_episode

OUT = Path("results/params")
STEPS = 600
VMAX = 0.2

# fitness weights (documented in README): reward coverage & exploration,
# penalise uneven spacing, collisions and distance travelled.
W = {"coverage": 1.0, "explored": 0.25, "evenness": 0.1, "collisions": 0.01, "path": 0.1}


def fitness(m: dict) -> float:
    cv = m["evenness_cv"] if np.isfinite(m["evenness_cv"]) else 2.0
    return (W["coverage"] * m["coverage_mean_2nd_half"]
            + W["explored"] * m["explored"]
            - W["evenness"] * min(cv, 2.0)
            - W["collisions"] * m["collisions_per_agent"]
            - W["path"] * m["path_per_agent"] / (VMAX * m["steps"]))


def sample_scenarios(rng: np.random.Generator, k: int) -> list[dict]:
    return [dict(plan=str(rng.choice(TRAIN_PLANS)),
                 n_agents=int(rng.integers(30, 91)),
                 sway=float(rng.uniform(0.0, 0.05)),
                 sensor_noise=float(rng.uniform(0.0, 0.15)),
                 fail_frac=0.2 if rng.random() < 0.5 else 0.0,
                 seed=int(rng.integers(1 << 30)),
                 steps=STEPS)
            for _ in range(k)]


def _eval(job):
    ctrl, params, scen = job
    return fitness(run_episode(EpisodeConfig(controller=ctrl, params=params, **scen)))


def evaluate(pool, ctrl, param_list, scenarios) -> np.ndarray:
    jobs = [(ctrl, p, s) for p in param_list for s in scenarios]
    f = np.array(pool.map(_eval, jobs, chunksize=1))
    return f.reshape(len(param_list), len(scenarios)).mean(1)


def tune(ctrl: str, gens: int, pop: int, episodes: int, seed: int, pool) -> dict:
    cls = CONTROLLERS[ctrl]
    rng = np.random.default_rng(seed)
    x0 = cls.encode({q.name: q.default for q in cls.space})
    es = cma.CMAEvolutionStrategy(x0, 0.25, {"bounds": [0, 1], "popsize": pop, "seed": seed + 1,
                                              "verbose": -9})
    history = []
    t0 = time.time()
    for g in range(gens):
        scen = sample_scenarios(rng, episodes)
        xs = es.ask()
        f = evaluate(pool, ctrl, [cls.decode(x) for x in xs], scen)
        es.tell(xs, list(-f))
        history.append({"gen": g, "best": float(f.max()), "mean": float(f.mean())})
        print(f"  [{ctrl}] gen {g + 1:>3}/{gens}  best {f.max():.3f}  mean {f.mean():.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # Final check on a fixed validation set (train plans, fresh seeds): the CMA
    # mean (robust under noise) vs. the hand-set defaults. Keep the better one.
    val = sample_scenarios(np.random.default_rng(10_000 + seed), max(2 * episodes, 12))
    tuned = cls.decode(es.result.xfavorite)
    default = {q.name: q.default for q in cls.space}
    f_tuned, f_default = evaluate(pool, ctrl, [tuned, default], val)
    best = tuned if f_tuned >= f_default else default
    print(f"  [{ctrl}] validation: tuned {f_tuned:.3f}  default {f_default:.3f}", flush=True)
    return {"controller": ctrl, "params": best, "tuned_params": tuned,
            "val_fitness_tuned": float(f_tuned), "val_fitness_default": float(f_default),
            "used": "tuned" if best is tuned else "default",
            "budget": {"gens": gens, "pop": pop, "episodes": episodes, "steps": STEPS, "seed": seed},
            "weights": W, "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=TUNABLE)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=8, help="scenarios per candidate per generation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    names = TUNABLE if args.all else [args.controller or "A"]
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        for name in names:
            print(f"Tuning {name} ({len(CONTROLLERS[name].space)} params)", flush=True)
            res = tune(name, args.gens, args.pop, args.episodes, args.seed, pool)
            (OUT / f"{name}.json").write_text(json.dumps(res, indent=2))
            print(f"  -> {OUT / (name + '.json')}: {res['params']}", flush=True)


if __name__ == "__main__":
    main()
