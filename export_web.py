"""Record runs and write viewer/data.js for the web viewer (viewer/index.html).

    python export_web.py                                  # rubble + office, all controllers
    python export_web.py --plans apartment --agents 90 --fail 0.4
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from crownswarm import CONTROLLERS, EpisodeConfig, get_plan, run_episode
from crownswarm.floorplans import COVER_CELL
from crownswarm.metrics import coverage_ceiling
from demo import CTRL_COLOR, TITLE, load_params

ORDER = ["B", "C", "A", "PF", "Boids", "Lloyd", "RW"]


def bits(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(bool).ravel()).tobytes()).decode()


def export_run(out: dict) -> dict:
    fr = out["frames"]
    rem = out.get("removed")
    return {
        "t": [f["t"] for f in fr],
        "p": [np.round(f["pos"], 2).ravel().tolist() for f in fr],
        "h": [np.round(f["heading"], 2).tolist() for f in fr],
        "s": ["".join("2" if c else ("1" if v < 0.1 else "0") for c, v in zip(f["contact"], f["speed"]))
              for f in fr],
        "id": [f["ids"].tolist() for f in fr],
        "c": [bits(f["covered"]) for f in fr],
        "cov": [round(float(f["covered"].mean()), 4) for f in fr],
        "k": [int(f["collisions"]) for f in fr],
        "removed": None if rem is None else {"t": int(rem[0]), "p": np.round(rem[1], 2).ravel().tolist()},
        "n0": int(out["n_agents"]),
        "final": round(float(out["coverage"]), 4),
    }


# per-plan scenario: what the swarm is, how many, how windy, how far each one sees
SCENARIOS = {
    "rubble":    dict(label="Search & rescue in a collapsed building"),
    "office":    dict(label="Office floor after an earthquake"),
    "apartment": dict(label="Apartment block, room-by-room search"),
    "sky":       dict(label="Drone swarm over a city block (gusty wind)", agents=80, sway=0.06, cover_r=3.0),
    "forest":    dict(label="Ground robots under a forest canopy", sway=0.03),
    "orchard":   dict(label="Farm robots spreading through an orchard", agents=50),
    "warehouse": dict(label="Inventory robots in warehouse aisles"),
    "cave":      dict(label="Rescue robots in a cave system", agents=45),
}
DATA = Path("viewer/data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--controllers", nargs="+", default=ORDER, choices=list(CONTROLLERS))
    ap.add_argument("--agents", type=int, help="override every scenario's swarm size")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--fail", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=4)
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    for pname in a.plans:
        sc = SCENARIOS.get(pname, {})
        n = a.agents or sc.get("agents", 60)
        sway, cover_r = sc.get("sway", 0.02), sc.get("cover_r", 2.0)
        plan = get_plan(pname)
        cx = (plan.cover_pts[:, 0] / COVER_CELL).astype(int)
        cy = (plan.cover_pts[:, 1] / COVER_CELL).astype(int)
        meta = {"label": sc.get("label", pname), "agents": n, "sway": sway, "cover_r": cover_r,
                "steps": a.steps}
        entry = {**meta, "w": plan.width, "h": plan.height, "nx": plan.free.shape[1], "ny": plan.free.shape[0],
                 "cs": plan.cs, "walls": bits(~plan.free), "cell": COVER_CELL,
                 "cx": cx.tolist(), "cy": cy.tolist(), "entrance": list(plan.entrance),
                 "ceiling": round(coverage_ceiling(plan, n, cover_r), 4), "runs": {}}
        for c in a.controllers:
            out = run_episode(EpisodeConfig(plan=pname, controller=c, params=load_params(c), n_agents=n,
                                            steps=a.steps, sway=sway, cover_r=cover_r, fail_frac=a.fail,
                                            seed=a.seed, record_every=a.every))
            entry["runs"][c] = export_run(out)
            print(f"{pname:>10} {c:>6}: coverage {out['coverage']:.0%}", flush=True)
        (DATA / f"{pname}.js").write_text(
            f"(window.SIM_PLANS ||= {{}})[{json.dumps(pname)}] = " + json.dumps(entry, separators=(",", ":")) + ";\n")
        (DATA / f"{pname}.meta.json").write_text(json.dumps(meta))

    # index of every exported plan (so exporting one plan keeps the others)
    order = list(SCENARIOS)
    names = sorted((f.name[:-10] for f in DATA.glob("*.meta.json")),
                   key=lambda x: order.index(x) if x in order else 99)
    index = {"controllers": {c: {"title": TITLE[c][0], "sub": TITLE[c][1], "color": CTRL_COLOR[c]}
                             for c in ORDER if c in CONTROLLERS},
             "plans": {p: json.loads((DATA / f"{p}.meta.json").read_text()) for p in names}}
    Path("viewer/data.js").write_text("window.SIM = " + json.dumps(index) + ";\n")
    print(f"wrote viewer/data.js + {len(names)} plan files in {DATA}/")


if __name__ == "__main__":
    main()
