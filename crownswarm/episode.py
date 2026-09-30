"""Run one simulated deployment and return its metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import world
from .controllers import MAX_TURN, Sensors, make_controller
from .floorplans import get_plan, spawn
from .metrics import covered_mask, spacing

TOUCH = 0.05  # bumper reach beyond the body (m)
NEAR = 0.15   # hysteresis band for counting a new collision (m)


@dataclass
class EpisodeConfig:
    plan: str = "office"
    controller: str = "A"
    params: dict | None = None
    flags: dict = field(default_factory=dict)   # e.g. {"occlusion": False} for ablations
    n_agents: int = 60
    steps: int = 800
    sway: float = 0.02          # wind sway: position noise (m/step); heading noise is 2.5x (rad/step)
    sensor_noise: float = 0.05  # relative noise on signals/ranges, and rad on bearings
    fail_frac: float = 0.0      # fraction of agents removed mid-run
    fail_step: int | None = None
    seed: int = 0
    radius: float = 0.3
    vmax: float = 0.2
    cover_r: float = 2.0        # sensing footprint for the coverage metric (m)
    series_every: int = 20
    record_every: int = 0       # >0 stores positions for animation


def _edges(pos, heading, R, plan, noise, rng):
    """Directed neighbour edges within R, sorted by source, with line-of-sight flags."""
    n = len(pos)
    if R <= 0 or n < 2:
        z = np.zeros(0, dtype=np.int64)
        return np.zeros(n + 1, dtype=np.int64), z, z, np.zeros(0), np.zeros(0, dtype=bool)
    ptr, src, dst, dist, _ = _csr(pos, R)
    los = world.los_pairs(plan.free, plan.cs, pos, pos, src, dst)
    return ptr, src, dst, dist, los


def _csr(pos, R):
    """Symmetric neighbour lists within R, as CSR sorted by source."""
    n = len(pos)
    pr = cKDTree(pos).query_pairs(R, output_type="ndarray")
    src = np.concatenate([pr[:, 0], pr[:, 1]]).astype(np.int64)
    dst = np.concatenate([pr[:, 1], pr[:, 0]]).astype(np.int64)
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=ptr[1:])
    return ptr, src, dst, np.hypot(*(pos[dst] - pos[src]).T), None


def run_episode(cfg: EpisodeConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    plan = get_plan(cfg.plan)
    ctrl = make_controller(cfg.controller, cfg.params, **cfg.flags)
    r, noise = cfg.radius, cfg.sensor_noise
    fail_step = cfg.fail_step if cfg.fail_step is not None else cfg.steps // 2

    pos = spawn(plan, cfg.n_agents, r, rng)
    n0 = len(pos)
    heading = rng.uniform(-np.pi, np.pi, n0)
    st = ctrl.init_state(n0, rng)
    contact = np.zeros(n0, dtype=bool)
    cvec = np.zeros((n0, 2))
    prev_keys = np.zeros(0, dtype=np.int64)
    prev_wall = np.zeros(n0, dtype=bool)

    path_total = 0.0
    collisions = 0
    wall_hits = 0
    contact_steps = 0
    explored = np.zeros(len(plan.cover_pts), dtype=bool)
    series_t, series_cov = [], []
    frames = []
    ids = np.arange(n0)
    removed = None
    cov_before_fail = np.nan
    ray_range = max(ctrl.sense_range, 3.0)

    for t in range(cfg.steps):
        n = len(pos)
        # ---------------------------------------------------------------- sense
        rays = world.cast_rays(pos, heading, plan.free, plan.cs, ray_range)
        rays *= np.clip(1 + noise * rng.standard_normal(rays.shape), 0.5, 1.5)
        cdir = world.wrap(np.arctan2(cvec[:, 1], cvec[:, 0]) - heading + noise * rng.standard_normal(n)) \
            if n else np.zeros(0)
        ptr, src, dst, dist, los = _edges(pos, heading, ctrl.sense_range, plan, noise, rng)
        signal = None
        if ctrl.needs_signal:
            signal = world.sector_signal(pos, heading, ptr, dst, dist, los, r, ctrl.sense_range,
                                         getattr(ctrl, "occlusion", True))
            signal *= np.clip(1 + noise * rng.standard_normal(signal.shape), 0, None)
        vis = los
        e_src, e_dst = src[vis], dst[vis]
        rel = pos[e_dst] - pos[e_src]
        e_bear = world.wrap(np.arctan2(rel[:, 1], rel[:, 0]) - heading[e_src]
                            + noise * rng.standard_normal(len(e_src)))
        e_dist = dist[vis] * np.clip(1 + noise * rng.standard_normal(len(e_src)), 0.5, 1.5)
        e_relh = world.wrap(heading[e_dst] - heading[e_src])
        s = Sensors(n, contact, cdir, rays, e_src, e_dist, e_bear, e_relh, signal, cfg.vmax,
                    pos=pos if ctrl.privileged else None,
                    heading=heading if ctrl.privileged else None,
                    plan=plan if ctrl.privileged else None)

        # ------------------------------------------------------------------ act
        v, omega = ctrl.act(s, st, rng)
        v = np.clip(v, -1, 1) * cfg.vmax
        heading = world.wrap(heading + np.clip(omega, -MAX_TURN, MAX_TURN)
                             + rng.normal(0, 2.5 * cfg.sway, n))
        disp = v[:, None] * np.column_stack([np.cos(heading), np.sin(heading)])
        disp += rng.normal(0, cfg.sway, (n, 2))

        # -------------------------------------------------------------- physics
        old = pos.copy()
        reach = 2 * r + 2 * float(np.abs(disp).max()) * 1.5 + 1e-6 if n else 0.0
        cptr, _, cnbr, _, _ = _csr(pos, reach) if n > 1 else (np.zeros(n + 1, np.int64), 0, np.zeros(0, np.int64), 0, 0)
        world.move_agents(pos, disp, plan.clear, plan.cs, r, cptr, cnbr, rng.permutation(n))
        if n > 1:
            ov = cKDTree(pos).query_pairs(2 * r + 0.1, output_type="ndarray").astype(np.int64)
            if len(ov):
                world.resolve_overlaps(pos, ov, r, plan.clear, plan.cs, 4)
        path_total += float(np.hypot(*(pos - old).T).sum())

        # bumper contacts for the next step. A collision is counted only when a pair
        # touches after having been further apart than NEAR (sway jitter between
        # resting neighbours is not a new collision).
        cvec = np.zeros((n, 2))
        tree = cKDTree(pos)
        near = tree.query_pairs(2 * r + NEAR, output_type="ndarray").astype(np.int64)
        if len(near):
            d = pos[near[:, 1]] - pos[near[:, 0]]
            dn = np.hypot(*d.T)
            touch = dn < 2 * r + TOUCH
            pr, d = near[touch], d[touch] / np.maximum(dn[touch], 1e-9)[:, None]
            np.add.at(cvec, pr[:, 0], d)
            np.add.at(cvec, pr[:, 1], -d)
            collisions += int((~np.isin(pr[:, 0] * n + pr[:, 1], prev_keys)).sum())
            prev_keys = near[:, 0] * n + near[:, 1]
        else:
            prev_keys = np.zeros(0, dtype=np.int64)
        iy = np.clip((pos[:, 1] / plan.cs).astype(int), 0, plan.free.shape[0] - 1)
        ix = np.clip((pos[:, 0] / plan.cs).astype(int), 0, plan.free.shape[1] - 1)
        clr = plan.clear[iy, ix]
        at_wall = clr < r + TOUCH
        cvec[at_wall] -= plan.normal[:, iy[at_wall], ix[at_wall]].T
        wall_hits += int((at_wall & ~prev_wall).sum())
        prev_wall = clr < r + NEAR
        contact = (np.hypot(*cvec.T) > 1e-9) | at_wall
        contact_steps += int(contact.sum())

        # ------------------------------------------------------------- logging
        if (t + 1) % cfg.series_every == 0 or t + 1 == cfg.steps or t + 1 == fail_step:
            cm = covered_mask(pos, plan, cfg.cover_r)
            explored |= cm
            series_t.append(t + 1)
            series_cov.append(cm.mean())
        if cfg.record_every and t % cfg.record_every == 0:
            frames.append({"t": t + 1, "pos": pos.copy(), "heading": heading.copy(), "ids": ids.copy(),
                           "speed": np.abs(v) / cfg.vmax, "contact": contact.copy(),
                           "covered": covered_mask(pos, plan, cfg.cover_r), "collisions": collisions})

        # ------------------------------------------------------------- failure
        if cfg.fail_frac > 0 and t + 1 == fail_step:
            cov_before_fail = series_cov[-1]
            keep = rng.random(n) >= cfg.fail_frac
            removed = (t + 1, pos[~keep].copy())
            pos, heading, contact, cvec = pos[keep], heading[keep], contact[keep], cvec[keep]
            ids = ids[keep]
            prev_wall = prev_wall[keep]
            st = {k: a[keep] for k, a in st.items()}
            prev_keys = np.zeros(0, dtype=np.int64)

    series_cov = np.array(series_cov)
    series_t = np.array(series_t)
    pre = series_t <= (fail_step if cfg.fail_frac > 0 else cfg.steps)
    peak = series_cov[pre].max()
    t90 = float(series_t[pre][np.argmax(series_cov[pre] >= 0.9 * peak)])
    out = {
        **{k: v for k, v in asdict(cfg).items() if k not in ("params", "flags")},
        "coverage": float(series_cov[-1]),
        "coverage_mean_2nd_half": float(series_cov[series_t > cfg.steps // 2].mean()),
        "explored": float(explored.mean()),
        "t90": t90,
        "collisions_per_agent": collisions / n0,
        "wall_hits_per_agent": wall_hits / n0,
        "path_per_agent": path_total / n0,
        "contact_frac": contact_steps / (n0 * cfg.steps),
        "post_over_pre_fail": float(series_cov[-1] / cov_before_fail) if cfg.fail_frac > 0 else np.nan,
        "n_final": len(pos),
        **spacing(pos, r),
    }
    if cfg.record_every:
        out["frames"] = frames
        out["series"] = (series_t, series_cov)
        out["removed"] = removed
    return out
