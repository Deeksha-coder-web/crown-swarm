"""Swarm controllers, ordered by how much they are allowed to sense.

Body frame: x points forward, y to the left, angles in radians. Controllers
return ``v`` as a fraction of max speed in [-1, 1] and ``omega`` in rad/step.
Only ``Lloyd`` reads privileged world state (global positions); it is an
upper-bound reference, not a like-for-like competitor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .world import N_SECTORS

MAX_TURN = 1.0  # rad/step
SECTOR_ANG = np.arange(N_SECTORS) * 2 * np.pi / N_SECTORS
SECTOR_VEC = np.column_stack([np.cos(SECTOR_ANG), np.sin(SECTOR_ANG)])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Sensors:
    """Everything a controller may read. Arrays are per agent unless noted."""
    n: int
    contact: np.ndarray       # bool: bumper touched an agent or a wall this step
    contact_dir: np.ndarray   # body-frame bearing towards the touched object(s)
    rays: np.ndarray          # (n, 8) distance to wall per body-frame sector, capped at range
    # neighbour edges (visible neighbours within the controller's sensing range)
    e_src: np.ndarray
    e_dist: np.ndarray
    e_bear: np.ndarray        # body-frame bearing to neighbour
    e_relh: np.ndarray        # neighbour heading relative to own heading
    signal: np.ndarray | None  # (n, 8) far-red-like neighbour signal (Rule B/C only)
    vmax: float
    # privileged (Lloyd only)
    pos: np.ndarray | None = None
    heading: np.ndarray | None = None
    plan: object = None


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    default: float
    log: bool = False


class Controller:
    name = "base"
    space: tuple[Param, ...] = ()
    privileged = False
    needs_signal = False
    occlusion = True

    def __init__(self, params: dict | None = None, **flags):
        self.p = {q.name: q.default for q in self.space}
        if params:
            unknown = set(params) - set(self.p)
            if unknown:
                raise KeyError(f"{self.name}: unknown params {unknown}")
            self.p.update(params)
        for k, v in flags.items():
            setattr(self, k, v)

    @property
    def sense_range(self) -> float:
        return float(self.p.get("R", 0.0))

    def init_state(self, n, rng) -> dict:
        return {}

    def act(self, s: Sensors, st: dict, rng) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    # ---- CMA-ES helpers: map parameters to/from the unit cube
    @classmethod
    def decode(cls, x) -> dict:
        out = {}
        for q, u in zip(cls.space, np.clip(x, 0, 1)):
            out[q.name] = (math.exp(math.log(q.lo) + u * (math.log(q.hi) - math.log(q.lo)))
                           if q.log else q.lo + u * (q.hi - q.lo))
        return out

    @classmethod
    def encode(cls, params: dict) -> np.ndarray:
        x = []
        for q in cls.space:
            v = params[q.name]
            x.append((math.log(v) - math.log(q.lo)) / (math.log(q.hi) - math.log(q.lo))
                     if q.log else (v - q.lo) / (q.hi - q.lo))
        return np.clip(np.array(x), 0, 1)


def _steer(phi, gain=1.0):
    return np.clip(gain * phi, -MAX_TURN, MAX_TURN)


def _wall_signal(rays, R, weight):
    """Walls treated as weak reflectors: same fall-off as neighbours, scaled by weight."""
    return weight * np.clip(1.0 - rays / R, 0.0, None) ** 2


# --------------------------------------------------------------------- abrasion layer

def _abrasion_init(n):
    return {"vigor": np.ones(n), "quiet": np.zeros(n), "retreat": np.zeros(n, dtype=int),
            "away": np.zeros(n)}


def _abrasion_update(s, st, damage, delay, rate, retreat_steps):
    """Tip damage on contact, regrowth after a quiet period. Returns mask of retreating agents."""
    c = s.contact
    st["away"][c] = wrap(s.contact_dir[c] + np.pi)
    st["retreat"][c] = int(round(retreat_steps))
    st["vigor"][c] *= 1.0 - damage
    st["quiet"][c] = 0
    st["quiet"][~c] += 1
    g = ~c & (st["quiet"] > delay)
    st["vigor"][g] += rate * (1.0 - st["vigor"][g])
    return st["retreat"] > 0


def _abrasion_retreat(st, r, turn_gain, v, omega):
    """Back away from the contact while turning to face away from it."""
    omega[r] = _steer(st["away"][r], turn_gain)
    v[r] = 0.5 * np.cos(st["away"][r])
    st["away"][r] = wrap(st["away"][r] - omega[r])
    st["retreat"][r] -= 1


# ------------------------------------------------------------------------ controllers

class RandomWalk(Controller):
    """Rung 0: bump sensor only, no memory. Persistent walk, random turn away on bump."""
    name = "RW"
    space = (Param("wander", 0.0, 0.6, 0.15), Param("spread", 0.0, math.pi, 1.2))

    def init_state(self, n, rng):
        return {"turn": np.zeros(n)}

    def act(self, s, st, rng):
        c = s.contact
        st["turn"][c] = wrap(s.contact_dir[c] + np.pi + rng.uniform(-1, 1, c.sum()) * self.p["spread"])
        turning = np.abs(st["turn"]) > 0.05
        omega = rng.normal(0, self.p["wander"], s.n)
        omega[turning] = _steer(st["turn"][turning])
        st["turn"][turning] -= omega[turning]
        st["turn"][~turning] = 0
        v = np.where(turning, 0.0, 1.0)
        return v, omega


class RuleA(Controller):
    """Rung 1 — abrasion. Senses only bumps. Contact damages 'vigour' (speed) and
    triggers a short retreat; after a quiet period vigour regrows."""
    name = "A"
    space = (Param("damage", 0.02, 0.9, 0.3),
             Param("regrow_delay", 0.0, 150.0, 30.0),
             Param("regrow_rate", 1e-3, 0.3, 0.02, log=True),
             Param("retreat_steps", 1.0, 30.0, 8.0),
             Param("turn_gain", 0.05, 1.0, 0.5),
             Param("wander", 0.0, 0.5, 0.1))

    def init_state(self, n, rng):
        return _abrasion_init(n)

    def act(self, s, st, rng):
        p = self.p
        r = _abrasion_update(s, st, p["damage"], p["regrow_delay"], p["regrow_rate"], p["retreat_steps"])
        omega = rng.normal(0, p["wander"], s.n)
        v = st["vigor"].copy()
        _abrasion_retreat(st, r, p["turn_gain"], v, omega)
        return v, omega


class RuleB(Controller):
    """Rung 2 — light sensing. 8-sector far-red-like signal (occluded, wall-blocked).
    Steer away from the signal-weighted direction (towards the weakest side),
    slow down as shading rises, stop above a threshold (with hysteresis).

    ``stop_on="quadrant"`` (default): shading = signal in the *weakest* 90-degree
    window, i.e. stop only when shaded on every side, like a crown boxed in by
    neighbours. ``stop_on="total"`` is the literal docx rule (sum of all sectors);
    it deadlocks a dense start because every agent is crowded, so it is kept
    as an ablation."""
    name = "B"
    needs_signal = True
    stop_on = "quadrant"
    space = (Param("R", 1.0, 6.0, 3.0),
             Param("stop_threshold", 0.05, 4.0, 1.5, log=True),
             Param("resume_frac", 0.2, 1.0, 0.7),
             Param("turn_gain", 0.05, 1.0, 0.5),
             Param("wall_weight", 0.0, 3.0, 1.0),
             Param("wander", 0.0, 0.5, 0.1))

    def init_state(self, n, rng):
        return {"stopped": np.zeros(n, dtype=bool)}

    def _light(self, s, st, rng):
        p = self.p
        S = s.signal + _wall_signal(s.rays, p["R"], p["wall_weight"])
        T = S.sum(1) if self.stop_on == "total" else (S + np.roll(S, -1, axis=1)).min(1)
        u = -S @ SECTOR_VEC
        thr = p["stop_threshold"]
        st["stopped"] = np.where(T > thr, True, np.where(T < thr * p["resume_frac"], False, st["stopped"]))
        phi = np.arctan2(u[:, 1], u[:, 0])
        has = np.hypot(u[:, 0], u[:, 1]) > 1e-9
        omega = rng.normal(0, p["wander"], s.n)
        omega[has] += _steer(phi[has], p["turn_gain"])
        v = np.where(has, np.clip(1 - T / thr, 0, 1) * np.maximum(np.cos(phi), 0), 1.0)
        v[st["stopped"]] = 0.0
        omega[st["stopped"]] = 0.0
        return v, np.clip(omega, -MAX_TURN, MAX_TURN)

    def act(self, s, st, rng):
        return self._light(s, st, rng)


class RuleC(RuleB):
    """Rung 2+ — hybrid: Rule B steering, plus Rule A contact damage/retreat."""
    name = "C"
    space = RuleB.space + (Param("damage", 0.02, 0.9, 0.3),
                           Param("regrow_rate", 1e-3, 0.3, 0.02, log=True),
                           Param("retreat_steps", 1.0, 30.0, 8.0))

    def init_state(self, n, rng):
        return {**super().init_state(n, rng), **_abrasion_init(n)}

    def act(self, s, st, rng):
        p = self.p
        v, omega = self._light(s, st, rng)
        r = _abrasion_update(s, st, p["damage"], 20.0, p["regrow_rate"], p["retreat_steps"])
        v = v * st["vigor"]
        _abrasion_retreat(st, r, p["turn_gain"], v, omega)
        return v, omega


class PotentialField(Controller):
    """Rung 3 — exact range & bearing to visible neighbours and walls; 1/d repulsion.
    (After Howard, Mataric & Sukhatme 2002.) Stops when net force is small."""
    name = "PF"
    space = (Param("R", 1.0, 6.0, 3.0),
             Param("gain", 0.01, 2.0, 0.2, log=True),
             Param("stop_eps", 1e-3, 0.5, 0.02, log=True),
             Param("wall_weight", 0.0, 3.0, 1.0))

    def act(self, s, st, rng):
        p = self.p
        R = p["R"]
        mag = p["gain"] * np.clip(1 / np.maximum(s.e_dist, 1e-3) - 1 / R, 0, None)
        fx = -np.bincount(s.e_src, mag * np.cos(s.e_bear), s.n)
        fy = -np.bincount(s.e_src, mag * np.sin(s.e_bear), s.n)
        wm = p["gain"] * p["wall_weight"] * np.clip(1 / np.maximum(s.rays, 1e-3) - 1 / R, 0, None)
        fx -= wm @ SECTOR_VEC[:, 0]
        fy -= wm @ SECTOR_VEC[:, 1]
        f = np.hypot(fx, fy)
        phi = np.arctan2(fy, fx)
        moving = f > p["stop_eps"]
        v = np.where(moving, np.clip(f, 0, 1) * np.maximum(np.cos(phi), 0), 0.0)
        omega = np.where(moving, _steer(phi), 0.0)
        return v, omega


class BoidsDispersion(Controller):
    """Rung 3 — Reynolds boids without cohesion: separation + alignment + cruise.
    Needs neighbours' relative headings (sensed or communicated). Never stops."""
    name = "Boids"
    space = (Param("R", 1.0, 6.0, 3.0),
             Param("w_sep", 0.05, 5.0, 1.0, log=True),
             Param("w_align", 0.0, 2.0, 0.3),
             Param("wander", 0.0, 0.5, 0.1))

    def act(self, s, st, rng):
        p = self.p
        R = p["R"]
        mag = np.clip(1 / np.maximum(s.e_dist, 1e-3) - 1 / R, 0, None)
        sx = -np.bincount(s.e_src, mag * np.cos(s.e_bear), s.n)
        sy = -np.bincount(s.e_src, mag * np.sin(s.e_bear), s.n)
        wm = np.clip(1 / np.maximum(s.rays, 1e-3) - 1 / R, 0, None)
        sx -= wm @ SECTOR_VEC[:, 0]
        sy -= wm @ SECTOR_VEC[:, 1]
        cnt = np.maximum(np.bincount(s.e_src, minlength=s.n), 1)
        ax = np.bincount(s.e_src, np.cos(s.e_relh), s.n) / cnt
        ay = np.bincount(s.e_src, np.sin(s.e_relh), s.n) / cnt
        dx = 1.0 + p["w_sep"] * sx + p["w_align"] * ax
        dy = p["w_sep"] * sy + p["w_align"] * ay
        phi = np.arctan2(dy, dx)
        omega = np.clip(_steer(phi) + rng.normal(0, p["wander"], s.n), -MAX_TURN, MAX_TURN)
        return np.maximum(np.cos(phi), 0), omega


class Lloyd(Controller):
    """Rung 4 — PRIVILEGED. Knows all positions and the map; moves each agent towards
    the centroid of its geodesic Voronoi cell (Cortes et al. 2004, with geodesic
    distance so cells do not cut through walls)."""
    name = "Lloyd"
    privileged = True
    relax = 1.8   # over-relaxation of the centroid step; 1 = plain Lloyd

    def act(self, s, st, rng):
        from .floorplans import LLOYD_CELL
        from .world import geodesic_lloyd
        g = s.plan.lloyd_grid
        ay = np.clip((s.pos[:, 1] / LLOYD_CELL).astype(np.int64), 0, g.shape[0] - 1)
        ax = np.clip((s.pos[:, 0] / LLOYD_CELL).astype(np.int64), 0, g.shape[1] - 1)
        wy, wx, cnt = geodesic_lloyd(g, ay, ax, self.relax)
        dx = (wx + 0.5) * LLOYD_CELL - s.pos[:, 0]
        dy = (wy + 0.5) * LLOYD_CELL - s.pos[:, 1]
        dist = np.hypot(dx, dy)
        phi = wrap(np.arctan2(dy, dx) - s.heading)
        v = np.clip(dist / s.vmax, 0, 1) * np.maximum(np.cos(phi), 0)
        v[cnt == 0] = 0
        return v, _steer(phi)


CONTROLLERS = {c.name: c for c in (RandomWalk, RuleA, RuleB, RuleC, PotentialField, BoidsDispersion, Lloyd)}
TUNABLE = [k for k, c in CONTROLLERS.items() if c.space]


def make_controller(name: str, params: dict | None = None, **flags) -> Controller:
    return CONTROLLERS[name](params, **flags)
