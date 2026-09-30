"""Floor plans as occupancy grids.

World units are metres. ``free[iy, ix]`` is True where there is no wall; cell
(ix, iy) covers x in [ix*cs, (ix+1)*cs), y in [iy*cs, (iy+1)*cs).

Plans are split into TRAIN_PLANS (seen by the optimiser) and TEST_PLANS
(only used for evaluation) so generalisation can be measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from numba import njit
from scipy import ndimage
from scipy.spatial import cKDTree

CELL = 0.1          # fine grid resolution (m)
COVER_CELL = 0.5    # resolution of the coverage target grid (m)
LLOYD_CELL = 0.3    # Lloyd baseline grid; < 2r/sqrt(2) so no two agents share a cell
WALL = 0.2          # wall thickness (m)
AGENT_RADIUS = 0.3  # used to decide which space agents can physically reach


@dataclass
class FloorPlan:
    name: str
    free: np.ndarray            # (ny, nx) bool
    entrance: tuple[float, float]
    cs: float = CELL
    # derived fields, filled by finalize()
    clear: np.ndarray = field(default=None, repr=False)      # distance to nearest wall (m)
    normal: np.ndarray = field(default=None, repr=False)     # (2, ny, nx) unit vector pointing away from walls
    target: np.ndarray = field(default=None, repr=False)     # free cells connected to the entrance
    reach: np.ndarray = field(default=None, repr=False)      # cells an agent centre can reach
    geo: np.ndarray = field(default=None, repr=False)        # geodesic distance (cells) from entrance
    cover_pts: np.ndarray = field(default=None, repr=False)  # (M, 2) coverage sample points
    cover_tree: cKDTree = field(default=None, repr=False)
    lloyd_grid: np.ndarray = field(default=None, repr=False)  # coarse reach grid (LLOYD_CELL)

    @property
    def width(self) -> float:
        return self.free.shape[1] * self.cs

    @property
    def height(self) -> float:
        return self.free.shape[0] * self.cs

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int(y / self.cs), int(x / self.cs)

    def finalize(self) -> "FloorPlan":
        free = self.free
        # Clearance measured from cell centres to the nearest wall cell centre,
        # minus half a cell so a cell touching a wall has clearance ~0.
        self.clear = np.maximum(ndimage.distance_transform_edt(free) * self.cs - 0.5 * self.cs, 0.0)
        gy, gx = np.gradient(ndimage.gaussian_filter(self.clear, 1.0))
        norm = np.hypot(gx, gy) + 1e-12
        self.normal = np.stack([gx / norm, gy / norm]).astype(np.float64)

        ey, ex = self.cell(*self.entrance)
        if not free[ey, ex]:
            raise ValueError(f"{self.name}: entrance is inside a wall")
        lab, _ = ndimage.label(free)
        self.target = lab == lab[ey, ex]

        lab, _ = ndimage.label(self.clear >= AGENT_RADIUS)
        if lab[ey, ex] == 0:
            raise ValueError(f"{self.name}: entrance has too little clearance")
        self.reach = lab == lab[ey, ex]
        self.geo = _bfs(self.reach, ey, ex)

        step = int(round(COVER_CELL / self.cs))
        ys, xs = np.nonzero(self.target[step // 2::step, step // 2::step])
        self.cover_pts = np.column_stack([(xs * step + step // 2 + 0.5) * self.cs,
                                          (ys * step + step // 2 + 0.5) * self.cs])
        self.cover_tree = cKDTree(self.cover_pts)
        ls = int(round(LLOYD_CELL / self.cs))
        self.lloyd_grid = self.reach[ls // 2::ls, ls // 2::ls].copy()
        return self


@njit(cache=True)
def _bfs(mask, sy, sx):
    ny, nx = mask.shape
    dist = np.full((ny, nx), -1, dtype=np.int64)
    qy = np.empty(ny * nx, dtype=np.int64)
    qx = np.empty(ny * nx, dtype=np.int64)
    head, tail = 0, 1
    qy[0], qx[0] = sy, sx
    dist[sy, sx] = 0
    while head < tail:
        y, x = qy[head], qx[head]
        head += 1
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx and mask[yy, xx] and dist[yy, xx] < 0:
                dist[yy, xx] = dist[y, x] + 1
                qy[tail], qx[tail] = yy, xx
                tail += 1
    return dist


class _Builder:
    """Draw a plan with rectangles given in metres."""

    def __init__(self, w: float, h: float, solid: bool = False):
        self.free = np.full((int(round(h / CELL)), int(round(w / CELL))), not solid)

    def _sl(self, x0, y0, x1, y1):
        return (slice(int(round(y0 / CELL)), int(round(y1 / CELL))),
                slice(int(round(x0 / CELL)), int(round(x1 / CELL))))

    def wall(self, x0, y0, x1, y1):
        self.free[self._sl(x0, y0, x1, y1)] = False
        return self

    def carve(self, x0, y0, x1, y1):
        self.free[self._sl(x0, y0, x1, y1)] = True
        return self

    def hwall(self, y, x0, x1, doors=(), door_w=1.2):
        self.wall(x0, y - WALL / 2, x1, y + WALL / 2)
        for d in doors:
            self.carve(d - door_w / 2, y - WALL / 2, d + door_w / 2, y + WALL / 2)
        return self

    def vwall(self, x, y0, y1, doors=(), door_w=1.2):
        self.wall(x - WALL / 2, y0, x + WALL / 2, y1)
        for d in doors:
            self.carve(x - WALL / 2, d - door_w / 2, x + WALL / 2, d + door_w / 2)
        return self

    def disk(self, x, y, r, free=False):
        ny, nx = self.free.shape
        yy, xx = np.ogrid[:ny, :nx]
        self.free[((xx + 0.5) * CELL - x) ** 2 + ((yy + 0.5) * CELL - y) ** 2 <= r * r] = free
        return self

    def border(self):
        h, w = self.free.shape[0] * CELL, self.free.shape[1] * CELL
        self.wall(0, 0, w, WALL).wall(0, h - WALL, w, h).wall(0, 0, WALL, h).wall(w - WALL, 0, w, h)
        return self


# --------------------------------------------------------------------------- plans

def _office() -> FloorPlan:
    """Central corridor with four rooms on each side (train)."""
    b = _Builder(24, 16).border()
    b.hwall(7, 0, 24, doors=(3, 9, 15, 21)).hwall(9, 0, 24, doors=(3, 9, 15, 21))
    for x in (6, 12, 18):
        b.vwall(x, 0, 6.9).vwall(x, 9.1, 16)
    return FloorPlan("office", b.free, (1.0, 8.0))


def _hall() -> FloorPlan:
    """Open hall with a grid of 1 m pillars (train)."""
    b = _Builder(24, 16).border()
    for x in (4, 8, 12, 16, 20):
        for y in (4, 8, 12):
            b.wall(x - 0.5, y - 0.5, x + 0.5, y + 0.5)
    return FloorPlan("hall", b.free, (1.5, 1.5))


def _mine() -> FloorPlan:
    """Branching tunnels cut into rock (train)."""
    b = _Builder(24, 16, solid=True)
    b.carve(0.2, 7, 23.8, 9)            # main drift
    b.carve(4, 1, 5.5, 15)              # cross-cut 1
    b.carve(11, 2, 12.5, 7)             # cross-cut 2 (down)
    b.carve(18, 9, 19.5, 15)            # cross-cut 3 (up)
    b.carve(13, 1, 22, 3).carve(20, 3, 21.5, 7)   # lower gallery
    b.carve(1, 12, 8, 14.5)             # upper-left gallery
    b.border()
    return FloorPlan("mine", b.free, (0.8, 8.0))


def _apartment() -> FloorPlan:
    """L-shaped corridor, rooms of mixed size (test, never seen in training)."""
    b = _Builder(24, 16).border()
    b.hwall(3.1, 0, 16.4, doors=(3, 8.5, 13.5))
    b.vwall(16.4, 3.1, 16, doors=(6, 12))
    b.vwall(18.1, 3.1, 16, doors=(5, 12.5))
    b.hwall(10, 0, 16.4, doors=(3, 8.5, 14))
    b.vwall(6, 3.1, 10).vwall(11, 10, 16)
    b.hwall(9.5, 18.1, 24)
    return FloorPlan("apartment", b.free, (1.0, 1.5))


def _rubble() -> FloorPlan:
    """Collapsed building: broken walls and debris (test, never seen in training)."""
    b = _Builder(24, 16).border()
    b.hwall(6, 0, 24, doors=(4, 12, 20)).hwall(10, 0, 24, doors=(4, 12, 20))
    for x in (8, 16):
        b.vwall(x, 0, 5.9).vwall(x, 10.1, 16)
    rng = np.random.default_rng(7)
    b.carve(10.5, 5.8, 13.5, 6.2)   # collapsed wall sections
    b.carve(6.5, 9.8, 8.5, 10.2)
    b.carve(15.8, 2.5, 16.2, 4.0)
    for _ in range(40):
        w, h = rng.uniform(0.3, 1.4, 2)
        x, y = rng.uniform(1, 23 - w), rng.uniform(0.5, 15.5 - h)
        if x < 3 and 6 < y < 10:       # keep entrance clear
            continue
        b.wall(x, y, x + w, y + h)
    return FloorPlan("rubble", b.free, (1.0, 8.0))


# ------------------------------------------------------- application showcase plans
# Never seen by the optimiser; each stands for a different use of the same swarm.

def _keep_entrance_component(b: _Builder, entrance) -> None:
    """Turn free pockets that cannot be reached from the entrance into solid."""
    lab, _ = ndimage.label(b.free)
    b.free = lab == lab[int(entrance[1] / CELL), int(entrance[0] / CELL)]


def _sky() -> FloorPlan:
    """Drone swarm over a city block: buildings and circular no-fly zones."""
    b = _Builder(36, 24).border()
    rng = np.random.default_rng(21)
    b.disk(24, 15, 2.8).disk(10, 18, 2.0)            # no-fly zones (masts, helipad)
    placed = 0
    while placed < 26:
        w, h = rng.uniform(1.5, 4.0, 2)
        x, y = rng.uniform(1, 35 - w), rng.uniform(1, 23 - h)
        if x < 5 and y < 5:                            # keep the launch pad clear
            continue
        b.wall(x, y, x + w, y + h)
        placed += 1
    _keep_entrance_component(b, (1.5, 1.5))
    return FloorPlan("sky", b.free, (1.5, 1.5))


def _forest() -> FloorPlan:
    """Ground robots under a forest canopy: randomly placed trunks."""
    b = _Builder(24, 16).border()
    rng = np.random.default_rng(5)
    for _ in range(75):
        x, y, r = rng.uniform(0.5, 23.5), rng.uniform(0.5, 15.5), rng.uniform(0.2, 0.55)
        if (x - 1) ** 2 + (y - 8) ** 2 > 4:
            b.disk(x, y, r)
    _keep_entrance_component(b, (1.0, 8.0))
    return FloorPlan("forest", b.free, (1.0, 8.0))


def _orchard() -> FloorPlan:
    """Farm robots/sensors in an orchard: regular tree rows off a headland lane."""
    b = _Builder(24, 16).border()
    for y in np.arange(2.2, 15, 2.4):
        for x in np.arange(4.0, 23, 1.6):
            b.disk(x, y, 0.4)
    return FloorPlan("orchard", b.free, (1.2, 1.2))


def _warehouse() -> FloorPlan:
    """Inventory robots in a warehouse: long racks, a cross aisle and a loading dock."""
    b = _Builder(24, 16).border()
    for y in np.arange(1.8, 15, 2.4):
        b.wall(3.5, y, 11, y + 0.8).wall(13, y, 22.5, y + 0.8)
    return FloorPlan("warehouse", b.free, (1.2, 8.0))


def _cave() -> FloorPlan:
    """Rescue robots in a cave: a winding passage with organic side chambers."""
    b = _Builder(24, 16, solid=True)
    rng = np.random.default_rng(11)
    noise = ndimage.gaussian_filter(rng.standard_normal(b.free.shape), 10)
    b.free = noise > np.quantile(noise, 0.55)
    for x in np.arange(0.3, 23.7, 0.2):                 # main passage
        b.disk(x, 8 + 4 * np.sin(x / 3.5), 1.0, free=True)
    b.border()
    ent = (1.0, 8 + 4 * np.sin(1 / 3.5))
    _keep_entrance_component(b, ent)
    return FloorPlan("cave", b.free, ent)


def _open(w: float = 24, h: float = 16) -> FloorPlan:
    b = _Builder(w, h).border()
    return FloorPlan(f"open:{w:g}x{h:g}", b.free, (w / 2, h / 2))


TRAIN_PLANS = ("office", "hall", "mine")
TEST_PLANS = ("apartment", "rubble")
_FACTORIES = {"office": _office, "hall": _hall, "mine": _mine,
              "apartment": _apartment, "rubble": _rubble, "sky": _sky, "forest": _forest,
              "orchard": _orchard, "warehouse": _warehouse, "cave": _cave}
APP_PLANS = ("sky", "forest", "orchard", "warehouse", "cave")


@lru_cache(maxsize=None)
def get_plan(name: str) -> FloorPlan:
    """``name`` is a registered plan or ``open`` / ``open:WxH`` (centre entrance)."""
    if name.startswith("open"):
        w, h = (24.0, 16.0) if name == "open" else map(float, name.split(":")[1].split("x"))
        return _open(w, h).finalize()
    return _FACTORIES[name]().finalize()


def spawn(plan: FloorPlan, n: int, radius: float, rng: np.random.Generator,
          spread: float = 0.05) -> np.ndarray:
    """Pack ``n`` non-overlapping agents as close to the entrance as the walls allow
    (by geodesic distance), like robots released through a single doorway."""
    gap = 2 * radius + spread
    # hexagonal lattice with random offset, filtered to reachable cells
    ox, oy = rng.uniform(0, gap, 2)
    xs = np.arange(ox, plan.width, gap)
    ys = np.arange(oy, plan.height, gap * np.sqrt(3) / 2)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx + (np.arange(len(ys))[:, None] % 2) * gap / 2
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    iy = (pts[:, 1] / plan.cs).astype(int)
    ix = (pts[:, 0] / plan.cs).astype(int)
    ok = (ix < plan.free.shape[1]) & (iy < plan.free.shape[0])
    pts, iy, ix = pts[ok], iy[ok], ix[ok]
    ok = plan.reach[iy, ix] & (plan.clear[iy, ix] >= radius + 0.05)
    pts, iy, ix = pts[ok], iy[ok], ix[ok]
    if len(pts) < n:
        raise ValueError(f"plan {plan.name} cannot hold {n} agents")
    d = plan.geo[iy, ix] + rng.uniform(0, 1e-3, len(pts))
    return pts[np.argsort(d)[:n]].copy()
