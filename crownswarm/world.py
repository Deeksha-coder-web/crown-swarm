"""Physics and sensing kernels (Numba).

Agents are discs of radius r on a unicycle model. The simulator knows world
positions; controllers only ever receive the body-frame quantities computed
here (see ``Sensors`` in controllers.py).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

N_SECTORS = 8
TWO_PI = 2.0 * math.pi


@njit(cache=True, inline="always")
def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


@njit(cache=True)
def clearance_at(clear, cs, x, y):
    """Bilinear interpolation of the clearance field (cell-centre samples)."""
    fx = x / cs - 0.5
    fy = y / cs - 0.5
    ix = int(math.floor(fx))
    iy = int(math.floor(fy))
    ny, nx = clear.shape
    if ix < 0 or iy < 0 or ix + 1 >= nx or iy + 1 >= ny:
        return 0.0
    tx = fx - ix
    ty = fy - iy
    return ((1 - tx) * (1 - ty) * clear[iy, ix] + tx * (1 - ty) * clear[iy, ix + 1]
            + (1 - tx) * ty * clear[iy + 1, ix] + tx * ty * clear[iy + 1, ix + 1])


@njit(cache=True)
def is_free(free, cs, x, y):
    ix = int(x / cs)
    iy = int(y / cs)
    if ix < 0 or iy < 0 or iy >= free.shape[0] or ix >= free.shape[1]:
        return False
    return free[iy, ix]


@njit(cache=True)
def resolve_overlaps(pos, pairs, r, clear, cs, iters):
    """Push overlapping discs apart (half each); never push an agent into a wall."""
    for _ in range(iters):
        for k in range(pairs.shape[0]):
            i, j = pairs[k, 0], pairs[k, 1]
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            d = math.sqrt(dx * dx + dy * dy)
            if d >= 2 * r:
                continue
            if d < 1e-9:
                dx, dy, d = 1.0, 0.0, 1.0
            push = 0.5 * (2 * r - d) / d
            xi, yi = pos[i, 0] - dx * push, pos[i, 1] - dy * push
            xj, yj = pos[j, 0] + dx * push, pos[j, 1] + dy * push
            ok_i = clearance_at(clear, cs, xi, yi) >= r
            ok_j = clearance_at(clear, cs, xj, yj) >= r
            if ok_i and ok_j:
                pos[i, 0], pos[i, 1] = xi, yi
                pos[j, 0], pos[j, 1] = xj, yj
            elif ok_j:   # i is pinned against a wall: j takes the full push
                xj, yj = pos[j, 0] + 2 * dx * push, pos[j, 1] + 2 * dy * push
                if clearance_at(clear, cs, xj, yj) >= r:
                    pos[j, 0], pos[j, 1] = xj, yj
            elif ok_i:
                xi, yi = pos[i, 0] - 2 * dx * push, pos[i, 1] - 2 * dy * push
                if clearance_at(clear, cs, xi, yi) >= r:
                    pos[i, 0], pos[i, 1] = xi, yi


@njit(cache=True)
def line_of_sight(free, cs, x0, y0, x1, y1):
    d = math.hypot(x1 - x0, y1 - y0)
    n = int(d / (0.5 * cs)) + 1
    for s in range(1, n):
        t = s / n
        if not is_free(free, cs, x0 + t * (x1 - x0), y0 + t * (y1 - y0)):
            return False
    return True


@njit(cache=True)
def los_pairs(free, cs, a, b, ia, ib):
    """Line of sight between a[ia[k]] and b[ib[k]] for every k."""
    out = np.empty(ia.shape[0], dtype=np.bool_)
    for k in range(ia.shape[0]):
        out[k] = line_of_sight(free, cs, a[ia[k], 0], a[ia[k], 1], b[ib[k], 0], b[ib[k], 1])
    return out


@njit(cache=True)
def cast_rays(pos, heading, free, cs, max_range):
    """Distance to the nearest wall along 8 body-frame directions (sector k at k*45deg)."""
    n = pos.shape[0]
    out = np.full((n, N_SECTORS), max_range)
    step = cs * 0.5
    for i in range(n):
        for k in range(N_SECTORS):
            a = heading[i] + k * TWO_PI / N_SECTORS
            c, s = math.cos(a), math.sin(a)
            d = step
            while d < max_range:
                if not is_free(free, cs, pos[i, 0] + c * d, pos[i, 1] + s * d):
                    out[i, k] = d
                    break
                d += step
    return out


@njit(cache=True)
def sector_signal(pos, heading, ptr, nbr, dist, los, r, R, occlusion):
    """Far-red-like neighbour signal in 8 body-frame sectors.

    Each visible neighbour contributes (1 - d/R)^2, split linearly between the
    two nearest sectors. A neighbour is hidden if a wall blocks the line of
    sight or (when ``occlusion``) a closer agent's disc covers its bearing.
    """
    n = pos.shape[0]
    out = np.zeros((n, N_SECTORS))
    width = TWO_PI / N_SECTORS
    for i in range(n):
        a, b = ptr[i], ptr[i + 1]
        if a == b:
            continue
        order = np.argsort(dist[a:b])
        occ_b = np.empty(b - a)
        occ_w = np.empty(b - a)
        m_occ = 0
        for oi in range(b - a):
            e = a + order[oi]
            if not los[e]:
                continue
            j = nbr[e]
            d = dist[e]
            bearing = wrap(math.atan2(pos[j, 1] - pos[i, 1], pos[j, 0] - pos[i, 0]) - heading[i])
            hidden = False
            if occlusion:
                for q in range(m_occ):
                    if abs(wrap(bearing - occ_b[q])) < occ_w[q]:
                        hidden = True
                        break
                occ_b[m_occ] = bearing
                occ_w[m_occ] = math.asin(min(1.0, r / max(d, 1e-9)))
                m_occ += 1
            if hidden or d >= R:
                continue
            s = (1.0 - d / R) ** 2
            f = (bearing % TWO_PI) / width
            k = int(f) % N_SECTORS
            w = f - int(f)
            out[i, k] += s * (1.0 - w)
            out[i, (k + 1) % N_SECTORS] += s * w
    return out


@njit(cache=True)
def geodesic_lloyd(grid, ay, ax, relax):
    """Discrete geodesic Lloyd step on a coarse grid (privileged baseline).

    Multi-source BFS gives each agent its geodesic Voronoi region; the target is
    the owned cell nearest ``agent + relax * (centroid - agent)`` (relax > 1 is
    over-relaxation, which speeds up unpacking); the returned waypoint is the next
    cell on the shortest in-region path from the agent to that target.
    """
    ny, nx = grid.shape
    n = ay.shape[0]
    owner = np.full((ny, nx), -1, dtype=np.int64)
    qy = np.empty(ny * nx + n, dtype=np.int64)
    qx = np.empty(ny * nx + n, dtype=np.int64)
    head, tail = 0, 0
    for i in range(n):
        if owner[ay[i], ax[i]] < 0:
            owner[ay[i], ax[i]] = i
            qy[tail], qx[tail] = ay[i], ax[i]
            tail += 1
    while head < tail:
        y, x = qy[head], qx[head]
        head += 1
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx and grid[yy, xx] and owner[yy, xx] < 0:
                owner[yy, xx] = owner[y, x]
                qy[tail], qx[tail] = yy, xx
                tail += 1
    sy = np.zeros(n)
    sx = np.zeros(n)
    cnt = np.zeros(n, dtype=np.int64)
    for y in range(ny):
        for x in range(nx):
            o = owner[y, x]
            if o >= 0:
                sy[o] += y
                sx[o] += x
                cnt[o] += 1
    ty = ay.copy()
    tx = ax.copy()
    best = np.full(n, 1e18)
    for y in range(ny):
        for x in range(nx):
            o = owner[y, x]
            if o >= 0:
                gy = ay[o] + relax * (sy[o] / cnt[o] - ay[o])
                gx = ax[o] + relax * (sx[o] / cnt[o] - ax[o])
                d = (y - gy) ** 2 + (x - gx) ** 2
                if d < best[o]:
                    best[o] = d
                    ty[o], tx[o] = y, x
    wy = ty.copy()
    wx = tx.copy()
    dist = np.full((ny, nx), -1, dtype=np.int64)
    for i in range(n):
        if cnt[i] == 0 or (ty[i] == ay[i] and tx[i] == ax[i]):
            continue
        head, tail = 0, 1
        qy[0], qx[0] = ty[i], tx[i]
        dist[ty[i], tx[i]] = 0
        while head < tail:
            y, x = qy[head], qx[head]
            head += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < ny and 0 <= xx < nx and owner[yy, xx] == i and dist[yy, xx] < 0:
                    dist[yy, xx] = dist[y, x] + 1
                    qy[tail], qx[tail] = yy, xx
                    tail += 1
        y0, x0 = ay[i], ax[i]
        if dist[y0, x0] >= 0:
            bd = dist[y0, x0]
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y0 + dy, x0 + dx
                if 0 <= yy < ny and 0 <= xx < nx and 0 <= dist[yy, xx] < bd:
                    bd = dist[yy, xx]
                    wy[i], wx[i] = yy, xx
        for k in range(tail):
            dist[qy[k], qx[k]] = -1
    return wy, wx, cnt


@njit(cache=True)
def _blocker(pos, i, x, y, ptr, nbr, r):
    """Index of the first agent overlapping a disc at (x, y), or -1."""
    for e in range(ptr[i], ptr[i + 1]):
        j = nbr[e]
        if (pos[j, 0] - x) ** 2 + (pos[j, 1] - y) ** 2 < 4 * r * r:
            return j
    return -1


@njit(cache=True)
def move_agents(pos, disp, clear, cs, r, ptr, nbr, order):
    """Sequential (random-order) move that never creates an overlap.

    Try the full step; if another agent blocks it, try the tangential part of the
    step (sliding around that agent); then axis-aligned slides (for walls);
    otherwise stay. ``ptr/nbr``: CSR list of agents reachable this step."""
    for k in range(order.shape[0]):
        i = order[k]
        x, y = pos[i, 0], pos[i, 1]
        dx, dy = disp[i, 0], disp[i, 1]
        cx = np.empty(4)
        cy = np.empty(4)
        cx[0], cy[0] = x + dx, y + dy
        m = 1
        j = _blocker(pos, i, x + dx, y + dy, ptr, nbr, r)
        if j >= 0:
            tx, ty = -(pos[j, 1] - y), pos[j, 0] - x
            tn = math.sqrt(tx * tx + ty * ty) + 1e-12
            tx, ty = tx / tn, ty / tn
            p = dx * tx + dy * ty
            cx[m], cy[m] = x + p * tx, y + p * ty
            m += 1
        cx[m], cy[m] = x + dx, y
        cx[m + 1], cy[m + 1] = x, y + dy
        m += 2
        for c in range(m):
            if clearance_at(clear, cs, cx[c], cy[c]) >= r and _blocker(pos, i, cx[c], cy[c], ptr, nbr, r) < 0:
                pos[i, 0], pos[i, 1] = cx[c], cy[c]
                break
