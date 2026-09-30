"""Swarm performance metrics."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .world import los_pairs


def covered_mask(pos: np.ndarray, plan, radius: float) -> np.ndarray:
    """Coverage points within ``radius`` of an agent *and* in its line of sight
    (a victim behind a wall is not found by looking through it)."""
    covered = np.zeros(len(plan.cover_pts), dtype=bool)
    if len(pos) == 0:
        return covered
    m = cKDTree(pos).sparse_distance_matrix(plan.cover_tree, radius, output_type="ndarray")
    if len(m) == 0:
        return covered
    ia = m["i"].astype(np.int64)
    ib = m["j"].astype(np.int64)
    ok = los_pairs(plan.free, plan.cs, pos, plan.cover_pts, ia, ib)
    covered[ib[ok]] = True
    return covered


def spacing(pos: np.ndarray, radius: float) -> dict:
    """Nearest-neighbour statistics. ``evenness_cv`` is std/mean of NN distance
    (lower = more even). ``gap`` is the free space between crown edges."""
    if len(pos) < 2:
        return {"nn_mean": np.nan, "evenness_cv": np.nan, "gap_mean": np.nan, "gap_median": np.nan,
                "shyness": np.nan}
    d, _ = cKDTree(pos).query(pos, k=2)
    nn = d[:, 1]
    gap = nn - 2 * radius
    return {"nn_mean": float(nn.mean()),
            "evenness_cv": float(nn.std() / nn.mean()),
            "gap_mean": float(gap.mean()),
            "gap_median": float(np.median(gap)),
            # fraction of agents whose crown is not touching any neighbour
            "shyness": float((gap > 0.05).mean())}


def coverage_ceiling(plan, n: int, radius: float = 2.0) -> float:
    """Static reference: greedy max-coverage placement of ``n`` sensors on reachable
    points (line-of-sight footprints). Greedy is within (1 - 1/e) of optimal and
    usually far closer, so this is a practical ceiling for final coverage."""
    from scipy.sparse import csr_matrix
    iy = (plan.cover_pts[:, 1] / plan.cs).astype(int)
    ix = (plan.cover_pts[:, 0] / plan.cs).astype(int)
    cand = plan.cover_pts[plan.reach[iy, ix]]
    m = cKDTree(cand).sparse_distance_matrix(plan.cover_tree, radius, output_type="ndarray")
    ia, ib = m["i"].astype(np.int64), m["j"].astype(np.int64)
    ok = los_pairs(plan.free, plan.cs, cand, plan.cover_pts, ia, ib)
    A = csr_matrix((np.ones(ok.sum()), (ia[ok], ib[ok])), shape=(len(cand), len(plan.cover_pts)))
    uncovered = np.ones(len(plan.cover_pts))
    for _ in range(n):
        gain = A @ uncovered
        k = int(np.argmax(gain))
        if gain[k] <= 0:
            break
        uncovered[A[k].indices] = 0
    return float(1 - uncovered.mean())
