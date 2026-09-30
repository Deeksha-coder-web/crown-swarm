"""Watch the swarm deploy.

    python demo.py                                   # Rule B on the rubble plan, live window
    python demo.py --controller A B --plan rubble    # side-by-side comparison
    python demo.py --controller B --fail 0.2 --gif b.gif
    python demo.py --plan apartment --grid snapshots.png   # final frame of every controller

What you see: robot discs coloured by state (moving / settled / bumping),
short motion trails, the floor area each robot can currently see (2 m,
line of sight) in bright teal, area seen earlier in dim teal, and a live
coverage curve against the best static placement ("ceiling").
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
import numpy as np

from crownswarm import CONTROLLERS, EpisodeConfig, get_plan, run_episode
from crownswarm.floorplans import COVER_CELL

PARAM_DIR = Path("results/params")

# ------------------------------------------------------------------- look & feel
BG = "#0b0f14"
FLOOR = "#121a23"
WALL = "#4a5a6e"
INK = "#e6edf3"
MUTED = "#8b98a8"
SEEN = np.array([0.18, 0.77, 0.71])          # teal
STATE = {"moving": "#7cc4ff", "settled": "#ffd166", "bumping": "#ff5c7a"}
# controller colours: dark-mode steps of the validated categorical palette
CTRL_COLOR = {"A": "#3987e5", "B": "#d95926", "C": "#199e70", "PF": "#c98500",
              "Boids": "#d55181", "Lloyd": "#008300", "RW": "#9085e9"}
TITLE = {"RW": ("Random walk", "bump sensor only"),
         "A": ("Rule A · abrasion", "bump sensor + vigour memory"),
         "B": ("Rule B · light", "8-sector neighbour signal, occluded"),
         "C": ("Rule C · hybrid", "signal + bump"),
         "PF": ("Potential field", "exact range & bearing"),
         "Boids": ("Boids dispersion", "range, bearing & neighbour headings"),
         "Lloyd": ("Lloyd", "privileged: global positions + map")}
TRAIL = 14          # frames of trail
FADE = 40           # frames the removed-robot markers stay visible


def load_params(name: str) -> dict | None:
    f = PARAM_DIR / f"{name}.json"
    return json.loads(f.read_text())["params"] if f.exists() else None


class MapView:
    """One floor plan with the swarm drawn on it."""

    def __init__(self, ax, plan, name, out, radius=0.3):
        from matplotlib.collections import EllipseCollection, LineCollection
        self.ax, self.plan, self.out, self.r = ax, plan, out, radius
        ax.set_facecolor(BG)
        ext = (0, plan.width, 0, plan.height)
        # coverage raster (0.5 m cells); floor underneath, walls on top
        self.cx = (plan.cover_pts[:, 0] / COVER_CELL).astype(int)
        self.cy = (plan.cover_pts[:, 1] / COVER_CELL).astype(int)
        shape = (int(np.ceil(plan.height / COVER_CELL)), int(np.ceil(plan.width / COVER_CELL)))
        self.rgba = np.zeros(shape + (4,))
        self.rgba[..., :3] = SEEN
        floor = np.zeros(plan.free.shape + (4,))
        floor[plan.free] = matplotlib.colors.to_rgba(FLOOR)
        ax.imshow(floor, origin="lower", extent=ext, interpolation="nearest", zorder=0)
        self.cov_img = ax.imshow(self.rgba, origin="lower", extent=ext, interpolation="bilinear", zorder=1)
        walls = np.zeros(plan.free.shape + (4,))
        walls[~plan.free] = matplotlib.colors.to_rgba(WALL)
        ax.imshow(walls, origin="lower", extent=ext, interpolation="nearest", zorder=2)
        ex, ey = plan.entrance
        ax.plot(ex, ey, marker="v", ms=9, color=INK, mec=BG, zorder=6)
        ax.text(ex, ey + 0.55, "entrance", color=MUTED, fontsize=7, ha="center", zorder=6)
        ax.set_xlim(0, plan.width)
        ax.set_ylim(0, plan.height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        self.trails = LineCollection([], linewidths=1.2, zorder=3, capstyle="round")
        ax.add_collection(self.trails)
        self.bodies = EllipseCollection(2 * radius, 2 * radius, 0, units="xy", offsets=np.zeros((0, 2)),
                                        offset_transform=ax.transData, zorder=4, linewidths=0.8)
        ax.add_collection(self.bodies)
        self.ticks = LineCollection([], colors=BG, linewidths=1.4, zorder=5)
        ax.add_collection(self.ticks)
        self.lost = ax.scatter([], [], marker="x", s=28, c=MUTED, linewidths=1.4, zorder=4)
        self.banner = ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center", va="center",
                              fontsize=15, weight="bold", color=STATE["bumping"], zorder=7)

        title, sub = TITLE.get(name, (name, ""))
        ax.text(0, 1.075, title, transform=ax.transAxes, color=INK, fontsize=13, weight="bold", va="bottom")
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=8.5, va="bottom")
        self.big = ax.text(1, 1.02, "", transform=ax.transAxes, color=CTRL_COLOR.get(name, INK),
                           fontsize=22, weight="bold", ha="right", va="bottom")
        self.stats = ax.text(1, -0.02, "", transform=ax.transAxes, color=MUTED, fontsize=8.5,
                             ha="right", va="top")
        self.hist = defaultdict(lambda: deque(maxlen=TRAIL))
        self.explored = np.zeros(len(plan.cover_pts), dtype=bool)

    def draw(self, k):
        f = self.out["frames"][k]
        pos, ids = f["pos"], f["ids"]
        # coverage: seen now = bright, seen before = dim
        if k == 0 or k != getattr(self, "_last", -1) + 1:   # restart or jump: no stale trails
            self.explored[:] = False
            self.hist.clear()
        self._last = k
        self.explored |= f["covered"]
        alpha = np.where(f["covered"], 0.42, np.where(self.explored, 0.12, 0.0))
        self.rgba[..., 3] = 0
        self.rgba[self.cy, self.cx, 3] = alpha
        self.cov_img.set_data(self.rgba)

        # trails
        segs, cols = [], []
        base = matplotlib.colors.to_rgba(STATE["moving"])
        alive = set(ids.tolist())
        for i in list(self.hist):
            if i not in alive:
                del self.hist[i]
        for i, p in zip(ids, pos):
            h = self.hist[i]
            h.append(p.copy())
            for a in range(1, len(h)):
                segs.append([h[a - 1], h[a]])
                cols.append((*base[:3], 0.35 * a / len(h)))
        self.trails.set_segments(segs)
        self.trails.set_color(cols)

        # bodies coloured by state
        state = np.where(f["contact"], "bumping", np.where(f["speed"] < 0.1, "settled", "moving"))
        face = [STATE[s] for s in state]
        self.bodies.set_offsets(pos)
        self.bodies.set_facecolors(face)
        self.bodies.set_edgecolors(BG)
        d = np.column_stack([np.cos(f["heading"]), np.sin(f["heading"])]) * self.r * 0.9
        self.ticks.set_segments(np.stack([pos, pos + d], axis=1))

        # failure event
        rem = self.out.get("removed")
        if rem is not None and f["t"] >= rem[0]:
            age = (f["t"] - rem[0]) / (self.out["frames"][1]["t"] - self.out["frames"][0]["t"])
            vis = age < FADE
            self.lost.set_offsets(rem[1] if vis else np.zeros((0, 2)))
            self.lost.set_alpha(max(0.0, 1 - age / FADE))
            frac = len(rem[1]) / (len(rem[1]) + len(pos))
            self.banner.set_text(f"−{frac:.0%} robots lost" if age < 18 else "")
        else:
            self.lost.set_offsets(np.zeros((0, 2)))
            self.banner.set_text("")

        self.big.set_text(f"{f['covered'].mean():.0%}")
        n0 = self.out["n_agents"]
        self.stats.set_text(f"step {f['t']:>4}   ·   robots {len(pos)}/{n0}   ·   "
                            f"collisions {f['collisions']}")
        return [self.cov_img, self.trails, self.bodies, self.ticks, self.lost, self.banner, self.big, self.stats]


def style_curve(ax, steps):
    ax.set_facecolor(BG)
    ax.set_xlim(0, steps)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(color="#1e2833", lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("step", color=MUTED, fontsize=8)
    ax.set_ylabel("floor seen now", color=MUTED, fontsize=8)


def render(names, outs, plan, show_ceiling=True):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    k = len(names)
    fig = plt.figure(figsize=(8.2 * k if k > 1 else 10, 8.4 if k == 1 else 7.6), facecolor=BG)
    gs = GridSpec(2, k, figure=fig, height_ratios=[3.2, 1], hspace=0.28, wspace=0.08,
                  left=0.05, right=0.97, top=0.9, bottom=0.08)
    views = [MapView(fig.add_subplot(gs[0, j]), plan, n, o) for j, (n, o) in enumerate(zip(names, outs))]
    cax = fig.add_subplot(gs[1, :])
    steps = outs[0]["steps"]
    style_curve(cax, steps)
    if show_ceiling:
        from crownswarm.metrics import coverage_ceiling
        c = coverage_ceiling(plan, outs[0]["n_agents"])
        cax.axhline(c, color=MUTED, ls=(0, (2, 3)), lw=1)
        cax.text(steps, c + 0.02, "best static placement ", color=MUTED, fontsize=7.5, ha="right", va="bottom")
    rem = outs[0].get("removed")
    if rem is not None:
        cax.axvline(rem[0], color=STATE["bumping"], lw=1, alpha=0.6)
        cax.text(rem[0], 0.04, f" {len(rem[1])} robots lost", color=STATE["bumping"], fontsize=7.5)
    curves = []
    for n, o in zip(names, outs):
        (line,) = cax.plot([], [], color=CTRL_COLOR.get(n, INK), lw=2.2, label=TITLE.get(n, (n,))[0])
        (dot,) = cax.plot([], [], "o", color=CTRL_COLOR.get(n, INK), ms=6, mec=BG, mew=1.5)
        t = np.array([f["t"] for f in o["frames"]])
        y = np.array([f["covered"].mean() for f in o["frames"]])
        curves.append((line, dot, t, y))
    handles = [Line2D([], [], marker="o", ls="", mfc=c, mec=BG, ms=8, label=s) for s, c in STATE.items()]
    handles += [Line2D([], [], marker="s", ls="", mfc=(*SEEN, a), mec="none", ms=9, label=l)
                for a, l in ((0.75, "seen now"), (0.3, "seen earlier"))]
    if k > 1:
        handles += [Line2D([], [], color=CTRL_COLOR.get(n, INK), lw=2.2, label=TITLE.get(n, (n,))[0])
                    for n in names]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8.5,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.995), handletextpad=0.3, columnspacing=1.4)

    def update(i):
        arts = []
        for v in views:
            arts += v.draw(min(i, len(v.out["frames"]) - 1))
        for line, dot, t, y in curves:
            j = min(i, len(t) - 1)
            line.set_data(t[: j + 1], y[: j + 1])
            dot.set_data([t[j]], [y[j]])
            arts += [line, dot]
        return arts

    return fig, update, max(len(o["frames"]) for o in outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", nargs="+", default=["B"], choices=list(CONTROLLERS),
                    help="one controller, or several to compare side by side")
    ap.add_argument("--plan", default="rubble")
    ap.add_argument("--agents", type=int, default=60)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--sway", type=float, default=0.02)
    ap.add_argument("--fail", type=float, default=0.2, help="fraction removed at mid-run (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=4, help="simulation steps per animation frame")
    ap.add_argument("--gif", help="write an animated GIF instead of opening a window")
    ap.add_argument("--grid", help="write a PNG with the final frame of every controller")
    args = ap.parse_args()

    if args.gif or args.grid:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    plan = get_plan(args.plan)

    def run(ctrl, every):
        return run_episode(EpisodeConfig(plan=args.plan, controller=ctrl, params=load_params(ctrl),
                                         n_agents=args.agents, steps=args.steps, sway=args.sway,
                                         fail_frac=args.fail, seed=args.seed, record_every=every))

    if args.grid:
        names = list(CONTROLLERS)
        cols = 4
        rows = int(np.ceil(len(names) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.3 * rows), facecolor=BG)
        for ax in axes.flat:
            ax.set_facecolor(BG)
            ax.axis("off")
        for ax, name in zip(axes.flat, names):
            out = run(name, args.steps - 1)
            ax.axis("on")
            v = MapView(ax, plan, name, out)
            v.draw(0)
            v.draw(len(out["frames"]) - 1)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.04, wspace=0.08, hspace=0.35)
        Path(args.grid).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.grid, dpi=120, facecolor=BG)
        print("wrote", args.grid)
        return

    outs = [run(n, args.every) for n in args.controller]
    for n, o in zip(args.controller, outs):
        print(f"{n}: final coverage {o['coverage']:.0%}, collisions/robot {o['collisions_per_agent']:.1f}")
    fig, update, nframes = render(args.controller, outs, plan)
    ani = animation.FuncAnimation(fig, update, frames=nframes, interval=40, blit=False)
    if args.gif:
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        ani.save(args.gif, writer=animation.PillowWriter(fps=25), dpi=80,
                 savefig_kwargs={"facecolor": BG})
        print("wrote", args.gif)
    else:
        plt.show()


if __name__ == "__main__":
    main()
