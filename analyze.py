"""Turn results/*.csv into tables (results/summary.md) and figures (results/figs/).

    python analyze.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from crownswarm import TEST_PLANS, TRAIN_PLANS, get_plan  # noqa: E402
from crownswarm.metrics import coverage_ceiling  # noqa: E402

RES = Path("results")
FIG = RES / "figs"
ORDER = ["RW", "A", "B", "C", "PF", "Boids", "Lloyd"]
# fixed controller -> colour (validated categorical palette, never re-cycled); markers as a 2nd cue
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "PF": "#eda100", "Boids": "#e87ba4",
         "Lloyd": "#008300", "RW": "#4a3aa7"}
MARK = {"A": "o", "B": "s", "C": "D", "PF": "^", "Boids": "v", "Lloyd": "P", "RW": "X"}
LABEL = {"RW": "Random walk", "A": "A: abrasion", "B": "B: light", "C": "C: hybrid",
         "PF": "Potential field", "Boids": "Boids", "Lloyd": "Lloyd (privileged)"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e3"
METRICS = [("coverage_mean_2nd_half", "Coverage (2nd half)", "higher"),
           ("explored", "Area explored", "higher"),
           ("evenness_cv", "Spacing CV", "lower"),
           ("collisions_per_agent", "Collisions / agent", "lower"),
           ("path_per_agent", "Path / agent (m)", "lower"),
           ("retained", "Coverage kept after 20% loss", "higher")]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})


def ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), *np.percentile(boots, [2.5, 97.5])


def fmt(x):
    m, lo, hi = ci(x)
    return "–" if np.isnan(m) else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def load(name):
    f = RES / f"{name}.csv"
    return pd.read_csv(f) if f.exists() else None


def ctrl_of(variant):
    return variant.split("-")[0]


# ------------------------------------------------------------------------ compare

def compare(md):
    df = load("compare")
    if df is None:
        return
    df["split"] = np.where(df.plan.isin(TEST_PLANS), "test", "train")
    # retained = final coverage with a 20% failure / final coverage of the same seed intact
    key = ["variant", "plan", "seed"]
    failed = df[df.fail_frac > 0].set_index(key).coverage
    df = df[df.fail_frac == 0].copy()
    df["retained"] = (failed / df.set_index(key).coverage).reindex(df.set_index(key).index).to_numpy()
    test = df[df.split == "test"]
    ceil = {p: coverage_ceiling(get_plan(p), 60) for p in TEST_PLANS}

    md.append("## Main comparison — unseen test plans (apartment, rubble)\n")
    md.append(f"60 agents, 800 steps, {df.seed.nunique()} seeds per plan. Metrics from intact runs; "
              "'coverage kept' pairs each run with the same seed where 20% of agents are removed at step 400. "
              "Mean [95% bootstrap CI]. Static coverage ceiling (greedy placement, "
              "line of sight): " + ", ".join(f"{p} {c:.3f}" for p, c in ceil.items()) + ".\n")
    md.append("| Controller | " + " | ".join(m[1] for m in METRICS) + " |")
    md.append("|---" * (len(METRICS) + 1) + "|")
    for v in ORDER:
        d = test[test.variant == v]
        md.append(f"| {LABEL[v]} | " + " | ".join(fmt(d[m]) for m, *_ in METRICS) + " |")

    md.append("\n### Generalisation (coverage, 2nd half): train plans vs unseen test plans\n")
    md.append("| Controller | Train | Test |\n|---|---|---|")
    for v in ORDER:
        d = df[df.variant == v]
        md.append(f"| {LABEL[v]} | {fmt(d[d.split == 'train'].coverage_mean_2nd_half)} | "
                  f"{fmt(d[d.split == 'test'].coverage_mean_2nd_half)} |")

    md.append("\n### Paired tests on test plans (same seeds → same start), Wilcoxon signed-rank\n")
    md.append("| Comparison | Metric | Mean diff | p |\n|---|---|---|---|")
    for a, b in [("A", "RW"), ("B", "PF"), ("C", "A"), ("C", "B"), ("A", "B"), ("B", "Boids")]:
        for m, name, _ in METRICS[:4]:
            x = test[test.variant == a].sort_values(["plan", "seed"])[m].to_numpy()
            y = test[test.variant == b].sort_values(["plan", "seed"])[m].to_numpy()
            if len(x) and len(x) == len(y):
                p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
                md.append(f"| {a} vs {b} | {name} | {np.mean(x - y):+.3f} | {p:.2g} |")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for ax, (m, name, better) in zip(axes.flat, METRICS):
        for k, v in enumerate(ORDER[::-1]):
            mu, lo, hi = ci(test[test.variant == v][m])
            ax.plot([lo, hi], [k, k], color=COLOR[v], lw=2, solid_capstyle="round")
            ax.plot(mu, k, marker=MARK[v], color=COLOR[v], ms=8, mec="white", mew=1.5)
        ax.set_yticks(range(len(ORDER)), [LABEL[v] for v in ORDER[::-1]])
        ax.set_title(f"{name} ({'↑' if better == 'higher' else '↓'} better)", fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", visible=False)
        if m == "coverage_mean_2nd_half":
            c = np.mean(list(ceil.values()))
            ax.axvline(c, color=MUTED, ls="--", lw=1)
            ax.text(c, -0.45, "ceiling ", color=MUTED, fontsize=8, ha="right", va="center")
    fig.suptitle("Unseen test plans: mean and 95% CI", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "compare.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------- sweeps

SWEEP_AXES = {"sway": ("sway", "Wind sway (m/step)"), "noise": ("sensor_noise", "Sensor noise"),
              "n": ("n_agents", "Swarm size"), "fail": ("fail_frac", "Fraction removed mid-run")}


def line_panel(ax, df, x, y, variants):
    for v in variants:
        d = df[df.variant == v]
        xs = sorted(d[x].unique())
        s = np.array([ci(d[d[x] == xv][y]) for xv in xs])
        ax.fill_between(xs, s[:, 1], s[:, 2], color=COLOR[ctrl_of(v)], alpha=0.15, lw=0)
        ax.plot(xs, s[:, 0], color=COLOR[ctrl_of(v)], lw=2, marker=MARK[ctrl_of(v)], ms=7,
                mec="white", mew=1.2, label=LABEL.get(v, v))


def sweeps(md):
    found = [(k, load(f"sweep_{k}")) for k in SWEEP_AXES]
    found = [(k, d) for k, d in found if d is not None]
    if not found:
        return
    fig, axes = plt.subplots(len(found), 2, figsize=(10, 3.2 * len(found)), squeeze=False)
    md.append("\n## Sweeps (test plans; coverage in 2nd half)\n")
    md.append("All sweep runs use 60 agents with 20% removed at step 400 unless that is the swept "
              "variable, so values sit below the intact runs of the main table.\n")
    for row, (k, df) in zip(axes, found):
        x, xl = SWEEP_AXES[k]
        line_panel(row[0], df, x, "coverage_mean_2nd_half", ORDER)
        line_panel(row[1], df, x, "collisions_per_agent", ORDER)
        row[0].set_ylabel("Coverage (2nd half)")
        row[1].set_ylabel("Collisions / agent")
        for ax in row:
            ax.set_xlabel(xl)
        piv = df.groupby(["variant", x]).coverage_mean_2nd_half.mean().unstack()
        md.append(f"**{xl}**\n\n" + piv.reindex(ORDER).round(3).to_markdown() + "\n")
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "sweeps.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------- ablate

def ablate(md):
    df = load("ablate")
    if df is None:
        return
    order = list(dict.fromkeys(df.variant))
    md.append("\n## Ablations (test plans, 20% removed at step 400)\n")
    md.append("| Variant | Coverage (2nd half) | Explored | Spacing CV | Collisions / agent |\n|---|---|---|---|---|")
    for v in order:
        d = df[df.variant == v]
        md.append(f"| {v} | {fmt(d.coverage_mean_2nd_half)} | {fmt(d.explored)} | "
                  f"{fmt(d.evenness_cv)} | {fmt(d.collisions_per_agent)} |")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (m, name) in zip(axes, [("coverage_mean_2nd_half", "Coverage (2nd half)"),
                                     ("collisions_per_agent", "Collisions / agent")]):
        for k, v in enumerate(order[::-1]):
            mu, lo, hi = ci(df[df.variant == v][m])
            c = COLOR[ctrl_of(v)]
            ax.plot([lo, hi], [k, k], color=c, lw=2)
            ax.plot(mu, k, marker=MARK[ctrl_of(v)], color=c if "-" not in v else "white", mec=c,
                    mew=1.8, ms=8)
        ax.set_yticks(range(len(order)), order[::-1])
        ax.grid(axis="y", visible=False)
        ax.set_title(name, loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "ablate.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------- biology

def biology(md):
    df = load("biology")
    if df is None:
        return
    md.append("\n## Biology check: gap width vs sway (open arena, 150 agents)\n")
    md.append("Gap = nearest-neighbour distance minus two body radii. Spearman ρ between sway and "
              "gap over all runs; the field prediction is ρ > 0.\n")
    from scipy.stats import spearmanr
    md.append("| Controller | ρ(sway, gap) | p | gap at min sway | gap at max sway |\n|---|---|---|---|---|")
    for v in dict.fromkeys(df.variant):
        d = df[df.variant == v]
        rho, p = spearmanr(d.sway, d.gap_mean)
        lo_s, hi_s = d.sway.min(), d.sway.max()
        md.append(f"| {LABEL.get(v, v)} | {rho:+.2f} | {p:.2g} | {fmt(d[d.sway == lo_s].gap_mean)} | "
                  f"{fmt(d[d.sway == hi_s].gap_mean)} |")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    vs = list(dict.fromkeys(df.variant))
    line_panel(axes[0], df, "sway", "gap_mean", vs)
    line_panel(axes[1], df, "sway", "shyness", vs)
    axes[0].set_ylabel("Mean gap between crowns (m)")
    axes[1].set_ylabel("Fraction not touching a neighbour")
    for ax in axes:
        ax.set_xlabel("Wind sway (m/step)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "biology.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------- scaling

def scaling(md):
    df = load("scaling")
    if df is None:
        return
    df["ms_per_step"] = df.sec_per_step * 1000
    vs = list(dict.fromkeys(df.variant))
    md.append("\n## Scaling (open arena, constant density)\n")
    md.append("**Coverage (2nd half)**\n\n" +
              df.groupby(["variant", "n_agents"]).coverage_mean_2nd_half.mean().unstack().round(3).to_markdown())
    md.append("\n**Wall-clock ms per step (single core)**\n\n" +
              df.groupby(["variant", "n_agents"]).ms_per_step.mean().unstack().round(1).to_markdown() + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    line_panel(axes[0], df, "n_agents", "coverage_mean_2nd_half", vs)
    line_panel(axes[1], df, "n_agents", "ms_per_step", vs)
    axes[0].set_ylabel("Coverage (2nd half)")
    axes[1].set_ylabel("ms per step")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Swarm size")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "scaling.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    md = ["# Results summary\n", f"Train plans: {', '.join(TRAIN_PLANS)}. Test plans (never seen by the "
          f"optimiser): {', '.join(TEST_PLANS)}.\n"]
    for f in (compare, sweeps, ablate, biology, scaling):
        f(md)
    (RES / "summary.md").write_text("\n".join(md) + "\n")
    print("wrote results/summary.md and results/figs/*.png")


if __name__ == "__main__":
    main()
