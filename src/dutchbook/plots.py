"""SVG charts for the README, drawn with matplotlib and made byte-for-byte repeatable.

Colours come from a validated categorical palette (blue, orange, aqua), always in
that order and always with a second encoding (marker shape or a direct label), so no
chart depends on colour alone. Every chart has the same numbers in a table in the
README or in results/report.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dutchbook.analysis import ModelScore
from dutchbook.items import FAMILIES

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "svg.hashsalt": "dutchbook",
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
    }
)


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def _cents(x: float) -> float:
    return 100.0 * x


def headline(models: Sequence[ModelScore], baselines: Sequence[ModelScore], path: Path) -> None:
    """Single-ask vs averaged vs noise-only arbitrage per model, with 95% CIs."""
    rows = [*models, *baselines]
    fig, ax = plt.subplots(figsize=(8.0, 0.55 * len(rows) + 1.4))
    series = [
        ("single", "asked once", BLUE, "o"),
        ("averaged", "averaged over 5 asks", ORANGE, "D"),
        ("null_averaged", "averaged, if it were only noise", AQUA, "s"),
    ]
    offsets = (-0.2, 0.0, 0.2)
    for (metric, label, colour, marker), off in zip(series, offsets, strict=True):
        ys, xs, lo, hi = [], [], [], []
        for i, m in enumerate(rows):
            if metric not in m.headline:
                continue
            iv = m.headline[metric]
            ys.append(len(rows) - 1 - i + off)
            xs.append(_cents(iv.value))
            lo.append(_cents(iv.value - iv.lo))
            hi.append(_cents(iv.hi - iv.value))
        ax.errorbar(
            xs,
            ys,
            xerr=[lo, hi],
            fmt=marker,
            color=colour,
            ecolor=colour,
            elinewidth=2,
            capsize=0,
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=label,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([m.model for m in reversed(rows)])
    if baselines:
        ax.axhline(len(baselines) - 0.5, color=GRID, lw=1)
    ax.set_xlabel("guaranteed loss per $1 staked (cents), mean over the six families")
    ax.set_xlim(left=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    ax.set_title("What a bookie can lock in against each forecaster")
    _save(fig, path)


def families(models: Sequence[ModelScore], path: Path) -> None:
    """Families x models, single-ask and averaged arbitrage, as two annotated grids."""
    fams = [f for f in FAMILIES if any(f in m.by_family for m in models)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(fams) + 1.6), sharey=True)
    for ax, metric, title in zip(
        axes, ("single", "averaged"), ("asked once", "averaged over 5 asks"), strict=True
    ):
        grid = np.array(
            [
                [
                    _cents(m.by_family[f][metric].value) if f in m.by_family else np.nan
                    for m in models
                ]
                for f in fams
            ]
        )
        vmax = float(np.nanmax(grid)) if np.isfinite(grid).any() else 1.0
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(vmax, 1e-9), aspect="auto")
        for i in range(len(fams)):
            for j in range(len(models)):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.1f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=SURFACE if v > 0.55 * vmax else INK,
                    )
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.model.replace("claude-", "") for m in models], rotation=20)
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels(fams)
        ax.set_title(f"{title} (cents per $1)")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    _save(fig, path)


def null_panels(models: Sequence[ModelScore], path: Path) -> None:
    """For each model, the noise-only distribution of mean averaged arbitrage."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models, strict=True):
        draws, observed = _macro_null(m)
        ax.hist(_cents(1) * draws, bins=25, color=AQUA, linewidth=0)
        ax.axvline(_cents(observed), color=ORANGE, lw=2)
        ax.text(
            _cents(observed),
            ax.get_ylim()[1] * 0.95,
            "observed ",
            color=INK,
            fontsize=8,
            ha="right",
            va="top",
        )
        ax.set_title(m.model.replace("claude-", ""), fontsize=10)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(left=0, right=max(_cents(observed) * 1.15, ax.get_xlim()[1]))
    axes[0].set_xlabel("averaged arbitrage, cents per $1")
    fig.suptitle(
        "Noise alone (histogram) vs what the model left (line)",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, path)


def _macro_null(m: ModelScore) -> tuple[Any, float]:
    """Null draws and the observed value, both as the mean over families."""
    fams = sorted({s.family for s in m.items})
    draws = np.mean(
        [np.mean([np.array(s.null_draws) for s in m.items if s.family == f], axis=0) for f in fams],
        axis=0,
    )
    observed = float(
        np.mean([np.mean([s.averaged for s in m.items if s.family == f]) for f in fams])
    )
    return draws, observed


def retest(
    models: Sequence[ModelScore], samples: Sequence[dict[str, list[float]]], path: Path
) -> None:
    """Within-proposition standard deviation across the k asks, per model."""
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(models) + 1.2))
    rng = np.random.default_rng(0)
    for i, (m, s) in enumerate(zip(models, samples, strict=True)):
        sds = np.array([np.std(v, ddof=1) for v in s.values() if len(v) > 1])
        y = len(models) - 1 - i
        ax.scatter(
            _cents(1) * sds,
            y + rng.uniform(-0.18, 0.18, len(sds)),
            s=9,
            color=BLUE,
            alpha=0.45,
            linewidths=0,
        )
        med = float(np.median(sds))
        ax.plot([_cents(med)] * 2, [y - 0.3, y + 0.3], color=INK, lw=2)
        ax.text(
            1.01,
            y,
            f"{m.retest.identical:.0%} identical",
            va="center",
            fontsize=8,
            color=MUTED,
            transform=ax.get_yaxis_transform(),
        )
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.model for m in reversed(models)])
    ax.set_xlabel("standard deviation of the 5 answers to one question (percentage points)")
    ax.set_title("Test-retest noise: one dot per question, bar at the median")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _save(fig, path)


def partition_sums(models: Sequence[ModelScore], labels: dict[str, str], path: Path) -> None:
    """Sum of the averaged answers over each partition, which should be exactly 1."""
    items = sorted({s.item for m in models for s in m.items if s.family == "partition"})
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(2.4 * len(models) + 2.2, 0.32 * len(items) + 1.3),
        sharey=True,
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models, strict=True):
        by = {s.item: s for s in m.items}
        for j, item in enumerate(items):
            s = by.get(item)
            if s is None:
                continue
            total = sum(s.means)
            y = len(items) - 1 - j
            ax.plot([1.0, total], [y, y], color=GRID, lw=2)
            ax.plot(total, y, "o", color=BLUE if total >= 1 else ORANGE, markersize=6)
        ax.axvline(1.0, color=INK, lw=1)
        ax.set_title(m.model.replace("claude-", ""), fontsize=10)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(items)))
    axes[0].set_yticklabels([labels.get(i, i) for i in reversed(items)], fontsize=8)
    axes[0].set_xlabel("sum of averaged answers (should be 1)")
    fig.suptitle(
        "Exhaustive, exclusive outcomes: do the probabilities add up?",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, path)


def ladder(
    models: Sequence[ModelScore],
    samples: Sequence[dict[str, list[float]]],
    props: Sequence[str],
    xs: Sequence[float],
    xlabel: str,
    title: str,
    path: Path,
) -> None:
    """One ladder: every answer as a faint dot, the mean as a line, per model."""
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(1)
    for ax, m, s in zip(axes, models, samples, strict=True):
        means = []
        for x, p in zip(xs, props, strict=True):
            v = s.get(p, [])
            jitter = rng.uniform(-0.01, 0.01, len(v)) * (max(xs) - min(xs))
            ax.scatter(np.full(len(v), x) + jitter, v, s=10, color=BLUE, alpha=0.35, linewidths=0)
            means.append(float(np.mean(v)) if v else np.nan)
        ax.plot(xs, means, "-o", color=ORANGE, lw=2, markersize=5)
        ax.set_title(m.model.replace("claude-", ""), fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("P(yes)")
    axes[len(axes) // 2].set_xlabel(xlabel)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def markets(
    model_values: dict[str, list[float]], market_values: dict[str, list[float]], path: Path
) -> None:
    """ECDF of per-item arbitrage: models asked once vs Polymarket mids."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    styles = [(BLUE, "-"), (ORANGE, "--"), (AQUA, ":")]
    groups = [*market_values.items(), *model_values.items()]
    for (label, vals), (colour, ls) in zip(groups, styles * 3, strict=False):
        v = np.sort(_cents(1) * np.array(vals))
        if len(v) == 0:
            continue
        ax.step(
            v,
            np.arange(1, len(v) + 1) / len(v),
            where="post",
            color=colour,
            ls=ls,
            lw=2,
            label=f"{label} (n={len(v)})",
        )
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("guaranteed loss per $1 on one partition or ladder (cents, log scale)")
    ax.set_ylabel("share at or below")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Polymarket mids vs models, partitions and ladders")
    _save(fig, path)


def bins_vs_ladder(
    models: Sequence[ModelScore],
    points: Sequence[list[tuple[float, float]]],
    path: Path,
) -> None:
    """Each range bin as asked (y) against the same bin implied by the ladder (x)."""
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m, pts in zip(axes, models, points, strict=True):
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        ax.plot([0, 1], [0, 1], color=INK, lw=1)
        above = ys > xs
        ax.scatter(xs[above], ys[above], s=18, color=BLUE, marker="o", linewidths=0)
        ax.scatter(xs[~above], ys[~above], s=18, color=ORANGE, marker="D", linewidths=0)
        ax.text(
            0.04,
            0.96,
            f"{above.mean():.0%} above the line",
            transform=ax.transAxes,
            fontsize=8,
            color=MUTED,
            va="top",
        )
        ax.set_xlim(-0.02, 0.8)
        ax.set_ylim(-0.02, 0.8)
        ax.set_aspect("equal")
        ax.set_title(m.model.replace("claude-", ""), fontsize=10)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("P(range) asked directly")
    axes[len(axes) // 2].set_xlabel("P(range) implied by the same model's ladder")
    fig.suptitle(
        "Range questions get more probability than the ladder leaves for them",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, path)
