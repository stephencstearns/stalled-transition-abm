"""
plots.py  —  Publication-quality figures from sweep results
=============================================================
Generates the three phase-diagram figures for the paper and
a single-run time-series figure for diagnostic/supplementary use.

All figures are saved as PDF (vector) and PNG (300 dpi raster).
Colour scheme is perceptually uniform and prints well in greyscale.

Usage
-----
    python plots.py                     # all figures from results/
    python plots.py --fig 1             # only Fig 1
    python plots.py --single            # single-run time series (demo)
    python plots.py --single --inst 0.7 --kin 0.002  # custom single run
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, Ellipse
from matplotlib.lines import Line2D

from model import Config, Simulation


# ── Empirical parameter ranges ────────────────────────────────────────────────
#
# These ranges are derived from the published literature and are overlaid on
# each phase diagram as a shaded ellipse with a central point (best estimate).
#
# KINSHIP DILUTION RATE (kin_dilution, per generation ≈ 20 yr):
#   Empirical within-group relatedness r for hunter-gatherers:
#     Bowles (2006) Science: r ≈ 0.07 (from FST ≈ 0.07 across archaeological samples)
#     Langergraber et al. (2011): r ≈ 0.01 (microsatellite data, contemporary foragers)
#     Lukas & Clutton-Brock (2011) type estimate; Birdsell model: r ≈ 0.02
#     Empirically defensible range: r_start ≈ 0.20–0.25 (Pleistocene bands) → r ≈ 0
#     over ~350 generations. Dilution rate = r_start / n_gens_to_zero.
#     Low end (slow): 0.25/600 ≈ 0.0004  (Langergraber-consistent, slow mixing)
#     High end (fast): 0.25/200 ≈ 0.0013  (Bowles-consistent, rapid state formation)
#     Best estimate: ~0.0007 (depletes r=0.25 to zero in ~357 gen ≈ 7,000 yr)
#
# INITIAL INSTITUTIONAL QUALITY (init_inst_q):
#   Proxied by cross-national governance indices (World Bank WGI rule-of-law,
#   Polity scores) and Ostrom's CPR institutional robustness studies.
#   Pre-Holocene / early Holocene societies: very weak formal institutions → 0.10–0.25
#   Modern WEIRD societies: strong impersonal institutions → 0.70–0.90
#   Intermediate (Neolithic–Bronze Age, current developing-world average): 0.35–0.60
#   Best estimate for "at transition": ~0.45 (weak but emergent institutions)
#
# BETWEEN-GROUP SELECTION WEIGHT (bgs):
#   Bowles (2009) Science: warfare mortality fraction d ≈ 0.12–0.14 across
#   archaeological/ethnographic samples; translates to effective bgs ≈ 0.15–0.35
#   in comparable models (Choi & Bowles 2007; Richerson et al. 2016 BBS).
#   Conservative (genetic BGS only): 0.10–0.20
#   Liberal (cultural BGS included): 0.25–0.55
#   Best estimate: ~0.28
#
# MISMATCH SEVERITY (mismatch):
#   Indexed by degree of evolutionary mismatch between Pleistocene social psychology
#   and modern institutional context. Schulz et al. (2019) Science show that
#   populations with longer Church exposure (lower kinship intensity) score higher
#   on impersonal prosociality; the gap between kin-psychology defaults and
#   institutional demands is largest in rapidly-urbanising societies.
#   Estimated range: 0.25–0.60; best estimate ~0.42
#
# SANCTION STRENGTH:
#   Fehr & Gächter (2002) Nature: punishment eliminates free-riding in one-shot
#   games → high effective sanction. Ostrom's design principles: graduated
#   sanctions in successful CPR institutions range from mild to strong.
#   Empirical governance effectiveness indices (WGI) suggest effective
#   sanction range 0.20–0.65 across modern states.
#   Best estimate: ~0.40

EMPIRICAL = {
    # Fig 1 axes: kin_dilution (x), inst_q (y)
    "fig1": {
        "x_center": 0.0007, "x_low": 0.0004, "x_high": 0.0013,
        "y_center": 0.45,   "y_low": 0.25,   "y_high": 0.65,
        "label": "Empirically plausible\nparameter range",
    },
    # Fig 2 axes: mismatch (x), bgs (y)
    "fig2": {
        "x_center": 0.42, "x_low": 0.25, "x_high": 0.60,
        "y_center": 0.28, "y_low": 0.15, "y_high": 0.45,
        "label": "Empirically plausible\nparameter range",
    },
    # Fig 3 axes: kin_dilution (x), sanction_strength (y)
    "fig3": {
        "x_center": 0.0007, "x_low": 0.0004, "x_high": 0.0013,
        "y_center": 0.40,   "y_low": 0.20,   "y_high": 0.65,
        "label": "Empirically plausible\nparameter range",
    },
}


def _draw_empirical_box(ax, emp, x_range, y_range):
    """
    Overlay an empirical parameter ellipse on a phase diagram panel.
    emp: dict with x_center, x_low, x_high, y_center, y_low, y_high.
    x_range, y_range: (min, max) of the full axis — used to clip the ellipse
    to the plotted area and to size it proportionally.
    """
    xc, yc   = emp["x_center"], emp["y_center"]
    xw = emp["x_high"] - emp["x_low"]   # full width
    yh = emp["y_high"] - emp["y_low"]   # full height

    ell = Ellipse(
        xy=(xc, yc), width=xw, height=yh,
        angle=0, linewidth=1.5,
        edgecolor="white", facecolor="white", alpha=0.25,
        transform=ax.transData, zorder=5,
    )
    ax.add_patch(ell)
    # Dashed border for contrast on any background colour
    ell2 = Ellipse(
        xy=(xc, yc), width=xw, height=yh,
        angle=0, linewidth=1.5, linestyle="--",
        edgecolor="black", facecolor="none", alpha=0.85,
        transform=ax.transData, zorder=6,
    )
    ax.add_patch(ell2)
    # Central best-estimate point
    ax.plot(xc, yc, marker="*", markersize=9, color="black",
            zorder=7, markeredgewidth=0.5, markeredgecolor="white")


# ── Style ─────────────────────────────────────────────────────────────────────

def _setup_style():
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Palatino", "Georgia", "Times New Roman", "serif"],
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.dpi":         150,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "lines.linewidth":    1.4,
        "pdf.fonttype":       42,    # embeds fonts properly
        "ps.fonttype":        42,
    })


# ── Regime colour map ─────────────────────────────────────────────────────────
#  0=Defection, 1=Pathological, 2=Stalled, 3=Complete
REGIME_CMAP  = mcolors.ListedColormap(
    ["#d73027",   # Defection     — red
     "#fc8d59",   # Pathological  — orange
     "#fee090",   # Stalled       — amber
     "#4575b4"]   # Complete      — blue
)
REGIME_NORM  = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
REGIME_LABELS = {0: "Defection", 1: "Pathological",
                  2: "Stalled / bistable", 3: "Transition complete"}

# Continuous colour map for cooperation rate
COOP_CMAP    = "RdYlBu"


# ── Helper: save figure ───────────────────────────────────────────────────────

def _save_fig(fig, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved → {out_dir / stem}.pdf / .png")


def _add_bistability_overlay(ax, bistability, extent, threshold=0.30):
    """
    Add a stipple/hatch overlay on cells where bistability > threshold,
    indicating that replicates were split across regimes at that grid point.
    This addresses Smaldino (2020) concern about visualising stochastic variance.
    threshold=0.30 means the dominant regime captured <70% of replicates.
    """
    # Create a masked array: True where bistability is high
    high_var = bistability > threshold
    # Overlay a transparent hatched image
    hatch_data = np.where(high_var, 1.0, np.nan)
    ax.imshow(hatch_data, origin="lower", extent=extent, aspect="auto",
              cmap=mcolors.ListedColormap(["none", "white"]),
              vmin=0, vmax=1, interpolation="nearest",
              alpha=0.0, zorder=3)
    # Draw contour at the threshold to show the boundary of the bistable zone
    from matplotlib import ticker as mticker
    y_res, x_res = bistability.shape
    x_coords = np.linspace(extent[0], extent[1], x_res)
    y_coords = np.linspace(extent[2], extent[3], y_res)
    try:
        ax.contour(x_coords, y_coords, bistability,
                   levels=[threshold], colors=["0.6"],
                   linewidths=[0.8], linestyles=[":"], zorder=4,
                   alpha=0.8)
    except Exception:
        pass  # contour may fail on very coarse grids

def fig1(npz_path: Path, out_dir: Path):
    _setup_style()
    data      = np.load(npz_path)
    inst_vals = data["axis_inst_q"]
    kin_vals  = data["axis_kin_dilution"]
    regime    = data["regime_int"]
    coop      = data["coop_final"]
    pathol    = data["pathol_final"]
    bistab    = data.get("bistability", np.zeros_like(regime))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.07, right=0.97, wspace=0.38)
    extent = [kin_vals[0], kin_vals[-1], inst_vals[0], inst_vals[-1]]

    def _fmt_x(ax):
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.tick_params(axis="x", labelsize=7)

    # Panel A — regime map
    ax = axes[0]
    ax.imshow(regime, origin="lower", extent=extent, aspect="auto",
              cmap=REGIME_CMAP, norm=REGIME_NORM, interpolation="nearest")
    ax.set_xlabel("Kinship dilution rate")
    ax.set_ylabel("Initial institutional quality")
    ax.set_title("A  Equilibrium regime")
    ax.contour(kin_vals, inst_vals, regime, levels=[1.5, 2.5],
               colors=["white"], linewidths=[0.7], linestyles=["--"])
    _draw_empirical_box(ax, EMPIRICAL["fig1"],
                        x_range=(kin_vals[0], kin_vals[-1]),
                        y_range=(inst_vals[0], inst_vals[-1]))
    _add_bistability_overlay(ax, bistab, extent)
    _fmt_x(ax)

    # Panel B — cooperation rate
    ax = axes[1]
    im2 = ax.imshow(coop, origin="lower", extent=extent, aspect="auto",
                    cmap=COOP_CMAP, vmin=0, vmax=1, interpolation="bilinear")
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label("Mean cooperation rate", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    ax.set_xlabel("Kinship dilution rate")
    ax.set_title("B  Cooperation rate at equilibrium")
    ax.contour(kin_vals, inst_vals, regime, levels=[1.5, 2.5],
               colors=["0.3"], linewidths=[0.6], linestyles=["--"])
    _draw_empirical_box(ax, EMPIRICAL["fig1"],
                        x_range=(kin_vals[0], kin_vals[-1]),
                        y_range=(inst_vals[0], inst_vals[-1]))
    _fmt_x(ax)

    # Panel C — pathology index
    ax = axes[2]
    im3 = ax.imshow(pathol, origin="lower", extent=extent, aspect="auto",
                    cmap="YlOrRd", vmin=0, vmax=1, interpolation="bilinear")
    cb3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    cb3.set_label("Pathology index", fontsize=8)
    cb3.ax.tick_params(labelsize=7)
    ax.set_xlabel("Kinship dilution rate")
    ax.set_title("C  Pathology index at equilibrium")
    ax.contour(kin_vals, inst_vals, regime, levels=[1.5, 2.5],
               colors=["0.3"], linewidths=[0.6], linestyles=["--"])
    _draw_empirical_box(ax, EMPIRICAL["fig1"],
                        x_range=(kin_vals[0], kin_vals[-1]),
                        y_range=(inst_vals[0], inst_vals[-1]))
    _fmt_x(ax)

    # Colour key below title, above panels
    legend_els = [Patch(facecolor=REGIME_CMAP(REGIME_NORM(i)), label=REGIME_LABELS[i])
                  for i in range(4)]
    legend_els += [
        Line2D([0], [0], linestyle="--", color="0.35", linewidth=1.2,
               label="Regime boundary (Panel A)"),
        Line2D([0], [0], linestyle=":", color="0.6", linewidth=0.9,
               label="Boundary uncertainty ≥ 30 % (Panel A)"),
        Line2D([0], [0], linestyle="--", color="black", linewidth=1.2,
               label="Empirical range"),
        Line2D([0], [0], marker="*", color="black", linestyle="none",
               markersize=9, label="Best estimate"),
    ]
    fig.legend(handles=legend_els, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), fontsize=7.5,
               framealpha=0.92, edgecolor="0.75", columnspacing=0.8,
               handlelength=1.4)

    _save_fig(fig, out_dir, "fig1_phase_inst_kin")
    plt.close(fig)


# ── Figure 2:  between-group selection × mismatch ────────────────────────────

def fig2(npz_path: Path, out_dir: Path):
    _setup_style()
    data      = np.load(npz_path)
    bgs_vals  = data["axis_bgs"]
    mis_vals  = data["axis_mismatch"]
    regime    = data["regime_int"]
    coop      = data["coop_final"]
    bistab    = data.get("bistability", np.zeros_like(regime))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.09, right=0.97, wspace=0.38)
    extent = [mis_vals[0], mis_vals[-1], bgs_vals[0], bgs_vals[-1]]

    ax = axes[0]
    ax.imshow(regime, origin="lower", extent=extent, aspect="auto",
              cmap=REGIME_CMAP, norm=REGIME_NORM, interpolation="nearest")
    ax.set_xlabel("Mismatch severity")
    ax.set_ylabel("Between-group selection weight")
    ax.set_title("A  Equilibrium regime")
    ax.contour(mis_vals, bgs_vals, regime, levels=[1.5, 2.5],
               colors=["white"], linewidths=[0.7], linestyles=["--"])
    _draw_empirical_box(ax, EMPIRICAL["fig2"],
                        x_range=(mis_vals[0], mis_vals[-1]),
                        y_range=(bgs_vals[0], bgs_vals[-1]))
    _add_bistability_overlay(ax, bistab, extent)

    ax = axes[1]
    im2 = ax.imshow(coop, origin="lower", extent=extent, aspect="auto",
                    cmap=COOP_CMAP, vmin=0, vmax=1, interpolation="bilinear")
    cb  = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Mean cooperation rate", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("Mismatch severity")
    ax.set_title("B  Cooperation rate at equilibrium")
    ax.contour(mis_vals, bgs_vals, regime, levels=[1.5, 2.5],
               colors=["0.3"], linewidths=[0.6], linestyles=["--"])
    _draw_empirical_box(ax, EMPIRICAL["fig2"],
                        x_range=(mis_vals[0], mis_vals[-1]),
                        y_range=(bgs_vals[0], bgs_vals[-1]))

    legend_els = [Patch(facecolor=REGIME_CMAP(REGIME_NORM(i)), label=REGIME_LABELS[i])
                  for i in range(4)]
    legend_els += [
        Line2D([0], [0], linestyle="--", color="0.35", linewidth=1.2,
               label="Regime boundary (Panel A)"),
        Line2D([0], [0], linestyle=":", color="0.6", linewidth=0.9,
               label="Boundary uncertainty ≥ 30 % (Panel A)"),
        Line2D([0], [0], linestyle="--", color="black", linewidth=1.2,
               label="Empirical range"),
        Line2D([0], [0], marker="*", color="black", linestyle="none",
               markersize=9, label="Best estimate"),
    ]
    fig.legend(handles=legend_els, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), fontsize=7.5,
               framealpha=0.92, edgecolor="0.75", columnspacing=0.8,
               handlelength=1.4)

    _save_fig(fig, out_dir, "fig2_phase_bgs_mismatch")
    plt.close(fig)


# ── Figure 3:  sanction strength × kinship dilution ──────────────────────────

def fig3(npz_path: Path, out_dir: Path):
    _setup_style()
    data      = np.load(npz_path)
    sanc_vals = data["axis_sanction_strength"]
    kin_vals  = data["axis_kin_dilution"]
    regime    = data["regime_int"]
    thresh    = data["thresh_final"]
    bistab    = data.get("bistability", np.zeros_like(regime))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.09, right=0.97, wspace=0.38)
    extent = [kin_vals[0], kin_vals[-1], sanc_vals[0], sanc_vals[-1]]

    def _fmt_x(ax):
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.tick_params(axis="x", labelsize=7)

    ax = axes[0]
    ax.imshow(regime, origin="lower", extent=extent, aspect="auto",
              cmap=REGIME_CMAP, norm=REGIME_NORM, interpolation="nearest")
    ax.set_xlabel("Kinship dilution rate")
    ax.set_ylabel("Sanction strength")
    ax.set_title("A  Equilibrium regime")
    _draw_empirical_box(ax, EMPIRICAL["fig3"],
                        x_range=(kin_vals[0], kin_vals[-1]),
                        y_range=(sanc_vals[0], sanc_vals[-1]))
    _add_bistability_overlay(ax, bistab, extent)
    _fmt_x(ax)

    ax = axes[1]
    im2 = ax.imshow(thresh, origin="lower", extent=extent, aspect="auto",
                    cmap="PuOr", vmin=0, vmax=1, interpolation="bilinear")
    cb  = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"Mean cooperation threshold ($\tau$)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("Kinship dilution rate")
    ax.set_title(r"B  Evolved cooperation threshold ($\tau$)")
    _draw_empirical_box(ax, EMPIRICAL["fig3"],
                        x_range=(kin_vals[0], kin_vals[-1]),
                        y_range=(sanc_vals[0], sanc_vals[-1]))
    _fmt_x(ax)

    legend_els = [Patch(facecolor=REGIME_CMAP(REGIME_NORM(i)), label=REGIME_LABELS[i])
                  for i in range(4)]
    legend_els += [
        Line2D([0], [0], linestyle="--", color="0.35", linewidth=1.2,
               label="Regime boundary (Panel A)"),
        Line2D([0], [0], linestyle=":", color="0.6", linewidth=0.9,
               label="Boundary uncertainty ≥ 30 % (Panel A)"),
        Line2D([0], [0], linestyle="--", color="black", linewidth=1.2,
               label="Empirical range"),
        Line2D([0], [0], marker="*", color="black", linestyle="none",
               markersize=9, label="Best estimate"),
    ]
    fig.legend(handles=legend_els, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), fontsize=7.5,
               framealpha=0.92, edgecolor="0.75", columnspacing=0.8,
               handlelength=1.4)

    _save_fig(fig, out_dir, "fig3_sanction_kin")
    plt.close(fig)


# ── Single-run time series (diagnostic / supplementary) ──────────────────────

def single_run(cfg_overrides: dict, out_dir: Path, label="single"):
    _setup_style()

    cfg = Config(**cfg_overrides)
    sim = Simulation(cfg, seed=42)
    res = sim.run()
    t   = np.arange(cfg.n_gens)

    # Compute regime shading
    def _regime_color(cr, iq, pa):
        if cr > 0.70 and iq > 0.65:   return "#c6dbef"   # blue: complete
        elif pa > 0.45:                 return "#fcbba1"   # red:  pathological
        elif cr > 0.35:                 return "#fdd49e"   # amber: stalled
        else:                           return "#f0f0f0"   # grey: defection

    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    window    = 10
    cr_smooth = np.convolve(res.coop_rate,    np.ones(window)/window, "same")
    iq_smooth = np.convolve(res.mean_inst_q,  np.ones(window)/window, "same")
    pa_smooth = np.convolve(res.pathology,    np.ones(window)/window, "same")

    # Panel 1: cooperation threshold
    axes[0].fill_between(t, res.mean_thresh - res.sd_thresh,
                              res.mean_thresh + res.sd_thresh,
                         alpha=0.25, color="#4575b4", label="±1 SD")
    axes[0].plot(t, res.mean_thresh, color="#4575b4", label="Mean τ")
    axes[0].set_ylabel("Cooperation threshold τ")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].axhline(0.5, color="0.7", lw=0.6, ls="--")

    # Panel 2: cooperation rate & institutional quality
    axes[1].plot(t, res.coop_rate,   color="#4dac26", alpha=0.4, lw=0.8)
    axes[1].plot(t, cr_smooth,        color="#4dac26", label="Cooperation rate")
    axes[1].plot(t, res.mean_inst_q, color="#d6604d", alpha=0.4, lw=0.8)
    axes[1].plot(t, iq_smooth,        color="#d6604d", label="Institutional quality")
    axes[1].set_ylabel("Rate / quality")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].axhline(0.5, color="0.7", lw=0.6, ls="--")

    # Panel 3: pathology index & relatedness
    axes[2].plot(t, res.pathology,   color="#b2182b", alpha=0.4, lw=0.8)
    axes[2].plot(t, pa_smooth,        color="#b2182b", label="Pathology index")
    axes[2].plot(t, res.relatedness,  color="#2166ac", ls="--",
                 lw=1.1, label="Kinship (relatedness)")
    axes[2].set_ylabel("Index")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Generation")
    axes[2].legend(fontsize=8, loc="upper right")

    regime = res.final_regime()
    stats  = res.summary_stats()
    fig.suptitle(
        f"Single-run time series  —  Regime: {regime}\n"
        f"Q₀={cfg_overrides.get('init_inst_q', cfg.init_inst_q):.2f}  "
        f"kin_dilution={cfg_overrides.get('kin_dilution', cfg.kin_dilution):.4f}  "
        f"bgs={cfg_overrides.get('bgs', cfg.bgs):.2f}  "
        f"mismatch={cfg_overrides.get('mismatch', cfg.mismatch):.2f}\n"
        f"Final: coop={stats['coop_final']:.3f}  "
        f"inst_q={stats['inst_q_final']:.3f}  "
        f"pathol={stats['pathol_final']:.3f}",
        fontsize=9
    )
    fig.tight_layout()
    _save_fig(fig, out_dir, f"single_run_{label}")
    plt.close(fig)
    print(f"  Regime: {regime}")
    for k, v in stats.items():
        if k != "regime":
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stalled Transition ABM plots")
    parser.add_argument("--fig",    type=int, default=0)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--inst",   type=float, default=0.50)
    parser.add_argument("--kin",    type=float, default=0.003)
    parser.add_argument("--bgs",    type=float, default=0.30)
    parser.add_argument("--mism",   type=float, default=0.40)
    parser.add_argument("--indir",  type=str,   default="results")
    parser.add_argument("--outdir", type=str,   default="figures")
    args    = parser.parse_args()

    in_dir  = Path(args.indir)
    out_dir = Path(args.outdir)

    if args.single:
        overrides = dict(init_inst_q=args.inst, kin_dilution=args.kin,
                         bgs=args.bgs, mismatch=args.mism, n_gens=500)
        single_run(overrides, out_dir,
                   label=f"inst{args.inst:.2f}_kin{args.kin:.4f}")
# ── Figure 4:  cultural weight (T_P) × kinship dilution  [ETII Extension 1] ──

def fig4(npz_path: Path, out_dir: Path):
    """
    Extension 1 phase diagram: cultural determination of cooperation
    threshold (T_P, Waring & Wood 2025) × kinship dilution rate.
    Tests whether cultural takeover of tau resolves the stall.
    """
    _setup_style()
    data    = np.load(npz_path)
    cw_vals = data["axis_cultural_weight"]
    kin_vals= data["axis_kin_dilution"]
    regime  = data["regime_int"]
    coop    = data["coop_final"]
    bistab  = data.get("bistability", np.zeros_like(regime))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.09, right=0.97, wspace=0.38)
    extent = [kin_vals[0], kin_vals[-1], cw_vals[0], cw_vals[-1]]

    def _fmt_x(ax):
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.tick_params(axis="x", labelsize=7)

    # Panel A — regime map
    ax = axes[0]
    ax.imshow(regime, origin="lower", extent=extent, aspect="auto",
              cmap=REGIME_CMAP, norm=REGIME_NORM, interpolation="nearest")
    ax.set_xlabel("Kinship dilution rate")
    ax.set_ylabel(r"Cultural weight of $\tau$ ($T_P$)")
    ax.set_title(r"A  Equilibrium regime")
    ax.contour(kin_vals, cw_vals, regime, levels=[1.5, 2.5],
               colors=["white"], linewidths=[0.7], linestyles=["--"])
    # Empirical ellipse: T_P estimated at ~0.35-0.65 for contemporary
    # modernising societies (Waring & Wood 2025, Stage II-III boundary)
    _draw_empirical_box(ax,
        {"x_center": 0.0007, "x_low": 0.0004, "x_high": 0.0013,
         "y_center": 0.50,   "y_low": 0.30,   "y_high": 0.70},
        x_range=(kin_vals[0], kin_vals[-1]),
        y_range=(cw_vals[0], cw_vals[-1]))
    _add_bistability_overlay(ax, bistab, extent)
    _fmt_x(ax)

    # Panel B — cooperation rate
    ax = axes[1]
    im2 = ax.imshow(coop, origin="lower", extent=extent, aspect="auto",
                    cmap=COOP_CMAP, vmin=0, vmax=1, interpolation="bilinear")
    cb  = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Mean cooperation rate", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("Kinship dilution rate")
    ax.set_title("B  Cooperation rate at equilibrium")
    ax.contour(kin_vals, cw_vals, regime, levels=[1.5, 2.5],
               colors=["0.3"], linewidths=[0.6], linestyles=["--"])
    _draw_empirical_box(ax,
        {"x_center": 0.0007, "x_low": 0.0004, "x_high": 0.0013,
         "y_center": 0.50,   "y_low": 0.30,   "y_high": 0.70},
        x_range=(kin_vals[0], kin_vals[-1]),
        y_range=(cw_vals[0], cw_vals[-1]))
    _fmt_x(ax)

    # Colour key
    legend_els = [Patch(facecolor=REGIME_CMAP(REGIME_NORM(i)), label=REGIME_LABELS[i])
                  for i in range(4)]
    legend_els += [
        Line2D([0], [0], linestyle="--", color="0.35", linewidth=1.2,
               label="Regime boundary (Panel A)"),
        Line2D([0], [0], linestyle=":", color="0.6", linewidth=0.9,
               label="Boundary uncertainty ≥ 30 % (Panel A)"),
        Line2D([0], [0], linestyle="--", color="black", linewidth=1.2,
               label="Empirical range"),
        Line2D([0], [0], marker="*", color="black", linestyle="none",
               markersize=9, label="Best estimate"),
    ]
    fig.legend(handles=legend_els, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), fontsize=7.5,
               framealpha=0.92, edgecolor="0.75", columnspacing=0.8,
               handlelength=1.4)

    _save_fig(fig, out_dir, "fig4_cultural_weight_kin")
    plt.close(fig)


# ── Figure 5:  rising cultural BGS × institutional quality  [ETII Ext 2] ────

def fig5(npz_path: Path, out_dir: Path):
    """
    Extension 2 phase diagram: rate of rise of cultural between-group
    selection × initial institutional quality.
    Tests whether a strengthening cultural BGS (Waring & Wood 2025)
    is sufficient to complete the transition, and whether it depends
    on the institutional baseline.
    """
    _setup_style()
    data       = np.load(npz_path)
    rise_vals  = data["axis_bgs_cultural_rise"]
    inst_vals  = data["axis_inst_q"]
    regime     = data["regime_int"]
    coop       = data["coop_final"]
    bgs_final  = data["bgs_eff_final"]
    bistab     = data.get("bistability", np.zeros_like(regime))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.07, right=0.97, wspace=0.38)
    extent = [rise_vals[0], rise_vals[-1], inst_vals[0], inst_vals[-1]]

    # Panel A — regime map
    ax = axes[0]
    ax.imshow(regime, origin="lower", extent=extent, aspect="auto",
              cmap=REGIME_CMAP, norm=REGIME_NORM, interpolation="nearest")
    ax.set_xlabel("Cultural BGS rise rate (per generation)")
    ax.set_ylabel("Initial institutional quality")
    ax.set_title("A  Equilibrium regime")
    ax.contour(rise_vals, inst_vals, regime, levels=[1.5, 2.5],
               colors=["white"], linewidths=[0.7], linestyles=["--"])
    # Empirical range: rise rate estimated from Waring & Wood trajectory;
    # institutional quality as in Fig 1
    _draw_empirical_box(ax,
        {"x_center": 0.0007, "x_low": 0.0002, "x_high": 0.0013,
         "y_center": 0.45,   "y_low": 0.25,   "y_high": 0.65},
        x_range=(rise_vals[0], rise_vals[-1]),
        y_range=(inst_vals[0], inst_vals[-1]))
    _add_bistability_overlay(ax, bistab, extent)

    # Panel B — cooperation rate
    ax = axes[1]
    im2 = ax.imshow(coop, origin="lower", extent=extent, aspect="auto",
                    cmap=COOP_CMAP, vmin=0, vmax=1, interpolation="bilinear")
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label("Mean cooperation rate", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    ax.set_xlabel("Cultural BGS rise rate (per generation)")
    ax.set_title("B  Cooperation rate at equilibrium")
    ax.contour(rise_vals, inst_vals, regime, levels=[1.5, 2.5],
               colors=["0.3"], linewidths=[0.6], linestyles=["--"])
    _draw_empirical_box(ax,
        {"x_center": 0.0007, "x_low": 0.0002, "x_high": 0.0013,
         "y_center": 0.45,   "y_low": 0.25,   "y_high": 0.65},
        x_range=(rise_vals[0], rise_vals[-1]),
        y_range=(inst_vals[0], inst_vals[-1]))

    # Panel C — effective BGS at equilibrium
    ax = axes[2]
    im3 = ax.imshow(bgs_final, origin="lower", extent=extent, aspect="auto",
                    cmap="plasma", vmin=0, vmax=1, interpolation="bilinear")
    cb3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    cb3.set_label("Effective BGS at equilibrium", fontsize=8)
    cb3.ax.tick_params(labelsize=7)
    ax.set_xlabel("Cultural BGS rise rate (per generation)")
    ax.set_title("C  Effective BGS reached")
    ax.contour(rise_vals, inst_vals, regime, levels=[1.5, 2.5],
               colors=["white"], linewidths=[0.6], linestyles=["--"])
    _draw_empirical_box(ax,
        {"x_center": 0.0007, "x_low": 0.0002, "x_high": 0.0013,
         "y_center": 0.45,   "y_low": 0.25,   "y_high": 0.65},
        x_range=(rise_vals[0], rise_vals[-1]),
        y_range=(inst_vals[0], inst_vals[-1]))

    # Colour key
    legend_els = [Patch(facecolor=REGIME_CMAP(REGIME_NORM(i)), label=REGIME_LABELS[i])
                  for i in range(4)]
    legend_els += [
        Line2D([0], [0], linestyle="--", color="0.35", linewidth=1.2,
               label="Regime boundary (Panel A)"),
        Line2D([0], [0], linestyle=":", color="0.6", linewidth=0.9,
               label="Boundary uncertainty ≥ 30 % (Panel A)"),
        Line2D([0], [0], linestyle="--", color="black", linewidth=1.2,
               label="Empirical range"),
        Line2D([0], [0], marker="*", color="black", linestyle="none",
               markersize=9, label="Best estimate"),
    ]
    fig.legend(handles=legend_els, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), fontsize=7.5,
               framealpha=0.92, edgecolor="0.75", columnspacing=0.8,
               handlelength=1.4)

    _save_fig(fig, out_dir, "fig5_bgs_rise_inst")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Stalled Transition — plot figures")
    parser.add_argument("--fig",    type=int, default=0,
                        help="Which figure to plot (1-5); 0 = all")
    parser.add_argument("--indir",  type=str, default="results",
                        help="Directory containing .npz sweep results")
    parser.add_argument("--outdir", type=str, default="figures",
                        help="Output directory for figures")
    args   = parser.parse_args()
    in_dir  = Path(args.indir)
    out_dir = Path(args.outdir)

    if not in_dir.exists():
        print(f"Results directory '{in_dir}' not found. Run sweep.py first.")
        return

    figs_to_plot = [args.fig] if args.fig else [1, 2, 3, 4, 5, 6, 7, 8]
    dispatch = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5,
                7: fig7_ext3, 8: fig8_tp_slice}
    for fn in figs_to_plot:
        if fn == 6:
            npz = in_dir / "fig6_sensitivity.npz"
            if not npz.exists():
                print(f"  Warning: {npz} not found — run sweep.py --fig 6 first")
                continue
            print(f"Plotting Fig 6 (sensitivity) from {npz}")
            fig_sensitivity(npz, out_dir)
            continue
        if fn not in dispatch:
            print(f"  Warning: no plot function for fig {fn}")
            continue
        npz = in_dir / f"fig{fn}_grids.npz" if fn <= 5 else (
              in_dir / "fig7_grids.npz" if fn == 7 else
              in_dir / "fig8_tp_slice.npz")
        if not npz.exists():
            print(f"  Warning: {npz} not found — run sweep.py --fig {fn} first")
            continue
        print(f"Plotting Fig {fn} from {npz}")
        dispatch[fn](npz, out_dir)

    print("Plotting complete.")


# ── Figure 7: Extension 3 — endogenous TP × initial institutional quality ─────

def fig7_ext3(npz_path: Path, out_dir: Path):
    """
    Extension 3 phase diagram: TP evolution rate × Q₀.
    Addresses Henrich (reviewer) concern that TP should be emergent.
    Directly comparable to Figure 5 (rising BGS × Q₀).
    """
    _setup_style()
    data      = np.load(npz_path)
    tp_vals   = data["axis_tp_adapt"]
    inst_vals = data["axis_inst_q"]
    regime    = data["regime_int"]
    coop      = data["coop_final"]
    tp_final  = data.get("tp_eff_final", np.zeros_like(regime))
    bistab    = data.get("bistability", np.zeros_like(regime))
    extent    = [tp_vals[0], tp_vals[-1], inst_vals[0], inst_vals[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.07, right=0.97, wspace=0.38)

    # Panel A — regime map
    ax = axes[0]
    ax.imshow(regime, origin="lower", extent=extent, aspect="auto",
              cmap=REGIME_CMAP, norm=REGIME_NORM, interpolation="nearest")
    ax.contour(tp_vals, inst_vals, regime, levels=[1.5, 2.5],
               colors=["white"], linewidths=[0.7], linestyles=["--"])
    _add_bistability_overlay(ax, bistab, extent)
    ax.set_xlabel("TP adaptation rate (per generation)")
    ax.set_ylabel("Initial institutional quality Q₀")
    ax.set_title("A  Equilibrium regime")

    # Panel B — cooperation rate
    ax = axes[1]
    im2 = ax.imshow(coop, origin="lower", extent=extent, aspect="auto",
                    cmap=COOP_CMAP, vmin=0, vmax=1, interpolation="bilinear")
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label("Mean cooperation rate", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    ax.set_xlabel("TP adaptation rate (per generation)")
    ax.set_title("B  Cooperation rate at equilibrium")
    ax.contour(tp_vals, inst_vals, regime, levels=[1.5, 2.5],
               colors=["0.3"], linewidths=[0.6], linestyles=["--"])

    # Panel C — evolved TP at equilibrium
    ax = axes[2]
    im3 = ax.imshow(tp_final, origin="lower", extent=extent, aspect="auto",
                    cmap="PuOr", vmin=0, vmax=1, interpolation="bilinear")
    cb3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    cb3.set_label("Effective TP at equilibrium", fontsize=8)
    cb3.ax.tick_params(labelsize=7)
    ax.set_xlabel("TP adaptation rate (per generation)")
    ax.set_title("C  Evolved TP at equilibrium")
    ax.contour(tp_vals, inst_vals, regime, levels=[1.5, 2.5],
               colors=["0.3"], linewidths=[0.6], linestyles=["--"])

    legend_els = [Patch(facecolor=REGIME_CMAP(REGIME_NORM(i)), label=REGIME_LABELS[i])
                  for i in range(4)]
    legend_els += [
        Line2D([0], [0], linestyle="--", color="0.4", linewidth=1.2,
               label="Regime boundary (Panel A)"),
        Line2D([0], [0], linestyle=":", color="0.6", linewidth=0.9,
               label="Boundary uncertainty ≥ 30 % (Panel A)"),
    ]
    fig.legend(handles=legend_els, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), fontsize=7.5,
               framealpha=0.92, edgecolor="0.75", columnspacing=0.8,
               handlelength=1.4)
    fig.suptitle("Figure 7. ETII Extension 3: endogenous TP evolution × initial institutional quality",
                 fontsize=9, y=0.995, va="top")

    _save_fig(fig, out_dir, "fig7_ext3_tp_evolving")
    plt.close(fig)


# ── Figure 8: Bowles 1D slice — TP from 0→1 at Q₀ = 0.45 ────────────────────

def fig8_tp_slice(npz_path: Path, out_dir: Path):
    """
    1D slice through Figure 4 at Q₀ = 0.45 (best-estimate institutional
    quality), showing cooperation rate and regime as TP increases from 0 to 1.
    Requested by Bowles (reviewer) as a supplement to Figure 4.
    Includes ±1 SD band across the 20 replicates.
    """
    _setup_style()
    data    = np.load(npz_path)
    tp_vals = data["axis_tp"]
    coop    = data["coop_final"]
    pathol  = data["pathol_final"]
    regime  = data["regime_int"]
    # SD bands if available
    coop_sd   = data["coop_sd"]   if "coop_sd"   in data.files else np.zeros_like(coop)
    pathol_sd = data["pathol_sd"] if "pathol_sd" in data.files else np.zeros_like(pathol)

    regime_colors = [REGIME_CMAP(REGIME_NORM(r)) for r in regime]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.0))
    # Increase top margin: suptitle at top, no competing figure-level legend there
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.10, right=0.97, wspace=0.38)

    # Panel A — cooperation rate with SD band
    ax = axes[0]
    ax.fill_between(tp_vals, coop - coop_sd, coop + coop_sd,
                    color="0.8", alpha=0.5, zorder=2, label="\u00b11 SD across replicates")
    ax.scatter(tp_vals, coop, c=regime_colors, s=55, zorder=5,
               edgecolors="white", linewidths=0.5)
    ax.plot(tp_vals, coop, color="0.6", lw=0.8, zorder=3)
    ax.set_xlabel("T\u209a (cultural phenotype index)")
    ax.set_ylabel("Mean cooperation rate c\u0305")
    ax.set_title("A  Cooperation rate vs T\u209a at Q\u2080 = 0.45")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 1.05)
    ax.axvspan(0.30, 0.70, color="0.9", alpha=0.4, zorder=1)
    ax.text(0.50, 0.06, "Waring & Wood\nStage II\u2013III\n(indicative)",
            ha="center", fontsize=7, color="0.4")
    # SD legend inside Panel A (upper left)
    ax.legend(fontsize=7, loc="upper left")

    # Colour key inside Panel A (lower right) — avoids figure-level collision with suptitle
    legend_els = [Patch(facecolor=REGIME_CMAP(REGIME_NORM(i)), label=REGIME_LABELS[i])
                  for i in range(4)]
    ax.legend(handles=legend_els, fontsize=6.5, loc="center right",
              title="Regime", title_fontsize=7,
              framealpha=0.92, edgecolor="0.75", handlelength=1.2,
              bbox_to_anchor=(1.0, 0.55))

    # Panel B — pathology index with SD band
    ax = axes[1]
    ax.fill_between(tp_vals, pathol - pathol_sd, pathol + pathol_sd,
                    color="0.8", alpha=0.5, zorder=2,
                    label="\u00b11 SD across replicates")
    ax.scatter(tp_vals, pathol, c=regime_colors, s=55, zorder=5,
               edgecolors="white", linewidths=0.5)
    ax.plot(tp_vals, pathol, color="0.6", lw=0.8, zorder=3)
    ax.axhline(0.45, color="0.5", lw=0.8, ls="--", alpha=0.8,
               label="PATHOLOGICAL threshold (P = 0.45)")
    ax.set_xlabel("T\u209a (cultural phenotype index)")
    ax.set_ylabel("Pathology index P")
    ax.set_title("B  Pathology index vs T\u209a at Q\u2080 = 0.45")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 0.80)
    ax.legend(fontsize=7, loc="upper right")
    ax.axvspan(0.30, 0.70, color="0.9", alpha=0.4, zorder=1)

    # Suptitle only — no competing figure-level colour legend
    fig.suptitle(
        "Figure 8 (Supplementary). Cooperation rate and pathology index vs T\u209a "
        "at Q\u2080\u2009=\u20090.45\n"
        "Grey band\u2009=\u2009\u00b11 SD across 20 replicates; "
        "shaded vertical band\u2009=\u2009Waring\u00a0&\u00a0Wood Stage\u00a0II\u2013III "
        "range (indicative); coloured points\u00a0=\u00a0mean regime (colour key in Panel\u00a0A).",
        fontsize=8.5, y=0.995, va="top"
    )

    _save_fig(fig, out_dir, "fig8_tp_slice")
    plt.close(fig)


# ── Figure 6: Sensitivity analysis ───────────────────────────────────────────

def fig_sensitivity(npz_path: Path, out_dir: Path):
    """
    3×3 grid of Fig-1-style phase diagrams showing robustness of the
    PATHOLOGICAL/STALLED regime structure to variation in N, G, and
    institutional lag. Addresses Smaldino (2020) concern about unmotivated
    structural parameter choices.

    Rows: N = 100, 300, 500
    Columns: G = 10 (lag 10 gen), G = 20 (lag 25 gen), G = 40 (lag 50 gen)
    """
    _setup_style()
    data   = np.load(npz_path, allow_pickle=True)
    labels = list(data["labels"])
    inst_vals = data["axis_inst_q"]
    kin_vals  = data["axis_kin_dilution"]
    extent    = [kin_vals[0], kin_vals[-1], inst_vals[0], inst_vals[-1]]

    # Decode labels: "N=100,G=10,lag=10" → row/col
    row_labels = ["N = 100", "N = 300", "N = 500"]
    col_labels = ["G = 10\nlag = 10 gen", "G = 20\nlag = 25 gen", "G = 40\nlag = 50 gen"]

    # Build 3×3 grid of regime arrays
    regime_grid = {}
    for lbl in labels:
        key = lbl.replace(",", "_").replace("=", "") + "_regime"
        if key in data:
            regime_grid[lbl] = data[key]

    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.09, right=0.97,
                        hspace=0.45, wspace=0.30)

    row_order = [("N=100,G=10,lag=10", "N=100,G=20,lag=25", "N=100,G=40,lag=50"),
                 ("N=300,G=10,lag=10", "N=300,G=20,lag=25", "N=300,G=40,lag=50"),
                 ("N=500,G=10,lag=10", "N=500,G=20,lag=25", "N=500,G=40,lag=50")]

    for row, row_labels_set in enumerate(row_order):
        for col, lbl in enumerate(row_labels_set):
            ax = axes[row, col]
            regime = regime_grid.get(lbl)
            if regime is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                continue

            ax.imshow(regime, origin="lower", extent=extent, aspect="auto",
                      cmap=REGIME_CMAP, norm=REGIME_NORM, interpolation="nearest")

            # Empirical star only
            ax.plot(0.0007, 0.45, marker="*", markersize=7, color="black",
                    zorder=7, markeredgewidth=0.4, markeredgecolor="white")

            ax.xaxis.set_major_formatter(
                mpl.ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.tick_params(labelsize=6)

            if row == 2:
                ax.set_xlabel("Kin. dilution rate", fontsize=7)
            if col == 0:
                ax.set_ylabel("Init. inst. quality", fontsize=7)

            # Column header on top row
            if row == 0:
                ax.set_title(col_labels[col], fontsize=8, pad=4)
            # Row label on left column
            if col == 0:
                ax.annotate(row_labels[row], xy=(-0.32, 0.5),
                            xycoords="axes fraction", fontsize=8,
                            ha="center", va="center", rotation=90,
                            fontweight="bold")

    # Shared legend
    legend_els = [Patch(facecolor=REGIME_CMAP(REGIME_NORM(i)),
                         label=REGIME_LABELS[i]) for i in range(4)]
    legend_els += [
        Line2D([0], [0], marker="*", color="black", linestyle="none",
               markersize=7, label="Best estimate (default: N=300, G=20, lag=25 gen)"),
    ]
    fig.legend(handles=legend_els, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), fontsize=7.5,
               framealpha=0.92, edgecolor="0.75", columnspacing=0.8,
               handlelength=1.4)

    fig.suptitle(
        "Figure 6. Sensitivity analysis: qualitative regime structure is robust to variation in N, G, and institutional lag",
        fontsize=10, y=0.995, va="top")

    _save_fig(fig, out_dir, "fig6_sensitivity")
    plt.close(fig)


if __name__ == "__main__":
    main()
