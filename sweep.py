"""
sweep.py  —  Parameter sweep over the phase space
===================================================
Generates the two-dimensional phase diagrams for the paper:
  • Fig 1:  institutional quality  ×  kinship dilution rate
  • Fig 2:  between-group selection  ×  mismatch severity
  • Fig 3:  sanction strength  ×  kinship dilution  (institution-kinship substitution)

Each grid point runs N_REPS independent replicates.  Replicates
are distributed across all available CPU performance cores via
multiprocessing.Pool.

Apple Silicon note
------------------
On M-series iMacs, multiprocessing uses 'spawn' (macOS default),
which requires the worker function to be importable at module
level — it is, since _run_one is a top-level function here.
Optimal pool size is cpu_count() - 1 to leave one core free for
the OS and rendering.  For M3/M4 iMac this is typically 9 or 11.

Usage
-----
    python sweep.py                    # runs all three sweeps
    python sweep.py --fig 1            # runs only Fig 1
    python sweep.py --fig 1 --fast     # coarse grid, 3 reps — smoke test
"""

import argparse
import os
import time
import multiprocessing as mp
from itertools import product
from pathlib import Path

import numpy as np

from model import Config, Simulation


# ── Worker function (must be top-level for spawn) ─────────────────────────────

def _run_one(args):
    """Run a single replicate and return summary stats + seed."""
    cfg_kwargs, seed = args
    cfg = Config(**cfg_kwargs)
    sim = Simulation(cfg, seed=seed)
    res = sim.run()
    stats = res.summary_stats()
    stats["seed"] = seed
    # Encode regime as integer for easy numpy storage
    regime_map = {"COMPLETE": 3, "STALLED": 2, "PATHOLOGICAL": 1, "DEFECTION": 0}
    stats["regime_int"] = regime_map[stats["regime"]]
    return stats


def _run_grid(grid_points, n_reps, n_workers, desc="sweep"):
    """
    Distribute grid_points × n_reps jobs across n_workers processes.
    Returns a list of dicts (one per job).
    """
    jobs = [
        (kwargs, seed)
        for kwargs, seed in (
            (kwargs, base_seed + rep)
            for (kwargs, base_seed) in (
                (gp, i * n_reps) for i, gp in enumerate(grid_points)
            )
            for rep in range(n_reps)
        )
    ]
    n_jobs = len(jobs)
    results = []

    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, jobs,
                                                        chunksize=4)):
            results.append(result)
            if (i + 1) % max(1, n_jobs // 20) == 0:
                elapsed = time.time() - t0
                pct     = (i + 1) / n_jobs * 100
                eta     = elapsed / (i + 1) * (n_jobs - i - 1)
                print(f"  {desc}: {i+1}/{n_jobs}  ({pct:.0f}%)  "
                      f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s")
    return results


# ── Grid definitions ──────────────────────────────────────────────────────────

def _grid_fig1(fast=False):
    """Institutional quality × kinship dilution — the primary phase diagram."""
    res       = 12 if fast else 30
    inst_vals = np.linspace(0.10, 0.90, res)
    kin_vals  = np.linspace(0.0001, 0.0014, res)

    base = dict(n_agents=300, n_groups=20, n_gens=600,
                bgs=0.28, mismatch=0.42, mut_rate=0.05,
                sanction_strength=0.40)

    points = [
        {**base, "init_inst_q": float(iq), "kin_dilution": float(kd)}
        for iq, kd in product(inst_vals, kin_vals)
    ]
    axes = {"inst_q": inst_vals, "kin_dilution": kin_vals}
    return points, axes, res


def _grid_fig2(fast=False):
    """Between-group selection × mismatch severity."""
    res       = 12 if fast else 30
    bgs_vals  = np.linspace(0.05, 0.70, res)
    mis_vals  = np.linspace(0.10, 0.75, res)

    base = dict(n_agents=300, n_groups=20, n_gens=600,
                init_inst_q=0.50, kin_dilution=0.0007, mut_rate=0.05,
                sanction_strength=0.40)

    points = [
        {**base, "bgs": float(b), "mismatch": float(m)}
        for b, m in product(bgs_vals, mis_vals)
    ]
    axes = {"bgs": bgs_vals, "mismatch": mis_vals}
    return points, axes, res


def _grid_fig4(fast=False):
    """
    Extension 1: Cultural weight (T_P) × kinship dilution.
    Directly tests the Waring & Wood (2025) claim that cultural takeover
    of the cooperation threshold resolves the stall.
    Same axes as Fig 1 but cultural_weight replaces inst_q on the y-axis.
    """
    res        = 12 if fast else 24
    cw_vals    = np.linspace(0.0, 1.0, res)   # 0 = genetic, 1 = fully cultural
    kin_vals   = np.linspace(0.0001, 0.0014, res)

    base = dict(n_agents=300, n_groups=20, n_gens=600,
                init_inst_q=0.50, bgs=0.28, mismatch=0.42,
                mut_rate=0.05, sanction_strength=0.40,
                bgs_cultural_rise=0.0)

    points = [
        {**base, "cultural_weight": float(cw), "kin_dilution": float(kd)}
        for cw, kd in product(cw_vals, kin_vals)
    ]
    axes = {"cultural_weight": cw_vals, "kin_dilution": kin_vals}
    return points, axes, res


def _grid_fig5(fast=False):
    """
    Extension 2: Rising cultural BGS rate × initial institutional quality.
    Tests whether a strengthening cultural group selection force is sufficient
    to push the system out of the STALLED zone, and whether it depends on
    the institutional baseline.
    """
    res        = 12 if fast else 24
    rise_vals  = np.linspace(0.0, 0.002, res)  # 0 = constant, 0.002 = doubles over 600 gens
    inst_vals  = np.linspace(0.10, 0.90, res)

    base = dict(n_agents=300, n_groups=20, n_gens=600,
                bgs=0.28, mismatch=0.42, mut_rate=0.05,
                sanction_strength=0.40, kin_dilution=0.0007,
                cultural_weight=0.0)

    points = [
        {**base, "bgs_cultural_rise": float(r), "init_inst_q": float(iq)}
        for r, iq in product(rise_vals, inst_vals)
    ]
    axes = {"bgs_cultural_rise": rise_vals, "inst_q": inst_vals}
    return points, axes, res


def _grid_fig3(fast=False):
    """Sanction strength × kinship dilution — institution-kinship substitution."""
    res         = 12 if fast else 24
    sanc_vals   = np.linspace(0.05, 0.80, res)
    kin_vals    = np.linspace(0.0001, 0.0014, res)

    base = dict(n_agents=300, n_groups=20, n_gens=600,
                init_inst_q=0.55, bgs=0.28, mismatch=0.42, mut_rate=0.05)

    points = [
        {**base, "sanction_strength": float(s), "kin_dilution": float(k)}
        for s, k in product(sanc_vals, kin_vals)
    ]
    axes = {"sanction_strength": sanc_vals, "kin_dilution": kin_vals}
    return points, axes, res


# ── Aggregate results onto grid ───────────────────────────────────────────────

def _grid_fig7(fast=False):
    """
    Extension 3: Endogenous TP evolution rate × initial institutional quality.
    Addresses Henrich (reviewer) concern that TP should emerge from the model
    rather than being fixed.  TP starts at tp_init=0 and rises at rate tp_adapt
    toward mean(Q_g) × mean(c̄_g), the same feedback that drives Q.
    Y-axis: Q₀ (initial institutional quality)
    X-axis: tp_adapt (rate of TP evolution, 0 = fixed at 0)
    Same axes layout as Figure 5 for direct comparison.
    """
    res        = 12 if fast else 24
    tp_adapt_vals = np.linspace(0.0, 0.08, res)   # 0 = no evolution; 0.08 ≈ 12-gen lag
    inst_vals     = np.linspace(0.10, 0.90, res)

    base = dict(n_agents=300, n_groups=20, n_gens=600,
                bgs=0.28, mismatch=0.42, mut_rate=0.05,
                sanction_strength=0.40, kin_dilution=0.0007,
                cultural_weight=0.0, bgs_cultural_rise=0.0,
                tp_init=0.0)

    points = [
        {**base, "tp_adapt": float(ta), "init_inst_q": float(iq)}
        for ta, iq in product(tp_adapt_vals, inst_vals)
    ]
    axes = {"tp_adapt": tp_adapt_vals, "inst_q": inst_vals}
    return points, axes, res


def _grid_fig4_slice(fast=False):
    """
    Bowles (reviewer) request: 1D slice through Figure 4 at Q₀ = 0.45
    (the best-estimate institutional quality).  X-axis: T_P from 0 to 1.
    Shows the T_P transition in a realistic parameter setting rather than
    requiring readers to read it off a 2D phase diagram.
    This generates a 1D array (res × 1) for a single line plot.
    """
    res     = 12 if fast else 30
    tp_vals = np.linspace(0.0, 1.0, res)
    kin_val = 0.0007   # best-estimate kinship dilution

    base = dict(n_agents=300, n_groups=20, n_gens=600,
                init_inst_q=0.45, kin_dilution=kin_val,
                bgs=0.28, mismatch=0.42, mut_rate=0.05,
                sanction_strength=0.40, bgs_cultural_rise=0.0,
                tp_adapt=0.0)

    points = [
        {**base, "cultural_weight": float(tp)}
        for tp in tp_vals
    ]
    # Store as 1 × res so aggregate still works (res_y=1)
    axes = {"cultural_weight": tp_vals, "dummy": np.array([0.45])}
    return points, axes, res, 1   # note: returns res_x, res_y separately

def _aggregate(results, axes_keys, res, n_reps, points):
    """
    Average over replicates and reshape into (res × res) grids.
    Returns dict of 2D arrays keyed by statistic name.

    Also computes per-regime fraction arrays (regime_frac_COMPLETE etc.)
    and a bistability index = 1 - max(fraction across 4 regimes), which
    is highest when replicates are split across regimes.  These are used
    by plots.py to add a variance/bistability overlay to phase diagrams,
    addressing Smaldino (2020) concern about stochastic variance.
    """
    metrics   = ["coop_final", "inst_q_final", "pathol_final",
                  "thresh_final", "regime_int", "bgs_eff_final", "tp_eff_final"]
    grids     = {m: np.zeros((res, res)) for m in metrics}
    counts    = np.zeros((res, res), dtype=int)

    # Per-regime counts for variance overlay
    # regime_int encoding: DEFECTION=0, PATHOLOGICAL=1, STALLED=2, COMPLETE=3
    regime_counts = np.zeros((res, res, 4), dtype=int)

    for r in results:
        pt_idx = r["seed"] // n_reps
        i, j   = divmod(pt_idx, res)
        for m in metrics:
            grids[m][i, j] += r[m]
        counts[i, j] += 1
        regime_counts[i, j, int(round(r["regime_int"]))] += 1

    safe_counts = np.where(counts > 0, counts, 1)
    for m in metrics:
        grids[m] /= safe_counts

    # Regime fractions and bistability index
    safe_rep = np.where(counts > 0, counts, 1)[..., np.newaxis]
    regime_fracs = regime_counts / safe_rep          # shape (res, res, 4)
    # Bistability index: 0 = all reps agree, 1 = perfectly split
    max_frac = regime_fracs.max(axis=2)             # shape (res, res)
    bistability = 1.0 - max_frac                    # high = uncertain/bistable

    grids["bistability"]         = bistability
    grids["frac_complete"]       = regime_fracs[:, :, 3]
    grids["frac_stalled"]        = regime_fracs[:, :, 2]
    grids["frac_pathological"]   = regime_fracs[:, :, 1]
    grids["frac_defection"]      = regime_fracs[:, :, 0]

    return grids


# ── Save results ──────────────────────────────────────────────────────────────

def _save(grids, axes, out_dir, fig_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fig_name}_grids.npz"
    np.savez(out_path, **{k: v for k, v in grids.items()},
             **{f"axis_{k}": v for k, v in axes.items()})
    print(f"  Saved → {out_path}")


def _grid_sensitivity(fast=False):
    """
    Sensitivity analysis: Fig 1 axes (inst_q × kin_dilution) repeated at
    9 combinations of N (100/300/500), G (10/20/40), and inst_adapt lag
    (10/25/50 generation timescale, corresponding to inst_adapt 0.10/0.04/0.02).
    Each sub-grid is 10×10 (fast) or 20×20 (full) with 10 reps.
    Returns a list of (label, points, axes, res) tuples.
    """
    res = 10 if fast else 20
    inst_vals = np.linspace(0.10, 0.90, res)
    kin_vals  = np.linspace(0.0001, 0.0014, res)
    n_reps_sens = 3 if fast else 10

    combos = [
        ("N=100,G=10,lag=10",  dict(n_agents=100, n_groups=10, inst_adapt=0.10)),
        ("N=100,G=20,lag=25",  dict(n_agents=100, n_groups=20, inst_adapt=0.04)),
        ("N=100,G=40,lag=50",  dict(n_agents=100, n_groups=40, inst_adapt=0.02)),
        ("N=300,G=10,lag=10",  dict(n_agents=300, n_groups=10, inst_adapt=0.10)),
        ("N=300,G=20,lag=25",  dict(n_agents=300, n_groups=20, inst_adapt=0.04)),
        ("N=300,G=40,lag=50",  dict(n_agents=300, n_groups=40, inst_adapt=0.02)),
        ("N=500,G=10,lag=10",  dict(n_agents=500, n_groups=10, inst_adapt=0.10)),
        ("N=500,G=20,lag=25",  dict(n_agents=500, n_groups=20, inst_adapt=0.04)),
        ("N=500,G=40,lag=50",  dict(n_agents=500, n_groups=40, inst_adapt=0.02)),
    ]

    base = dict(n_gens=600, bgs=0.28, mismatch=0.42, mut_rate=0.05,
                sanction_strength=0.40)

    result_list = []
    for label, overrides in combos:
        cfg = {**base, **overrides}
        points = [
            {**cfg, "init_inst_q": float(iq), "kin_dilution": float(kd)}
            for iq, kd in product(inst_vals, kin_vals)
        ]
        axes = {"inst_q": inst_vals, "kin_dilution": kin_vals}
        result_list.append((label, points, axes, res, n_reps_sens))
    return result_list

def main():
    parser = argparse.ArgumentParser(description="Stalled Transition ABM sweep")
    parser.add_argument("--fig",   type=int, default=0,
                        help="Which figure to run (1/2/3); 0 = all")
    parser.add_argument("--fast",  action="store_true",
                        help="Coarse grid + 3 reps — quick smoke test")
    parser.add_argument("--reps",  type=int, default=None,
                        help="Override number of replicates per grid point")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override number of worker processes")
    parser.add_argument("--outdir", type=str, default="results",
                        help="Output directory for .npz files")
    args = parser.parse_args()

    n_reps   = args.reps   or (3  if args.fast else 20)

    # On Intel Macs, os.cpu_count() returns logical CPUs (physical × 2 due
    # to hyperthreading).  For CPU-bound NumPy work, parallelism saturates
    # at the physical core count; using all logical CPUs adds scheduling
    # overhead without throughput gain.  We cap at physical_cores - 1 to
    # keep the machine responsive.  On the 8-core i7 iMac: 7 workers.
    logical_cpus   = os.cpu_count() or 8
    physical_cores = logical_cpus // 2   # hyperthreading: 16 logical → 8 physical
    n_workers      = args.workers or max(1, physical_cores - 1)
    out_dir        = Path(args.outdir)

    print(f"Logical CPUs: {logical_cpus}  |  Physical cores (est.): {physical_cores}")
    print(f"Workers: {n_workers}  |  Reps per point: {n_reps}  "
          f"|  Fast mode: {args.fast}")

    figs_to_run = [args.fig] if args.fig else [1, 2, 3, 4, 5, 6, 7, 8]

    for fig_num in figs_to_run:
        if fig_num == 1:
            points, axes, res = _grid_fig1(args.fast)
            desc = "Fig1 (inst_q × kin_dilution)"
        elif fig_num == 2:
            points, axes, res = _grid_fig2(args.fast)
            desc = "Fig2 (bgs × mismatch)"
        elif fig_num == 3:
            points, axes, res = _grid_fig3(args.fast)
            desc = "Fig3 (sanction_strength × kin_dilution)"
        elif fig_num == 4:
            points, axes, res = _grid_fig4(args.fast)
            desc = "Fig4 (cultural_weight × kin_dilution)  [ETII Ext 1]"
        elif fig_num == 5:
            points, axes, res = _grid_fig5(args.fast)
            desc = "Fig5 (bgs_cultural_rise × inst_q)  [ETII Ext 2]"

        elif fig_num == 7:
            # Extension 3: endogenous TP evolution
            points, axes, res = _grid_fig7(args.fast)
            desc = "Fig7 (tp_adapt × inst_q)  [ETII Ext 3: endogenous TP]"

        elif fig_num == 8:
            # Bowles 1D slice: TP from 0→1 at Q₀ = 0.45
            result = _grid_fig4_slice(args.fast)
            points, axes, res_x, res_y = result
            desc = "Fig8 (TP slice at Q₀=0.45)  [Bowles supplementary]"
            n_pts = len(points)
            print(f"\n{'─'*60}")
            print(f"Running {desc}")
            print(f"  Grid: {res_x}×1 = {n_pts} points × {n_reps} reps = {n_pts*n_reps} runs")
            t0 = time.time()
            results = _run_grid(points, n_reps, n_workers, desc=desc)
            # Aggregate as 1D — compute mean and SD
            metrics_1d = ["coop_final", "inst_q_final", "pathol_final",
                          "regime_int", "tp_eff_final"]
            grids_1d   = {m: np.zeros(res_x) for m in metrics_1d}
            grids_sq   = {m: np.zeros(res_x) for m in metrics_1d}  # for SD
            regime_counts_1d = np.zeros((res_x, 4), dtype=int)
            counts_1d = np.zeros(res_x, dtype=int)
            for r in results:
                idx = r["seed"] // n_reps
                for m in metrics_1d:
                    v = r.get(m, 0)
                    grids_1d[m][idx] += v
                    grids_sq[m][idx] += v * v
                counts_1d[idx] += 1
                regime_counts_1d[idx, int(round(r["regime_int"]))] += 1
            safe = np.where(counts_1d > 0, counts_1d, 1)
            for m in metrics_1d:
                grids_1d[m] /= safe
            # SD = sqrt(E[x²] - E[x]²)
            coop_sd   = np.sqrt(np.maximum(0, grids_sq["coop_final"]/safe
                                           - grids_1d["coop_final"]**2))
            pathol_sd = np.sqrt(np.maximum(0, grids_sq["pathol_final"]/safe
                                           - grids_1d["pathol_final"]**2))
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(out_dir / "fig8_tp_slice.npz",
                     **grids_1d,
                     coop_sd=coop_sd,
                     pathol_sd=pathol_sd,
                     axis_tp=axes["cultural_weight"])
            print(f"  Saved → {out_dir}/fig8_tp_slice.npz")
            print(f"  Done in {time.time()-t0:.1f}s")
            continue

        elif fig_num == 6:
            # Sensitivity analysis — multiple sub-grids
            sens_list = _grid_sensitivity(args.fast)
            sub_grids = {}
            for label, pts, axes_s, res_s, n_reps_s in sens_list:
                n_pts_s = len(pts)
                desc_s = f"Sensitivity [{label}]"
                print(f"\n  {desc_s}: {res_s}×{res_s} = {n_pts_s} points × {n_reps_s} reps")
                results_s = _run_grid(pts, n_reps_s, n_workers, desc=desc_s)
                grids_s = _aggregate(results_s, list(axes_s.keys()), res_s, n_reps_s, pts)
                sub_grids[label] = {
                    "regime_int": grids_s["regime_int"],
                    "axis_inst_q": axes_s["inst_q"],
                    "axis_kin_dilution": axes_s["kin_dilution"],
                }
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(out_dir / "fig6_sensitivity.npz", **{
                f"{lbl.replace(',','_').replace('=','')}_regime": v["regime_int"]
                for lbl, v in sub_grids.items()
            }, **{
                "labels": np.array(list(sub_grids.keys()), dtype=object),
                "axis_inst_q": list(sub_grids.values())[0]["axis_inst_q"],
                "axis_kin_dilution": list(sub_grids.values())[0]["axis_kin_dilution"],
            })
            print(f"  Saved → {out_dir}/fig6_sensitivity.npz")
            continue
        n_pts = len(points)
        print(f"\n{'─'*60}")
        print(f"Running {desc}")
        print(f"  Grid: {res}×{res} = {n_pts} points × {n_reps} reps "
              f"= {n_pts * n_reps} total runs")

        t0 = time.time()
        results = _run_grid(points, n_reps, n_workers, desc=desc)
        grids   = _aggregate(results, list(axes.keys()), res, n_reps, points)
        _save(grids, axes, out_dir, f"fig{fig_num}")
        print(f"  Done in {time.time()-t0:.1f}s")

    print("\nAll sweeps complete.")


if __name__ == "__main__":
    # Required on macOS for multiprocessing spawn
    mp.set_start_method("spawn", force=True)
    main()
