

# Stalled Transition — Agent-Based Model

**Associated paper:** Stearns SC (submitted) "Stalled: Humans and the major evolutionary transition from individual to group." *Evolutionary Human Sciences.*

---

## Overview

This repository contains the simulation code for a formal agent-based model (ABM) testing whether humans are stalled partway through a major evolutionary transition from individual to group, and whether the Waring and Wood (2025) ETII framework resolves that stall. The model produces phase diagrams across empirically calibrated parameter spaces and implements three ETII extensions: fixed cultural weight (T_P, Extension 1), rising cultural group selection (BGS, Extension 2), and endogenous T_P evolution (Extension 3).

---

## Files

| File | Purpose |
|---|---|
| `model.py` | Core simulation engine (vectorised NumPy). Defines `Config` and `World`. |
| `sweep.py` | Parameter sweep across phase space using `multiprocessing.Pool`. |
| `plots.py` | Publication-quality figures from sweep results. |
| `README.md` | This file. |

After running, two subdirectories appear:

- `results/`  — `.npz` files with sweep data, one per figure
- `figures/`  — `.pdf` and `.png` files at 180 dpi

---

## Setup (once only)

Requires Python 3.10+.

```bash
cd ~/Desktop/stalled_transition

python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy
```

On Apple Silicon (M1–M4), NumPy links automatically to Apple's Accelerate framework. On Linux with conda, `conda install numpy` links against MKL, which is faster for the vectorised inner loops.

---

## Running

### Smoke test (5–10 minutes)

Runs coarse 8×8 grids with 3 replicates — enough to verify output and inspect figure layout:

```bash
source .venv/bin/activate
python sweep.py --fast
python plots.py
```

Open `figures/` to inspect all eight figures.

### Full production sweep

30×30 grids (Figs 1–5, 7), 24×24 with 10 reps (Fig 6 sensitivity), 30-point slice (Fig 8):

```bash
python sweep.py
python plots.py
```

### Individual figures

```bash
python sweep.py --fig 1    # Fig 1: institutional quality × kinship dilution (baseline)
python sweep.py --fig 2    # Fig 2: BGS weight × mismatch severity
python sweep.py --fig 3    # Fig 3: sanction strength × kinship dilution
python sweep.py --fig 4    # Fig 4: ETII Ext 1 — fixed T_P × kinship dilution
python sweep.py --fig 5    # Fig 5: ETII Ext 2 — rising BGS × institutional quality
python sweep.py --fig 6    # Fig 6: sensitivity analysis (9 sub-grids: N × G × lag)
python sweep.py --fig 7    # Fig 7: ETII Ext 3 — endogenous T_P × institutional quality
python sweep.py --fig 8    # Fig 8 (Supp.): 1D T_P slice at fixed Q₀ = 0.45

python plots.py --fig 1    # plot Fig 1 from saved results (sweep must have run first)
python plots.py --fig 6    # etc.
```

### Limiting CPU workers

```bash
python sweep.py --workers 4    # useful on shared or hot machines
```

---

## Parameters

All parameters live in the `Config` dataclass in `model.py`. Values below are the defaults used throughout the paper.

### Population structure

| Parameter | Default | Notes |
|---|---|---|
| `n_agents` | 300 | Total agents across all groups. Robustness confirmed at 100 and 500 (Fig 6). |
| `n_groups` | 20 | Number of groups (mean group size = 15). Robustness confirmed at 10 and 40. |
| `n_gens` | 600 | Generations per run ≈ 12,000 years (1 generation ≈ 20 years). |
| `init_thresh` | 0.50 | Mean of initial cooperation threshold τ distribution. |
| `init_thresh_sd` | 0.15 | SD of initial τ distribution; clipped to [0, 1]. |
| `mut_rate` | 0.05 | Probability of Gaussian mutation to τ each generation. |
| `mut_sd` | 0.07 | SD of mutation step. |
| `migration` | 0.04 | Fraction of agents randomly reassigned to a new group each generation. |

### Selection and ecology

| Parameter | Default | Empirical range | Paper symbol |
|---|---|---|---|
| `bgs` | 0.28 | 0.15–0.45 | ω (BGS weight). y-axis in Fig 2; fixed elsewhere. |
| `kin_dilution` | 0.0007 | 0.0004–0.0013 /gen | ṙ (kinship dilution rate). x-axis in Figs 1, 3, 4. |
| `mismatch` | 0.42 | 0.25–0.60 | φ (mismatch severity). x-axis in Fig 2; fixed elsewhere. |
| `sanction_strength` | 0.40 | 0.20–0.65 | s. y-axis in Fig 3; fixed elsewhere. |
| `init_inst_q` | 0.50 | 0.25–0.65 | Q₀ (initial institutional quality). y-axis in Figs 1, 5, 7. |
| `inst_adapt` | 0.04 | — | α: rate Q tracks local cooperation (25-generation lag). Tested at 0.10 and 0.02 in Fig 6. |

### ETII extensions

| Parameter | Default | Description |
|---|---|---|
| `cultural_weight` | 0.0 | T_P: fraction of offspring τ set by cultural conformity transmission toward group mean. 0 = baseline (purely genetic inheritance); 1 = fully cultural. y-axis in Fig 4. |
| `bgs_cultural_rise` | 0.0 | Δω: per-generation increment added to effective BGS ω. Effective ω at generation t = min(bgs + bgs_cultural_rise × t, 1.0). x-axis in Fig 5. |
| `tp_adapt` | 0.0 | α_TP: rate at which T_P evolves toward Q̄ × c̄. 0 = T_P is fixed at `cultural_weight` (Exts 1 and 2). x-axis in Fig 7. |
| `tp_init` | 0.0 | Starting T_P value when `tp_adapt > 0` (Extension 3). |

---

## Fitness functions

Agent fitness for cooperators and defectors in group g:

```
w_coop   = 1 + c̄_g × 1.8 × max(r, 0.05) + r × 0.3 − 0.15 + Q_g × s × 0.5
w_defect = 1 + c̄_g × 0.9 − φ × 0.2 − Q_g × s × c̄_g

G_fit_g  = max(0.05, 1 + c̄_g × (0.5 + r×0.8) × (1 + ω_eff×0.5) − φ × (1−c̄_g) × (1−Q_g))

agent_fitness = max(0.01, w_within + G_fit_g × ω_eff)
```

Institutional quality update (cultural timescale):

```
Q_g(t+1) = Q_g(t) + α × (0.20 + c̄_g × 0.60 + Q₀ × 0.20 − Q_g(t))
```

Cultural inheritance of cooperation threshold (Extension 1 and 3):

```
τ_offspring = (1 − T_P) × τ_parent + T_P × group_mean(τ) + ε
              where ε ~ N(0, mut_sd) with probability mut_rate
```

Extension 3 — endogenous T_P evolution:

```
T_P(t+1) = T_P(t) + α_TP × (Q̄ × c̄ − T_P(t))
```

---

## Regime classification

Regimes are classified from the mean of the final 50 generations, in order of precedence:

| Regime | Integer code | Condition |
|---|---|---|
| COMPLETE | 3 | c̄ > 0.70 AND Q̄ > 0.65 |
| PATHOLOGICAL | 1 | P > 0.45, where P = (1 − c̄)(1 − Q̄)(1 + φ) × 0.8 |
| STALLED | 2 | c̄ ∈ [0.35, 0.70], P ≤ 0.45, not COMPLETE |
| DEFECTION | 0 | c̄ ≤ 0.35 AND P ≤ 0.45 |

These labels identify model-defined threshold states. The bistability index (1 − max_regime_fraction across the 20 replicates per grid cell) is computed per cell; values above 0.30 appear as a grey dotted contour on Panel A of each phase diagram, indicating genuine sensitivity to initial conditions at that parameter combination.

---

## Output files

After `python sweep.py`, `results/` contains:

| File | Figure | Axes |
|---|---|---|
| `fig1_grids.npz` | Fig 1 | Q₀ × ṙ |
| `fig2_grids.npz` | Fig 2 | ω × φ |
| `fig3_grids.npz` | Fig 3 | s × ṙ |
| `fig4_grids.npz` | Fig 4 | T_P (cultural_weight) × ṙ |
| `fig5_grids.npz` | Fig 5 | Δω (bgs_cultural_rise) × Q₀ |
| `fig6_sensitivity.npz` | Fig 6 | 9 sub-grids: N × G × lag |
| `fig7_grids.npz` | Fig 7 | α_TP (tp_adapt) × Q₀ |
| `fig8_tp_slice.npz` | Fig 8 | T_P (1D) at fixed Q₀ = 0.45 |

Each 2D grid `.npz` file contains:

```python
import numpy as np
data = np.load("results/fig1_grids.npz")

data["regime_int"]        # (30, 30) — mean regime (0=defection, 1=pathological,
                          #             2=stalled, 3=complete)
data["coop_final"]        # (30, 30) — mean cooperation rate c̄
data["inst_q_final"]      # (30, 30) — mean institutional quality Q̄
data["pathol_final"]      # (30, 30) — mean pathology index P
data["thresh_final"]      # (30, 30) — mean cooperation threshold τ
data["bgs_eff_final"]     # (30, 30) — mean effective BGS weight ω_eff
data["bistability"]       # (30, 30) — 1 − max_regime_fraction (0 = all agree)
data["frac_complete"]     # (30, 30) — fraction of replicates → COMPLETE
data["frac_stalled"]      # (30, 30) — fraction of replicates → STALLED
data["frac_pathological"] # (30, 30) — fraction of replicates → PATHOLOGICAL
data["frac_defection"]    # (30, 30) — fraction of replicates → DEFECTION
data["axis_inst_q"]       # (30,) — axis values for institutional quality
data["axis_kin_dilution"] # (30,) — axis values for kinship dilution rate
```

The 1D slice file (`fig8_tp_slice.npz`) contains:

```python
data["coop_final"]    # (30,) — mean cooperation rate
data["pathol_final"]  # (30,) — mean pathology index
data["regime_int"]    # (30,) — mean regime
data["coop_sd"]       # (30,) — SD of cooperation rate across 20 replicates
data["pathol_sd"]     # (30,) — SD of pathology index across 20 replicates
data["axis_tp"]       # (30,) — T_P values (0 to 1)
```

---

## Figure descriptions

| Figure | Key finding |
|---|---|
| **Fig 1** (3 panels) | Primary phase diagram Q₀ × ṙ. Best-estimate star falls in PATHOLOGICAL across baseline parameter space. |
| **Fig 2** (2 panels) | ω × φ. High BGS alone is insufficient; mismatch is the binding constraint. |
| **Fig 3** (2 panels) | s × ṙ substitution surface. Sanction-strength threshold at s ≈ 0.45–0.50 separates COMPLETE from PATHOLOGICAL. |
| **Fig 4** (2 panels) | ETII Ext 1 (fixed T_P × ṙ). Even T_P ≈ 0.05–0.10 eliminates most PATHOLOGICAL and STALLED zones; indicative empirical range lies entirely in COMPLETE. |
| **Fig 5** (3 panels) | ETII Ext 2 (Δω × Q₀). Rising BGS alone insufficient below Q₀ ≈ 0.45–0.50. Panel C confirms ω is accumulating. |
| **Fig 6** (3×3 panels) | Sensitivity analysis. Qualitative phase structure preserved across all N × G × lag combinations. |
| **Fig 7** (2 panels) | ETII Ext 3 (α_TP × Q₀). Endogenous T_P evolution expands COMPLETE zone at moderate Q₀; stall may be self-resolving above Q₀ threshold. |
| **Fig 8** (2 panels, Supp.; = Supplementary Figure S1 in the paper) | 1D T_P slice at Q₀ = 0.45. ±1 SD bands show bistability in transition zone T_P ≈ 0.05–0.15. |

---

## Performance

The sweep uses `multiprocessing.Pool` with `cpu_count() − 1` workers by default. Expected timings for the full sweep (all 8 figures); these scale with grid size, so re-measure if you change it:

| Hardware | Approx. time |
|---|---|
| Apple M1 iMac | ~170 min |
| Apple M3/M4 iMac | ~115–135 min |
| Linux, 16 cores | ~45–50 min |
| Linux, 32 cores | ~30 min |

The sensitivity analysis (Fig 6: 9 sub-grids × 576 points × 10 reps = 51,840 runs) is the longest single job. Figs 1–5 and 7 each run 900 points × 20 reps = 18,000 runs.

---

## Citation

If you use this code in published work, please cite:

> Stearns SC (submitted) Stalled: Humans and the major evolutionary transition from individual to group. *Evolutionary Human Sciences.* Code available at [repository URL — to be added before submission].

