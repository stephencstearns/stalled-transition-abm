"""
model.py  —  Stalled Transition ABM: core simulation engine
=============================================================
Implements the agent-based model for Stearns (in prep),
"Stalled: Humans and the Major Evolutionary Transition from
Individual to Group."

Architecture
------------
Agents carry a heritable cooperation threshold τ ∈ [0,1].
An agent cooperates in a group if the group's scaled size
falls below τ (adjusted upward by local institutional quality).
Groups run a public goods game; their fitness depends on
cooperation rate and institutional quality.
Institutional quality Q per group evolves on a slow cultural
timescale toward the local cooperation rate.
Kinship dilution r decreases at a fixed rate per generation,
reducing the kin-selection component of within-group payoffs.
Individual fitness combines within-group PGG payoff and a
between-group competition term weighted by cfg.bgs.
Reproduction is fitness-proportionate with Gaussian mutation
on τ. A small migration rate shuffles agents between groups.

All inner loops use NumPy vectorisation; no Numba required.
The module is import-safe for multiprocessing on macOS
(spawn start method).

Generation-time calibration
----------------------------
One model generation corresponds to one human generation, taken here
as approximately 20 years — the intergenerational interval in Holocene
hunter-gatherer and early agricultural populations (Hill et al. 2011;
Fenner 2005).  This anchors the model to real historical time:

  600 generations  =  ~12,000 years  (end of Pleistocene to present)
  300 generations  =  ~6,000 years   (Neolithic to present)

Default kinship dilution rate (0.0007 per generation) depletes starting
relatedness r = 0.25 to zero in ~357 generations (~7,140 years from the
Pleistocene/Holocene boundary), placing the relatedness collapse in the
early Bronze Age — roughly coincident with the emergence of the first
state-level societies (~3000 BCE).  This is the intended calibration.

Cultural adaptation rate (inst_adapt = 0.04/generation) gives institutions
a characteristic response time of ~25 generations = 500 years, consistent
with archaeological and historical estimates of institutional change rates.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Population
    n_agents:      int   = 300
    n_groups:      int   = 20
    n_gens:        int   = 600

    # Initial conditions
    init_thresh:   float = 0.50   # mean initial cooperation threshold
    init_thresh_sd:float = 0.15   # spread of initial thresholds
    init_inst_q:   float = 0.50   # initial institutional quality (all groups)

    # Evolution parameters
    mut_rate:      float = 0.05   # per-agent per-generation mutation probability
    mut_sd:        float = 0.07   # std dev of mutation step
    migration:     float = 0.04   # prob agent moves to random group each gen

    # Social dynamics
    bgs:           float = 0.28   # between-group selection weight
    kin_dilution:  float = 0.0007 # decrease in relatedness per generation
    mismatch:      float = 0.42   # mismatch penalty intensity
    inst_adapt:    float = 0.04   # rate institutional quality tracks coop rate

    # PGG parameters
    synergy:       float = 1.8    # multiplier on cooperative contributions
    coop_cost:     float = 0.15   # cost to cooperator

    # Institution-as-sanction strength:
    # Fraction of the group cooperation rate that becomes a fitness penalty
    # on defectors when institutional quality is high.  This is the key
    # parameter that allows institutions to *substitute* for kinship as a
    # cooperation-sustaining force once relatedness is depleted by dilution.
    sanction_strength: float = 0.40

    # ── ETII Extension 1: Cultural determination of cooperation threshold ──────
    # Implements Waring & Wood (2025) T_P index as a model parameter.
    # cultural_weight in [0,1] sets the fraction of each agent's cooperation
    # threshold tau that is culturally (rather than genetically) determined.
    # When 0.0: tau evolves purely genetically (original model).
    # When 1.0: tau is reset each generation toward the local group mean
    #           (cultural conformity transmission); mutation represents guided
    #           cultural innovation rather than genetic drift.
    # At intermediate values, offspring tau is a weighted blend of genetic
    # inheritance (parent tau) and cultural inheritance (group mean tau).
    cultural_weight: float = 0.0

    # ── ETII Extension 2: Rising cultural group selection ─────────────────────
    # Implements Waring & Wood's claim that cultural group selection
    # strengthens over time as cultural adaptation accumulates.
    # bgs_cultural_rise is the per-generation increment added to the
    # effective between-group selection weight, starting from cfg.bgs.
    # Effective BGS at generation t: min(bgs + bgs_cultural_rise * t, 1.0)
    # When 0.0: BGS is constant (original model).
    bgs_cultural_rise: float = 0.0

    # ── ETII Extension 3: Endogenous TP evolution ─────────────────────────────
    # Addresses Henrich (reviewer) concern that TP should emerge from the
    # model's dynamics rather than being a fixed exogenous parameter.
    # TP evolves toward a target driven by institutional quality and cooperation:
    #   target_tp = mean(Q_g) * mean(c̄_g)
    # Update rule (analogous to institutional quality):
    #   TP(t+1) = TP(t) + tp_adapt * (target_tp - TP(t))
    # This captures the idea that cultural group-level adaptations accumulate
    # more readily in populations already high in institutional quality and
    # cooperation — the conditions Waring & Wood associate with Stage III.
    # When tp_adapt = 0.0: TP is fixed at tp_init (reduces to Extensions 1/2).
    tp_adapt: float = 0.0     # rate TP evolves toward target (0 = fixed)
    tp_init:  float = 0.0     # initial TP value when tp_adapt > 0

    # Derived (auto-set)
    group_size_ref: float = field(init=False)  # reference for size fraction

    def __post_init__(self):
        self.group_size_ref = (self.n_agents / self.n_groups) * 2.5


# ── Simulation ────────────────────────────────────────────────────────────────

class Simulation:
    """
    Run a single replicate of the ABM and return a Results object.
    Designed to be pickle-safe for multiprocessing.Pool.
    """

    def __init__(self, cfg: Config, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    # ── Initialise state ──────────────────────────────────────────────────────

    def _init_state(self):
        cfg = self.cfg
        rng = self.rng

        # Agent arrays (length n_agents)
        self.thresh  = np.clip(
            rng.normal(cfg.init_thresh, cfg.init_thresh_sd, cfg.n_agents),
            0.0, 1.0
        )
        self.groups  = rng.integers(0, cfg.n_groups, cfg.n_agents)

        # Group arrays (length n_groups)
        self.inst_q  = np.clip(
            rng.normal(cfg.init_inst_q, 0.08, cfg.n_groups),
            0.0, 1.0
        )

        # Population-level state
        self.relatedness = 0.25
        self.generation  = 0          # current generation (for bgs_cultural_rise)
        # Extension 3: TP starts at tp_init and evolves if tp_adapt > 0
        # When tp_adapt == 0, tp_eff == cultural_weight (fixed, as in Ext 1)
        self.tp_eff = cfg.tp_init if cfg.tp_adapt > 0.0 else cfg.cultural_weight

    # ── One generation ────────────────────────────────────────────────────────

    def _step(self):
        cfg  = self.cfg
        rng  = self.rng
        N    = cfg.n_agents
        G    = cfg.n_groups

        # --- Group membership counts and cooperation decisions ----------------
        group_sizes = np.bincount(self.groups, minlength=G).astype(float)
        size_frac   = group_sizes / cfg.group_size_ref          # (G,)

        # Effective threshold per agent: own τ + institutional boost
        eff_thresh = self.thresh + self.inst_q[self.groups] * 0.25  # (N,)
        cooperates = size_frac[self.groups] <= eff_thresh            # (N,) bool

        # Cooperation rate per group
        coop_count  = np.bincount(self.groups, weights=cooperates.astype(float),
                                   minlength=G)
        nonzero     = group_sizes > 0
        coop_rate_g = np.where(nonzero, coop_count / np.where(nonzero, group_sizes, 1), 0.0)

        # --- Institutional quality update (cultural timescale) ----------------
        target_q = 0.20 + coop_rate_g * 0.60 + cfg.init_inst_q * 0.20
        self.inst_q += cfg.inst_adapt * (target_q - self.inst_q)
        np.clip(self.inst_q, 0.0, 1.0, out=self.inst_q)

        # --- Kinship dilution -------------------------------------------------
        self.relatedness = max(0.0, self.relatedness - cfg.kin_dilution)

        # --- Effective between-group selection weight (Extension 2) -----------
        # BGS rises linearly with time, representing accumulation of group-level
        # cultural adaptations (Waring & Wood 2025). Capped at 1.0.
        bgs_eff = min(1.0, cfg.bgs + cfg.bgs_cultural_rise * self.generation)

        # --- Group fitness (for between-group selection) ----------------------
        pgb   = coop_rate_g * (0.5 + self.relatedness * 0.8) * (1 + bgs_eff * 0.5)
        mpen  = cfg.mismatch * (1 - coop_rate_g) * (1 - self.inst_q)
        g_fit = np.maximum(0.05, 1.0 + pgb - mpen)              # (G,)

        # --- Individual fitness -----------------------------------------------
        # Within-group component
        # Cooperators benefit from synergy, kin-selection, and a bonus from
        # strong local institutions (enforcement reduces effective coop cost).
        inst_bonus  = self.inst_q[self.groups] * cfg.sanction_strength
        pg_bonus    = coop_rate_g[self.groups] * cfg.synergy * max(self.relatedness, 0.05)
        w_coop      = (1.0 + pg_bonus + self.relatedness * 0.3
                       - cfg.coop_cost + inst_bonus * 0.5)

        # Defectors gain a free-rider surplus but face institutional sanction
        # proportional to both institutional quality and local cooperation rate.
        # This is the key mechanism: strong institutions substitute for kinship
        # as a driver of cooperation once relatedness has been diluted.
        defect_sanction = (self.inst_q[self.groups] * cfg.sanction_strength
                           * coop_rate_g[self.groups])
        w_defect    = (1.0 + coop_rate_g[self.groups] * 0.9
                       - cfg.mismatch * 0.2 - defect_sanction)

        within_fit  = np.where(cooperates, w_coop, w_defect)

        # Between-group component (uses rising bgs_eff)
        between_fit   = g_fit[self.groups] * bgs_eff
        agent_fit     = np.maximum(0.01, within_fit + between_fit)  # (N,)

        # --- Reproduction (fitness-proportionate selection) -------------------
        fit_sum  = agent_fit.sum()
        probs    = agent_fit / fit_sum
        parents  = rng.choice(N, size=N, replace=True, p=probs)

        new_thresh = self.thresh[parents].copy()
        new_groups = self.groups[parents].copy()

        # --- Extension 3: Endogenous TP evolution --------------------------------
        # TP rises toward mean(Q) * mean(c̄) on a cultural timescale.
        # When tp_adapt == 0 this block has no effect (tp_eff stays fixed).
        if cfg.tp_adapt > 0.0:
            target_tp = float(np.clip(
                self.inst_q.mean() * coop_rate_g.mean(), 0.0, 1.0))
            self.tp_eff = float(np.clip(
                self.tp_eff + cfg.tp_adapt * (target_tp - self.tp_eff),
                0.0, 1.0))

        # Extension 1 / 3: Cultural determination of cooperation threshold
        # (Waring & Wood 2025, T_P mechanism)
        # Uses self.tp_eff, which equals cfg.cultural_weight when tp_adapt==0
        # (fixed T_P, Extensions 1/2) or the evolved value when tp_adapt>0
        # (endogenous T_P, Extension 3).
        if self.tp_eff > 0.0:
            group_mean_thresh = np.zeros(G)
            for g in range(G):
                mask = new_groups == g
                if mask.any():
                    group_mean_thresh[g] = new_thresh[mask].mean()
                else:
                    group_mean_thresh[g] = new_thresh.mean()
            cultural_target = group_mean_thresh[new_groups]
            new_thresh = ((1.0 - self.tp_eff) * new_thresh
                          + self.tp_eff * cultural_target)

        # Mutation on threshold (genetic or cultural innovation)
        mutating     = rng.random(N) < cfg.mut_rate
        n_mut        = mutating.sum()
        if n_mut > 0:
            new_thresh[mutating] += rng.normal(0, cfg.mut_sd, n_mut)
            np.clip(new_thresh, 0.0, 1.0, out=new_thresh)

        # Migration
        migrating    = rng.random(N) < cfg.migration
        n_mig        = migrating.sum()
        if n_mig > 0:
            new_groups[migrating] = rng.integers(0, G, n_mig)

        self.thresh = new_thresh
        self.groups = new_groups
        self.generation += 1

        # --- Compute summary statistics ---------------------------------------
        mean_t   = self.thresh.mean()
        sd_t     = self.thresh.std()
        mean_cr  = coop_rate_g.mean()
        mean_q   = self.inst_q.mean()
        pathol   = float(np.clip(
            (1 - mean_cr) * (1 - mean_q) * (1 + cfg.mismatch) * 0.8, 0, 1))

        return mean_t, sd_t, mean_cr, mean_q, pathol, g_fit.copy(), bgs_eff, float(self.tp_eff)

    # ── Run full simulation ───────────────────────────────────────────────────

    def _init_history(self):
        G = self.cfg.n_gens
        self.h_mean_thresh   = np.empty(G)
        self.h_sd_thresh     = np.empty(G)
        self.h_coop_rate     = np.empty(G)
        self.h_mean_inst_q   = np.empty(G)
        self.h_pathology     = np.empty(G)
        self.h_group_fitness = np.empty((G, self.cfg.n_groups))
        self.h_relatedness   = np.empty(G)
        self.h_bgs_eff       = np.empty(G)
        self.h_tp_eff        = np.empty(G)   # Extension 3: evolved TP value

    def run(self) -> "Results":
        self._init_state()
        self._init_history()
        cfg = self.cfg

        for t in range(cfg.n_gens):
            mt, st, cr, iq, pa, gf, be, te = self._step()
            self.h_mean_thresh[t]   = mt
            self.h_sd_thresh[t]     = st
            self.h_coop_rate[t]     = cr
            self.h_mean_inst_q[t]   = iq
            self.h_pathology[t]     = pa
            self.h_group_fitness[t] = gf
            self.h_relatedness[t]   = self.relatedness
            self.h_bgs_eff[t]       = be
            self.h_tp_eff[t]        = te

        return Results(
            cfg             = cfg,
            mean_thresh     = self.h_mean_thresh,
            sd_thresh       = self.h_sd_thresh,
            coop_rate       = self.h_coop_rate,
            mean_inst_q     = self.h_mean_inst_q,
            pathology       = self.h_pathology,
            group_fitness   = self.h_group_fitness,
            relatedness     = self.h_relatedness,
            bgs_eff         = self.h_bgs_eff,
            tp_eff          = self.h_tp_eff,
        )


# ── Results container ─────────────────────────────────────────────────────────

@dataclass
class Results:
    cfg:            Config
    mean_thresh:    np.ndarray   # (n_gens,)
    sd_thresh:      np.ndarray   # (n_gens,)
    coop_rate:      np.ndarray   # (n_gens,)
    mean_inst_q:    np.ndarray   # (n_gens,)
    pathology:      np.ndarray   # (n_gens,)
    group_fitness:  np.ndarray   # (n_gens, n_groups)
    relatedness:    np.ndarray   # (n_gens,)
    bgs_eff:        np.ndarray   # (n_gens,)  effective BGS (rises if bgs_cultural_rise>0)
    tp_eff:         np.ndarray   # (n_gens,)  effective TP (evolves if tp_adapt>0)

    def final_regime(self) -> str:
        """Classify the equilibrium regime from the last 50 generations."""
        window    = slice(-50, None)
        cr_final  = self.coop_rate[window].mean()
        iq_final  = self.mean_inst_q[window].mean()
        pa_final  = self.pathology[window].mean()

        if cr_final > 0.70 and iq_final > 0.65:
            return "COMPLETE"
        elif pa_final > 0.45:
            return "PATHOLOGICAL"
        elif cr_final > 0.35:
            return "STALLED"
        else:
            return "DEFECTION"

    def summary_stats(self) -> dict:
        w = slice(-50, None)
        return {
            "regime":          self.final_regime(),
            "coop_final":      float(self.coop_rate[w].mean()),
            "inst_q_final":    float(self.mean_inst_q[w].mean()),
            "pathol_final":    float(self.pathology[w].mean()),
            "thresh_final":    float(self.mean_thresh[w].mean()),
            "relatedness_final": float(self.relatedness[w].mean()),
            "bgs_eff_final":   float(self.bgs_eff[w].mean()),
            "tp_eff_final":    float(self.tp_eff[w].mean()),
        }
