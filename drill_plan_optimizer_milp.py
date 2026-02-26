"""
Drill Plan Optimizer — MILP Model and Tkinter UI
=====================================
Optimal 2-D drill-plan solver using Pyomo + Gurobi.

MILP Formulation
----------------
Decision variables
  x_k ∈ {0,1}   — whether candidate hole k is selected

The fragmentation field at each evaluation grid cell g is:
  F_g = Σ_k  f_{g,k} · x_k          (linear in x_k)

where f_{g,k} = I1(g,k) + I2(g,k) is pre-computed.

Indicator linearisation
  A grid cell g is "fragmented"      if F_g > T
  A grid cell g is "over-fragmented" if F_g ≥ T_over

We introduce binary auxiliary variables:
  z_g  ∈ {0,1}   — 1 if cell g is fragmented      (F_g > T)
  v_g  ∈ {0,1}   — 1 if cell g is over-fragmented (F_g ≥ T_over)

Big-M linearisation:
  F_g ≥ T  + ε  − M(1 − z_g)    → z_g = 1 only if F_g > T
  F_g ≤ T        + M · z_g       → z_g = 0 only if F_g ≤ T

  F_g ≥ T_over   − M(1 − v_g)   → v_g = 1 only if F_g ≥ T_over
  F_g ≤ T_over − ε + M · v_g    → v_g = 0 only if F_g < T_over

Objective (maximise):
  Σ_g  z_g · Δ²  −  0.5 · Σ_g  v_g · Δ²
  = Δ² · (Σ_g z_g − 0.5 · Σ_g v_g)

Since Δ² is a constant multiplier the MILP maximises:
  Σ_g z_g − 0.5 Σ_g v_g

subject to:
  Σ_k f_{g,k} x_k  =  F_g             ∀ g  (substituted directly)
  z_g ≤ (F_g) / T_thresh              upper bound cuts
  Big-M constraints above
  x_k ∈ {0,1},  z_g ∈ {0,1},  v_g ∈ {0,1}

Grid reduction: to keep the MILP tractable we use DELTA=0.5 for the
optimisation grid and DELTA=0.1 only for the final metric evaluation.
Users can override via the --grid-delta CLI flag or the UI spinner.

Dependencies: pyomo, gurobipy (Gurobi licence required), numpy
"""

from __future__ import annotations
import json
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Physical / model constants
# ─────────────────────────────────────────────────────────────────────────────
A_CONST   = 0.75
C_CONST   = 0.55
T         = 1.0
T_OVER    = 1.8
DELTA_EVAL = 0.1    # fine grid for final metric reporting
LAMBDA    = 0.5
S         = 5.0
B         = 2.5
BOOTLEG_R = 0.75
EPS       = 1e-4    # strict-inequality offset

# Gaussian sigma values (derived from spec)
_log_term = math.log(2 * A_CONST / C_CONST)
σx1 = S / math.sqrt(8  * _log_term)
σy1 = B / math.sqrt(6  * _log_term)
σx2 = S / math.sqrt(32 * _log_term)
σy2 = B / math.sqrt(2  * _log_term)

# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def point_in_polygon(px: float, py: float, poly: list) -> bool:
    n, inside, j = len(poly), False, len(poly) - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def point_dist_to_polygon(px: float, py: float, poly: list) -> float:
    """Minimum distance from (px, py) to the nearest polygon edge."""
    min_dist = float('inf')
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            dist = math.hypot(px - ax, py - ay)
        else:
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
            dist = math.hypot(px - (ax + t * abx), py - (ay + t * aby))
        if dist < min_dist:
            min_dist = dist
    return min_dist


def points_in_polygon_mask(Xs: np.ndarray, Ys: np.ndarray, poly: list) -> np.ndarray:
    poly_arr = np.array(poly)
    n = len(poly_arr)
    mask = np.zeros(Xs.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly_arr[i]; xj, yj = poly_arr[j]
        cond = (poly_arr[i, 1] > Ys) != (poly_arr[j, 1] > Ys)
        x_int = (xj - xi) * (Ys - yi) / (yj - yi + 1e-12) + xi
        mask ^= cond & (Xs < x_int)
        j = i
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Fragmentation field helpers
# ─────────────────────────────────────────────────────────────────────────────

def hole_contribution(Xs: np.ndarray, Ys: np.ndarray, hx: float, hy: float) -> np.ndarray:
    """Return I1+I2 field for a single hole (same shape as Xs/Ys)."""
    dx, dy = Xs - hx, Ys - hy
    I1 = A_CONST * np.exp(-(dx**2 / (2*σx1**2) + dy**2 / (2*σy1**2)))
    I2 = A_CONST * np.exp(-(dx**2 / (2*σx2**2) + dy**2 / (2*σy2**2)))
    return I1 + I2


def compute_metrics(holes: list, delta: float, boundary: list) -> tuple:
    """Compute (Afrag, Aover, objective) on fine grid."""
    poly = np.array(boundary)
    xmin, ymin = poly.min(0); xmax, ymax = poly.max(0)
    gx = np.arange(xmin, xmax + delta, delta)
    gy = np.arange(ymin, ymax + delta, delta)
    Xs, Ys = np.meshgrid(gx, gy)
    mask = points_in_polygon_mask(Xs, Ys, boundary)
    F = np.zeros_like(Xs)
    for hx, hy in holes:
        F += hole_contribution(Xs, Ys, hx, hy)
    F = np.where(mask, F, 0.0)
    cell = delta**2
    Afrag = float(np.sum(F > T)      * cell)
    Aover = float(np.sum(F >= T_OVER) * cell)
    return Afrag, Aover, Afrag - LAMBDA * Aover


# ─────────────────────────────────────────────────────────────────────────────
# Candidate generation
# ─────────────────────────────────────────────────────────────────────────────

def build_candidates(boundary: list, bootlegs: list,
                     spacing_x: float = S, spacing_y: float = B,
                     boundary_margin: float = 0.0) -> list[tuple]:
    """Staggered spacing_x × spacing_y grid filtered to feasible interior positions.

    Parameters
    ----------
    spacing_x       : along-row hole spacing (m), default = S = 5.0
    spacing_y       : row-to-row spacing (m),     default = B = 2.5
    boundary_margin : minimum distance (m) from any boundary edge required
                      before a candidate is accepted (>0 keeps holes off the
                      boundary; 0 = legacy behaviour)
    """
    poly = np.array(boundary)
    xmin, ymin = poly.min(0); xmax, ymax = poly.max(0)
    candidates = []
    row = 0
    y = ymin + spacing_y / 2
    while y <= ymax - spacing_y / 2:
        x = xmin + spacing_x / 2 + (row % 2) * (spacing_x / 2)
        while x <= xmax - spacing_x / 2:
            if point_in_polygon(x, y, boundary):
                if boundary_margin > 0.0:
                    if point_dist_to_polygon(x, y, boundary) < boundary_margin:
                        x += spacing_x
                        continue
                if all(math.hypot(x-bx, y-by) >= BOOTLEG_R for bx, by in bootlegs):
                    candidates.append((x, y))
            x += spacing_x
        y += spacing_y; row += 1
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# MILP solver (Pyomo + Gurobi)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_gurobi_solver(log):
    """
    Try every known way to reach Gurobi from Pyomo, in order of preference:
      1. gurobipy native API (fastest, no file I/O)
      2. Pyomo gurobi_direct  (solver_io="python")
      3. Pyomo gurobi_persistent
      4. Pyomo shell interface (solver_io="lp")  ← works whenever gurobi_cl / gurobi is on PATH

    Returns a configured (solver, solver_io_mode) tuple, or raises RuntimeError.
    """
    from pyomo.environ import SolverFactory

    # ── attempt 1: pure gurobipy ────────────────────────────────────────────
    try:
        import gurobipy  # noqa: F401
        solver = SolverFactory("gurobi", solver_io="python")
        if solver.available():
            log("Solver backend: gurobipy (native Python API)")
            return solver, "python"
    except Exception as e:
        log(f"  gurobipy native: not available ({e})")

    # ── attempt 2: gurobi_direct via Pyomo plugin ───────────────────────────
    try:
        solver = SolverFactory("gurobi_direct")
        if solver.available():
            log("Solver backend: gurobi_direct (Pyomo plugin)")
            return solver, "direct"
    except Exception as e:
        log(f"  gurobi_direct: not available ({e})")

    # ── attempt 3: gurobi_persistent ────────────────────────────────────────
    try:
        solver = SolverFactory("gurobi_persistent")
        if solver.available():
            log("Solver backend: gurobi_persistent")
            return solver, "persistent"
    except Exception as e:
        log(f"  gurobi_persistent: not available ({e})")

    # ── attempt 4: shell / LP file (works if gurobi_cl or gurobi on PATH) ──
    try:
        solver = SolverFactory("gurobi", solver_io="lp")
        if solver.available():
            log("Solver backend: gurobi shell (LP file interface)")
            return solver, "lp"
    except Exception as e:
        log(f"  gurobi shell (lp): not available ({e})")

    # ── nothing worked ───────────────────────────────────────────────────────
    raise RuntimeError(
        "Cannot connect to Gurobi via any interface.\n\n"
        "Checklist:\n"
        "  1. Is Gurobi installed?  →  https://www.gurobi.com/downloads/\n"
        "  2. Is the licence active?  →  run: grbgetkey <your-key>\n"
        "  3. Is gurobipy installed for THIS Python?\n"
        "       python -m pip install gurobipy\n"
        "     (Gurobi 10+ ships gurobipy on PyPI; older versions need the\n"
        "      Gurobi installer to put it on PYTHONPATH manually.)\n"
        "  4. Is gurobi / gurobi_cl on your system PATH?\n"
        "       where gurobi_cl   (Windows)\n"
        "       which  gurobi_cl  (Linux/macOS)"
    )


def solve_milp(
    candidates: list,
    boundary:   list,
    bootlegs:   list,
    opt_delta:  float = 0.5,
    time_limit: float = 300.0,
    mip_gap:    float = 0.01,
    log_cb              = None,
) -> tuple[list, dict]:
    """
    Build and solve the MILP with Pyomo + Gurobi.

    Automatically selects the best available Gurobi interface:
      gurobipy  >  gurobi_direct  >  gurobi_persistent  >  shell (lp file)

    Returns
    -------
    holes : list of (x, y) for selected holes
    info  : dict with solver status, bounds, gap, etc.
    """
    try:
        from pyomo.environ import (
            ConcreteModel, Var, Binary, Constraint, Objective,
            maximize, value,
        )
    except ImportError as e:
        raise ImportError("Pyomo not found. Install with: pip install pyomo") from e

    def log(msg: str):
        if log_cb:
            log_cb(msg)

    # ── 1. Evaluation grid (coarser for MILP tractability) ─────────────────
    poly = np.array(boundary)
    xmin, ymin = poly.min(0); xmax, ymax = poly.max(0)
    gx = np.arange(xmin, xmax + opt_delta, opt_delta)
    gy = np.arange(ymin, ymax + opt_delta, opt_delta)
    Xs, Ys = np.meshgrid(gx, gy)
    inside = points_in_polygon_mask(Xs, Ys, boundary)

    grid_idx = [(i, j)
                for i in range(Ys.shape[0])
                for j in range(Xs.shape[1])
                if inside[i, j]]

    K = len(candidates)
    G = len(grid_idx)
    log(f"MILP: {K} binary hole vars × {G} grid cells → "
        f"{K + 2*G} binary vars, ~{3*G} constraints")

    # ── 2. Pre-compute f[g,k] coefficient matrix ────────────────────────────
    log("Pre-computing f[g,k] coefficient matrix …")
    t0 = time.perf_counter()

    gc_x = np.array([Xs[i, j] for i, j in grid_idx])   # (G,)
    gc_y = np.array([Ys[i, j] for i, j in grid_idx])
    hx_arr = np.array([c[0] for c in candidates])        # (K,)
    hy_arr = np.array([c[1] for c in candidates])

    dx = gc_x[:, None] - hx_arr[None, :]   # (G, K)
    dy = gc_y[:, None] - hy_arr[None, :]
    f_gk = (
        A_CONST * np.exp(-(dx**2 / (2*σx1**2) + dy**2 / (2*σy1**2))) +
        A_CONST * np.exp(-(dx**2 / (2*σx2**2) + dy**2 / (2*σy2**2)))
    )   # (G, K)

    M_g = f_gk.sum(axis=1)   # tight per-cell Big-M

    log(f"  … done in {time.perf_counter()-t0:.1f}s  "
        f"(max possible F = {M_g.max():.3f})")

    SPARSE_THR = 1e-6   # ignore negligible contributions

    # ── 3. Build Pyomo model ────────────────────────────────────────────────
    log("Building Pyomo model …")
    m = ConcreteModel()

    K_set = range(K)
    G_set = range(G)

    m.x = Var(K_set, domain=Binary)   # hole selection
    m.z = Var(G_set, domain=Binary)   # fragmented indicator
    m.v = Var(G_set, domain=Binary)   # over-fragmented indicator

    # Objective: maximise Σ z_g − 0.5 Σ v_g
    m.obj = Objective(
        expr=sum(m.z[g] for g in G_set) - LAMBDA * sum(m.v[g] for g in G_set),
        sense=maximize,
    )

    log("  Adding z-constraints (fragmentation) …")
    #
    # z_g = 1  iff  F_g > T      (binary indicator, maximised in objective)
    #
    # Both directions are needed:
    #
    #   (UB) Prevent z_g=1 when F_g ≤ T:
    #        F_g ≤ T + M_g · z_g
    #        → z_g=0 ⟹ F_g ≤ T   ✓
    #
    #   (LB) Prevent z_g=0 when F_g > T (force z_g=1):
    #        F_g ≥ (T+ε) − M_g·(1−z_g)
    #     →  F_g − M_g·z_g ≥ T+ε − M_g
    #        → z_g=1 ⟹ F_g ≥ T+ε  ✓
    #        → z_g=0: requires 0 ≥ T+ε − M_g, i.e. M_g ≥ T+ε  ← threshold!
    #
    # Pre-fix rules:
    #   M_g ≤ T      → cell can never exceed T    → z_g = 0, v_g = 0 (fixed)
    #   T < M_g ≤ T+ε → border cell, negligible  → z_g = 0, v_g = 0 (fixed)
    #   M_g > T+ε    → add both UB and LB constraints
    #
    m.z_ub = Constraint(G_set)
    m.z_lb = Constraint(G_set)
    for g in G_set:
        Mg = float(M_g[g])
        if Mg <= T + EPS:
            # Cell can't meaningfully exceed T → fix z=0, v=0
            m.z[g].fix(0); m.v[g].fix(0)
            continue
        Fg = sum(float(f_gk[g, k]) * m.x[k]
                 for k in K_set if f_gk[g, k] > SPARSE_THR)
        m.z_ub[g] = Fg <= T + Mg * m.z[g]               # z=0 ⟹ F ≤ T
        m.z_lb[g] = Fg - Mg * m.z[g] >= T + EPS - Mg    # z=1 ⟹ F ≥ T+ε

    log("  Adding v-constraints (over-fragmentation) …")
    #
    # v_g = 1  iff  F_g ≥ T_over   (binary indicator, PENALISED in objective)
    #
    # Both directions needed (objective minimises v, so Gurobi wants v=0):
    #
    #   (UB) Prevent v_g=1 when F_g < T_over:
    #        F_g ≤ (T_over−ε) + M_g · v_g
    #        → v_g=0 ⟹ F_g ≤ T_over−ε  ✓
    #
    #   (LB) Prevent v_g=0 when F_g ≥ T_over (force v_g=1):
    #        F_g − M_g·v_g ≥ T_over − M_g
    #        → v_g=0: requires 0 ≥ T_over−M_g, i.e. M_g ≥ T_over  ← threshold!
    #
    # Pre-fix: M_g < T_over → cell can never reach T_over → v_g = 0 (fixed)
    #
    m.v_ub = Constraint(G_set)
    m.v_lb = Constraint(G_set)
    for g in G_set:
        if m.z[g].is_fixed():
            continue                    # already fixed z=v=0
        Mg = float(M_g[g])
        if Mg < T_OVER:
            # Cell can never reach T_over → v_g = 0 always
            m.v[g].fix(0)
            continue
        Fg = sum(float(f_gk[g, k]) * m.x[k]
                 for k in K_set if f_gk[g, k] > SPARSE_THR)
        m.v_ub[g] = Fg <= (T_OVER - EPS) + Mg * m.v[g]  # v=0 ⟹ F < T_over
        m.v_lb[g] = Fg - Mg * m.v[g] >= T_OVER - Mg     # v=1 ⟹ F ≥ T_over

    # v_g ≤ z_g  (over-frag implies frag) — only for non-fixed vars
    m.v_implies_z = Constraint(G_set)
    for g in G_set:
        if not m.z[g].is_fixed() and not m.v[g].is_fixed():
            m.v_implies_z[g] = m.v[g] <= m.z[g]

    # Report stats
    n_z_fixed = sum(1 for g in G_set if m.z[g].is_fixed())
    n_v_fixed = sum(1 for g in G_set if m.v[g].is_fixed())
    n_v_full  = G - n_v_fixed
    log(f"  Cells fixed z=v=0 (M_g ≤ T+ε)        : {n_z_fixed}")
    log(f"  Cells fixed v=0   (M_g < T_over)      : {n_v_fixed - n_z_fixed}")
    log(f"  Cells with active z constraints        : {G - n_z_fixed}")
    log(f"  Cells with active v constraints        : {n_v_full}")

    # ── 4. Select solver backend ─────────────────────────────────────────────
    log("Detecting Gurobi interface …")
    solver, backend = _pick_gurobi_solver(log)

    # Common options (work across all backends)
    solver.options["TimeLimit"] = time_limit
    solver.options["MIPGap"]    = mip_gap
    solver.options["Threads"]   = 0
    solver.options["Presolve"]  = 2
    solver.options["MIPFocus"]  = 1
    solver.options["Cuts"]      = 2
    # Suppress Gurobi's own stdout (we capture via Pyomo tee)
    solver.options["OutputFlag"] = 0

    log("Model built. Launching Gurobi …")

    # ── 5. Solve ─────────────────────────────────────────────────────────────
    results = solver.solve(m, tee=False)

    # ── 6. Extract solution ───────────────────────────────────────────────────
    # Log raw Pyomo result for diagnostics (helpful when status parsing varies)
    try:
        log(f"Raw solver result  : {results.solver}")
    except Exception:
        pass

    # Pyomo's SolverStatus changed from a plain string-keyed object to an
    # enum across versions.  Always go through str() to be safe.
    status = str(results.solver.status)
    term   = str(results.solver.termination_condition)

    log(f"Solver status      : {status}")
    log(f"Termination        : {term}")

    # Safe objective read — wrapped in try/except for all Pyomo versions
    obj_val = None
    try:
        obj_val = float(value(m.obj))
        log(f"Objective (opt Δ)  : {obj_val:.4f}")
    except Exception:
        log("  (objective value not available)")

    # Accept any result that carried a feasible incumbent.
    # Covers: optimal, time-limit with incumbent, interrupted, etc.
    ok_terms = {
        "optimal", "feasible",
        "maxTimeLimit", "maxIterations",
        "abortedOnSizeLimit", "intermediateNonInteger",
        "SolverStatus.ok", "ok",
    }
    ok_status = {"ok", "SolverStatus.ok", "warning"}

    has_solution = (
        status.lower().rstrip(".").split(".")[-1] in {"ok", "warning"}
        or any(t in term for t in ("optimal", "feasible", "maxTimeLimit",
                                   "Intermediate", "aborted"))
    )

    selected_holes = []
    if has_solution:
        for k in K_set:
            try:
                v = value(m.x[k])
                if v is not None and v > 0.5:
                    selected_holes.append(candidates[k])
            except Exception:
                pass
    else:
        log(f"  No feasible solution found (status={status}, term={term})")

    info = {
        "status":       status,
        "termination":  term,
        "milp_obj":     obj_val,
        "backend":      backend,
        "n_holes":      len(selected_holes),
        "n_candidates": K,
        "n_grid_cells": G,
        "opt_delta":    opt_delta,
    }
    return selected_holes, info


# ─────────────────────────────────────────────────────────────────────────────
# Tkinter UI
# ─────────────────────────────────────────────────────────────────────────────

class MILPOptimizerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Drill Plan Optimizer — MILP (Pyomo + Gurobi)")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e2e")

        self.boundary: list  = []
        self.bootlegs: list  = []
        self.holes:    list  = []

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        s = ttk.Style(); s.theme_use("clam")
        s.configure("TFrame",   background="#1e1e2e")
        s.configure("TLabel",   background="#1e1e2e", foreground="#cdd6f4",
                    font=("Segoe UI", 10))
        s.configure("TButton",  background="#89b4fa", foreground="#1e1e2e",
                    font=("Segoe UI", 10, "bold"), padding=6)
        s.map("TButton", background=[("active", "#74c7ec")])
        s.configure("TLabelframe",       background="#1e1e2e", foreground="#89b4fa",
                    font=("Segoe UI", 10, "bold"))
        s.configure("TLabelframe.Label", background="#1e1e2e", foreground="#89b4fa",
                    font=("Segoe UI", 10, "bold"))
        s.configure("TSpinbox", fieldbackground="#313244", foreground="#cdd6f4",
                    background="#313244")
        s.configure("TEntry",   fieldbackground="#313244", foreground="#cdd6f4")

        # ── Toolbar ──────────────────────────────────────────────────────────
        tb = ttk.Frame(self.root)
        tb.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Button(tb, text="📂  Load JSON",       command=self.load_json).pack(side=tk.LEFT, padx=4)
        ttk.Button(tb, text="▶  Solve MILP",       command=self.run_milp).pack(side=tk.LEFT, padx=4)
        ttk.Button(tb, text="💾  Export Results",  command=self.export_results).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Load a JSON file to begin.")
        ttk.Label(tb, textvariable=self.status_var,
                  font=("Segoe UI", 9, "italic")).pack(side=tk.LEFT, padx=16)

        # ── Main ─────────────────────────────────────────────────────────────
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        # Canvas
        cf = tk.Frame(main, bg="#181825", relief=tk.SUNKEN, bd=2)
        cf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(cf, bg="#181825", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        # Right panel
        rp = ttk.Frame(main, width=260)
        rp.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        rp.pack_propagate(False)

        # ── Solver settings ───────────────────────────────────────────────
        sf = ttk.LabelFrame(rp, text="Solver Settings", padding=8)
        sf.pack(fill=tk.X, pady=(0, 8))

        def _row(parent, label, widget_factory, **kw):
            f = tk.Frame(parent, bg="#1e1e2e")
            f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, bg="#1e1e2e", fg="#a6adc8",
                     font=("Segoe UI", 9), width=14, anchor=tk.W).pack(side=tk.LEFT)
            w = widget_factory(f, **kw)
            w.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            return w

        self.delta_var = tk.DoubleVar(value=0.5)
        _row(sf, "Opt. grid Δ (m)",
             ttk.Spinbox,
             textvariable=self.delta_var, from_=0.2, to=2.0, increment=0.1,
             width=8, format="%.1f")

        self.timelimit_var = tk.IntVar(value=300)
        _row(sf, "Time limit (s)",
             ttk.Spinbox,
             textvariable=self.timelimit_var, from_=10, to=3600, increment=30,
             width=8)

        self.gap_var = tk.DoubleVar(value=0.01)
        _row(sf, "MIP gap",
             ttk.Spinbox,
             textvariable=self.gap_var, from_=0.001, to=0.1, increment=0.005,
             width=8, format="%.3f")

        # ── Candidate settings ────────────────────────────────────────────
        cf2 = ttk.LabelFrame(rp, text="Candidate Settings", padding=8)
        cf2.pack(fill=tk.X, pady=(0, 8))

        self.spacing_x_var = tk.DoubleVar(value=S)
        _row(cf2, "Spacing S (m)",
             ttk.Spinbox,
             textvariable=self.spacing_x_var, from_=0.5, to=20.0, increment=0.5,
             width=8, format="%.1f")

        self.spacing_y_var = tk.DoubleVar(value=B)
        _row(cf2, "Spacing B (m)",
             ttk.Spinbox,
             textvariable=self.spacing_y_var, from_=0.5, to=20.0, increment=0.5,
             width=8, format="%.1f")

        self.margin_var = tk.DoubleVar(value=0.0)
        _row(cf2, "Boundary margin (m)",
             ttk.Spinbox,
             textvariable=self.margin_var, from_=0.0, to=10.0, increment=0.25,
             width=8, format="%.2f")

        # ── Metrics ───────────────────────────────────────────────────────
        mf = ttk.LabelFrame(rp, text="Results", padding=8)
        mf.pack(fill=tk.X, pady=(0, 8))

        self.metric_labels: dict[str, tk.Label] = {}
        for key, label in [
            ("n_holes",   "Holes placed"),
            ("afrag",     "A_frag (m²)"),
            ("aover",     "A_over (m²)"),
            ("obj",       "Objective"),
            ("milp_obj",  "MILP obj (opt Δ)"),
            ("status",    "Solver status"),
            ("gap",       "MIP gap"),
        ]:
            row = tk.Frame(mf, bg="#313244", pady=2)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label + ":", bg="#313244", fg="#a6adc8",
                     font=("Segoe UI", 9), width=16, anchor=tk.W).pack(side=tk.LEFT, padx=4)
            v = tk.Label(row, text="—", bg="#313244", fg="#cdd6f4",
                         font=("Segoe UI", 9, "bold"), anchor=tk.E)
            v.pack(side=tk.RIGHT, padx=4)
            self.metric_labels[key] = v

        # ── Log ───────────────────────────────────────────────────────────
        ttk.Label(rp, text="Log", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(4, 2))
        lf = tk.Frame(rp, bg="#181825")
        lf.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(lf, bg="#181825", fg="#a6e3a1",
                                font=("Consolas", 8), wrap=tk.WORD,
                                state=tk.DISABLED, relief=tk.FLAT)
        sb = ttk.Scrollbar(lf, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 4))

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        def _do():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        self.root.after(0, _do)

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _set_metric(self, key: str, val: str):
        self.root.after(0, lambda: self.metric_labels[key].config(text=val))

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load_json(self):
        path = filedialog.askopenfilename(
            title="Select input JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self.boundary = [tuple(p) for p in data["boundary"]]
            # Deduplicate bootlegs
            seen: set = set(); bl = []
            for p in data["bootlegs"]:
                key = (round(p[0], 4), round(p[1], 4))
                if key not in seen:
                    seen.add(key); bl.append(tuple(p))
            self.bootlegs = bl
            self.holes = []
            self._log(f"Loaded: {path}")
            self._log(f"  Boundary vertices : {len(self.boundary)}")
            self._log(f"  Unique bootlegs   : {len(self.bootlegs)}")
            self._set_status(f"Loaded — {path.split('/')[-1]}")
            self._redraw()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def export_results(self):
        if not self.holes:
            messagebox.showwarning("No results", "Run the solver first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save holes JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return
        with open(path, "w") as f:
            json.dump({"holes": [list(h) for h in self.holes]}, f, indent=2)
        self._log(f"Exported {len(self.holes)} holes → {path}")
        self._set_status(f"Exported → {path.split('/')[-1]}")

    # ── Solver thread ─────────────────────────────────────────────────────────

    def run_milp(self):
        if not self.boundary:
            messagebox.showwarning("No data", "Load a JSON file first.")
            return
        self.progress.start(10)
        self._set_status("Solving MILP …")
        threading.Thread(target=self._solve_thread, daemon=True).start()

    def _solve_thread(self):
        try:
            boundary        = self.boundary
            bootlegs        = self.bootlegs
            opt_delta       = float(self.delta_var.get())
            tlim            = float(self.timelimit_var.get())
            mip_gap         = float(self.gap_var.get())
            spacing_x       = float(self.spacing_x_var.get())
            spacing_y       = float(self.spacing_y_var.get())
            boundary_margin = float(self.margin_var.get())

            self._log("═" * 40)
            self._log("MILP Drill Plan Optimizer (Pyomo + Gurobi)")
            self._log(f"  Opt grid Δ : {opt_delta} m")
            self._log(f"  Time limit : {tlim} s")
            self._log(f"  MIP gap    : {mip_gap}")
            self._log(f"  Spacing    : S={spacing_x} m  B={spacing_y} m")
            self._log(f"  Bdy margin : {boundary_margin} m")
            self._log("─" * 40)

            # Candidate generation
            self._log("Generating candidate positions …")
            candidates = build_candidates(boundary, bootlegs,
                                          spacing_x, spacing_y, boundary_margin)
            self._log(f"  {len(candidates)} feasible candidates")
            if not candidates:
                self._log("ERROR: no feasible candidates found!")
                self.root.after(0, self.progress.stop)
                return

            # Solve
            t_start = time.perf_counter()
            holes, info = solve_milp(
                candidates, boundary, bootlegs,
                opt_delta=opt_delta,
                time_limit=tlim,
                mip_gap=mip_gap,
                log_cb=self._log,
            )
            elapsed = time.perf_counter() - t_start

            self.holes = holes
            self._log("─" * 40)
            self._log(f"Solve time : {elapsed:.1f} s")
            self._log(f"Holes selected : {len(holes)}")

            # Fine-grid metrics
            if holes:
                self._log(f"Computing fine-grid metrics (Δ={DELTA_EVAL}) …")
                Afrag, Aover, obj = compute_metrics(holes, DELTA_EVAL, boundary)
                self._log(f"  A_frag    = {Afrag:.2f} m²")
                self._log(f"  A_over    = {Aover:.2f} m²")
                self._log(f"  Objective = {obj:.4f}")
                self._set_metric("n_holes",  str(len(holes)))
                self._set_metric("afrag",    f"{Afrag:.2f}")
                self._set_metric("aover",    f"{Aover:.2f}")
                self._set_metric("obj",      f"{obj:.4f}")
            else:
                Afrag = Aover = obj = 0.0
                self._log("No holes selected — check solver status.")

            self._set_metric("milp_obj", f"{info.get('milp_obj', '—')}")
            self._set_metric("status",   info.get("status", "?"))

            self.root.after(0, self._redraw)
            self.root.after(0, self.progress.stop)
            self._set_status(
                f"Done — {len(holes)} holes | obj={obj:.2f} | {elapsed:.0f}s"
            )

        except (ImportError, RuntimeError) as e:
            self._log(f"\n{'!'*40}")
            for line in str(e).splitlines():
                self._log(line)
            self._log("─" * 40)
            self._log("Quick fix for most cases:")
            self._log("  pip install pyomo gurobipy")
            self._log("  grbgetkey <your-licence-key>")
            self._log("Free academic licence: https://www.gurobi.com/academia/")
            self.root.after(0, self.progress.stop)
            self._set_status("Gurobi connection error — see log.")
        except Exception:
            import traceback
            self._log("ERROR:\n" + traceback.format_exc())
            self.root.after(0, self.progress.stop)
            self._set_status("Solver error — see log.")

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _redraw(self):
        if not self.boundary:
            return
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        poly = np.array(self.boundary)
        xmin, ymin = poly.min(0); xmax, ymax = poly.max(0)
        pad = 44
        sx = (w - 2*pad) / max(xmax - xmin, 1e-6)
        sy = (h - 2*pad) / max(ymax - ymin, 1e-6)
        sc = min(sx, sy)
        ox = pad + ((w - 2*pad) - (xmax-xmin)*sc) / 2
        oy = pad + ((h - 2*pad) - (ymax-ymin)*sc) / 2

        def to_c(x, y):
            return ox + (x-xmin)*sc, oy + (ymax-y)*sc

        # Boundary fill
        pts = [to_c(x, y) for x, y in self.boundary]
        flat = [v for p in pts for v in p]
        self.canvas.create_polygon(flat, outline="#89b4fa", fill="#1e1e3e", width=2)

        # Bootleg exclusion circles
        r_px = BOOTLEG_R * sc
        for bx, by in self.bootlegs:
            cx, cy = to_c(bx, by)
            self.canvas.create_oval(cx-r_px, cy-r_px, cx+r_px, cy+r_px,
                                    outline="#f38ba8", fill="#2d0011", width=1)

        # Candidate grid (light dots)
        if self.boundary and not self.holes:
            cands = build_candidates(
                self.boundary, self.bootlegs,
                float(self.spacing_x_var.get()),
                float(self.spacing_y_var.get()),
                float(self.margin_var.get()),
            )
            for hx, hy in cands:
                cx, cy = to_c(hx, hy)
                self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2,
                                        fill="#45475a", outline="")

        # Drill holes
        r_h = max(3, int(sc * 0.3))
        for hx, hy in self.holes:
            cx, cy = to_c(hx, hy)
            self.canvas.create_oval(cx-r_h, cy-r_h, cx+r_h, cy+r_h,
                                    fill="#a6e3a1", outline="#40a02b", width=1)
            # centre dot
            self.canvas.create_oval(cx-1, cy-1, cx+1, cy+1,
                                    fill="#40a02b", outline="")

        # Axis labels
        self.canvas.create_text(w//2, h-6, text=f"X (m)  [{xmin:.0f} … {xmax:.0f}]",
                                fill="#6c7086", font=("Segoe UI", 8))
        self.canvas.create_text(8, h//2, text="Y", fill="#6c7086",
                                font=("Segoe UI", 8), angle=90)

        # Legend
        items = [
            ("#89b4fa", "Boundary polygon"),
            ("#f38ba8", f"Bootleg zones ×{len(self.bootlegs)} (r=0.75 m)"),
            ("#a6e3a1", f"Drill holes × {len(self.holes)}"),
            ("#45475a", "Candidates (pre-solve)"),
        ]
        lx, ly = 12, 12
        for col, lbl in items:
            self.canvas.create_rectangle(lx, ly, lx+12, ly+12, fill=col, outline=col)
            self.canvas.create_text(lx+18, ly+6, text=lbl, fill="#cdd6f4",
                                    font=("Segoe UI", 8), anchor=tk.W)
            ly += 18


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (for headless / CI use)
# ─────────────────────────────────────────────────────────────────────────────

def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Drill plan MILP optimizer (CLI mode, no GUI)")
    parser.add_argument("input",  help="Input JSON path")
    parser.add_argument("output", help="Output JSON path")
    parser.add_argument("--delta",      type=float, default=0.5,
                        help="Optimisation grid resolution (default 0.5 m)")
    parser.add_argument("--time-limit", type=float, default=300.0,
                        help="Gurobi time limit in seconds (default 300)")
    parser.add_argument("--mip-gap",    type=float, default=0.01,
                        help="Gurobi MIP gap tolerance (default 0.01)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    boundary = [tuple(p) for p in data["boundary"]]
    seen: set = set(); bootlegs = []
    for p in data["bootlegs"]:
        key = (round(p[0], 4), round(p[1], 4))
        if key not in seen:
            seen.add(key); bootlegs.append(tuple(p))

    print(f"Boundary: {len(boundary)} vertices | Bootlegs: {len(bootlegs)}")
    candidates = build_candidates(boundary, bootlegs)
    print(f"Candidates: {len(candidates)}")

    holes, info = solve_milp(
        candidates, boundary, bootlegs,
        opt_delta=args.delta,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        log_cb=print,
    )

    Afrag, Aover, obj = compute_metrics(holes, DELTA_EVAL, boundary)
    print(f"\nResults:")
    print(f"  Holes     : {len(holes)}")
    print(f"  A_frag    : {Afrag:.2f} m²")
    print(f"  A_over    : {Aover:.2f} m²")
    print(f"  Objective : {obj:.4f}")

    with open(args.output, "w") as f:
        json.dump({"holes": [list(h) for h in holes]}, f, indent=2)
    print(f"Saved → {args.output}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — GUI by default, CLI if --input flag present
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-h"):
        cli_main()
    else:
        root = tk.Tk()
        app = MILPOptimizerApp(root)
        root.mainloop()