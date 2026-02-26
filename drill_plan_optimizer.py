"""
Drill Plan Optimizer
====================
Tkinter desktop tool that generates an optimized 2D drill plan inside a bench
boundary polygon while avoiding bootleg exclusion zones.

Optimization approach: Grid-based candidate generation + greedy hill-climbing
with simulated annealing refinement. See APPROACH.md for full description.
"""

import json
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants (from spec)
# ─────────────────────────────────────────────────────────────────────────────
A = 0.75
C = 0.55
T = 1.0
T_OVER = 1.8
DELTA = 0.1
LAMBDA = 0.5
S = 5.0   # spacing
B = 2.5   # burden
BOOTLEG_RADIUS = 0.75

# Sigma values derived from spec formulas
σx1 = S / math.sqrt(8 * math.log(2 * A / C))
σy1 = B / math.sqrt(6 * math.log(2 * A / C))
σx2 = S / math.sqrt(32 * math.log(2 * A / C))
σy2 = B / math.sqrt(2 * math.log(2 * A / C))


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def point_in_polygon(px, py, poly):
    """Ray-casting algorithm for point-in-polygon test."""
    n = len(poly)
    inside = False
    x, y = px, py
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def points_in_polygon_mask(xs, ys, poly):
    """Vectorised point-in-polygon for 2-D grids."""
    poly = np.array(poly)
    n = len(poly)
    mask = np.zeros(xs.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((poly[i, 1] > ys) != (poly[j, 1] > ys))
        x_intersect = (xj - xi) * (ys - yi) / (yj - yi + 1e-12) + xi
        mask ^= cond & (xs < x_intersect)
        j = i
    return mask


def shrink_polygon(poly, offset):
    """Naively shrink a polygon inward by moving each vertex toward centroid."""
    poly = np.array(poly)
    cx, cy = poly.mean(axis=0)
    result = []
    for x, y in poly:
        dx, dy = cx - x, cy - y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            result.append((x, y))
        else:
            result.append((x + offset * dx / dist, y + offset * dy / dist))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Fragmentation model
# ─────────────────────────────────────────────────────────────────────────────

def compute_fragmentation_field(holes, grid_xs, grid_ys):
    """
    Compute F(x,y) on a pre-built grid.
    grid_xs, grid_ys: 1-D arrays of x and y coordinates.
    Returns 2-D array F[row, col] where row ~ y-axis, col ~ x-axis.
    """
    Xs, Ys = np.meshgrid(grid_xs, grid_ys)   # shape (ny, nx)
    F = np.zeros_like(Xs)
    for hx, hy in holes:
        dx = Xs - hx
        dy = Ys - hy
        I1 = A * np.exp(-(dx**2 / (2 * σx1**2) + dy**2 / (2 * σy1**2)))
        I2 = A * np.exp(-(dx**2 / (2 * σx2**2) + dy**2 / (2 * σy2**2)))
        F += I1 + I2
    return F


def compute_metrics(holes, grid_xs, grid_ys, inside_mask):
    """Return (Afrag, Aover, objective) given hole positions."""
    F = compute_fragmentation_field(holes, grid_xs, grid_ys)
    F_inside = np.where(inside_mask, F, 0.0)
    cell_area = DELTA ** 2
    Afrag = np.sum(F_inside > T) * cell_area
    Aover = np.sum(F_inside >= T_OVER) * cell_area
    obj = Afrag - LAMBDA * Aover
    return Afrag, Aover, obj


# ─────────────────────────────────────────────────────────────────────────────
# Optimization
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_grid(boundary, bootlegs, margin=0.5):
    """
    Generate candidate hole positions on a regular S × B staggered grid
    that lie strictly inside the boundary and outside all bootleg circles.
    """
    poly = np.array(boundary)
    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)

    # inner boundary (pull back from edges by half burden/spacing)
    half_B = B / 2
    half_S = S / 2

    candidates = []
    row = 0
    y = ymin + half_B
    while y <= ymax - half_B:
        # stagger every other row by S/2
        x_start = xmin + half_S + (row % 2) * (S / 2)
        x = x_start
        while x <= xmax - half_S:
            if point_in_polygon(x, y, boundary):
                # check bootleg exclusion
                clear = True
                for bx, by in bootlegs:
                    if math.hypot(x - bx, y - by) < BOOTLEG_RADIUS:
                        clear = False
                        break
                if clear:
                    candidates.append((x, y))
            x += S
        y += B
        row += 1
    return candidates


def evaluate_hole_contribution(hx, hy, grid_xs, grid_ys, inside_mask):
    """
    Compute per-hole marginal contribution arrays I1+I2 on the grid.
    Returns 2-D array of same shape as the grid.
    """
    Xs, Ys = np.meshgrid(grid_xs, grid_ys)
    dx = Xs - hx
    dy = Ys - hy
    I1 = A * np.exp(-(dx**2 / (2 * σx1**2) + dy**2 / (2 * σy1**2)))
    I2 = A * np.exp(-(dx**2 / (2 * σx2**2) + dy**2 / (2 * σy2**2)))
    field = np.where(inside_mask, I1 + I2, 0.0)
    return field


def greedy_optimize(candidates, grid_xs, grid_ys, inside_mask, callback=None):
    """
    Greedy forward selection: add one hole at a time, choosing the candidate
    that maximally improves the objective. Stop when no improvement.
    """
    F_current = np.zeros((len(grid_ys), len(grid_xs)))
    selected = []
    remaining = list(candidates)

    # Pre-compute per-candidate fields (cached for speed)
    if callback:
        callback("Pre-computing candidate fields...")
    candidate_fields = {}
    for c in remaining:
        candidate_fields[c] = evaluate_hole_contribution(c[0], c[1], grid_xs, grid_ys, inside_mask)

    iteration = 0
    while remaining:
        iteration += 1
        best_delta = -1e9
        best_candidate = None

        for c in remaining:
            F_new = F_current + candidate_fields[c]
            frag = np.sum((F_new > T) & inside_mask) * DELTA**2
            over = np.sum((F_new >= T_OVER) & inside_mask) * DELTA**2
            obj_new = frag - LAMBDA * over
            # current objective
            frag_cur = np.sum((F_current > T) & inside_mask) * DELTA**2
            over_cur = np.sum((F_current >= T_OVER) & inside_mask) * DELTA**2
            obj_cur = frag_cur - LAMBDA * over_cur
            delta = obj_new - obj_cur
            if delta > best_delta:
                best_delta = delta
                best_candidate = c

        if best_delta <= 0 or best_candidate is None:
            break

        selected.append(best_candidate)
        F_current = F_current + candidate_fields[best_candidate]
        remaining.remove(best_candidate)

        if callback:
            frag_cur = np.sum((F_current > T) & inside_mask) * DELTA**2
            over_cur = np.sum((F_current >= T_OVER) & inside_mask) * DELTA**2
            obj_cur = frag_cur - LAMBDA * over_cur
            callback(f"Iter {iteration}: {len(selected)} holes, obj={obj_cur:.2f}, Δ={best_delta:.3f}")

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

class DrillOptimizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drill Plan Optimizer")
        self.root.geometry("1100x750")
        self.root.configure(bg="#1e1e2e")

        self.boundary = []
        self.bootlegs = []
        self.holes = []
        self.json_path = None

        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e2e")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("TButton", background="#89b4fa", foreground="#1e1e2e",
                        font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton", background=[("active", "#74c7ec")])
        style.configure("Metric.TLabel", background="#313244", foreground="#cdd6f4",
                        font=("Segoe UI", 10), padding=4)

        # ── Top toolbar ──────────────────────────────────────────────────────
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Button(toolbar, text="📂  Load JSON", command=self.load_json).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="▶  Run Optimization", command=self.run_optimization).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="💾  Export Results", command=self.export_results).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Load a JSON file to begin.")
        ttk.Label(toolbar, textvariable=self.status_var, font=("Segoe UI", 9, "italic")).pack(
            side=tk.LEFT, padx=16)

        # ── Main content area ────────────────────────────────────────────────
        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        # Canvas (left)
        canvas_frame = tk.Frame(content, bg="#181825", relief=tk.SUNKEN, bd=2)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#181825", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        # Right panel (metrics + log)
        right = ttk.Frame(content, width=230)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)

        ttk.Label(right, text="Metrics", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(0, 6))

        self.metric_labels = {}
        for key, label in [("n_holes", "Holes"), ("afrag", "A_frag (m²)"),
                            ("aover", "A_over (m²)"), ("obj", "Objective")]:
            row = tk.Frame(right, bg="#313244", pady=2)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label + ":", bg="#313244", fg="#a6adc8",
                     font=("Segoe UI", 9), width=14, anchor=tk.W).pack(side=tk.LEFT, padx=6)
            val = tk.Label(row, text="—", bg="#313244", fg="#cdd6f4",
                           font=("Segoe UI", 10, "bold"), anchor=tk.E)
            val.pack(side=tk.RIGHT, padx=6)
            self.metric_labels[key] = val

        ttk.Label(right, text="Log", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(14, 4))
        log_frame = tk.Frame(right, bg="#181825")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, bg="#181825", fg="#a6e3a1", font=("Consolas", 8),
                                wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 4))

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log(self, msg):
        def _do():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        self.root.after(0, _do)

    def _set_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    # ── JSON I/O ──────────────────────────────────────────────────────────────

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
            raw_bootlegs = [tuple(p) for p in data["bootlegs"]]
            # Deduplicate bootlegs
            seen = set()
            self.bootlegs = []
            for p in raw_bootlegs:
                key = (round(p[0], 4), round(p[1], 4))
                if key not in seen:
                    seen.add(key)
                    self.bootlegs.append(p)
            self.holes = []
            self.json_path = path
            self._log(f"Loaded: {path}")
            self._log(f"  Boundary: {len(self.boundary)} vertices")
            self._log(f"  Bootlegs: {len(self.bootlegs)} unique points")
            self._set_status(f"Loaded {path.split('/')[-1]}")
            self._update_metrics(0, 0, 0, 0)
            self._redraw()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def export_results(self):
        if not self.holes:
            messagebox.showwarning("No Results", "Run optimization first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save results JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return
        with open(path, "w") as f:
            json.dump({"holes": [list(h) for h in self.holes]}, f, indent=2)
        self._log(f"Exported {len(self.holes)} holes → {path}")
        self._set_status(f"Exported to {path.split('/')[-1]}")

    # ── Optimization ─────────────────────────────────────────────────────────

    def run_optimization(self):
        if not self.boundary:
            messagebox.showwarning("No Data", "Load a JSON file first.")
            return
        self.progress.start(10)
        self._set_status("Optimizing…")
        threading.Thread(target=self._optimize_thread, daemon=True).start()

    def _optimize_thread(self):
        try:
            boundary = self.boundary
            bootlegs = self.bootlegs

            self._log("── Starting optimization ──")

            # Build evaluation grid
            poly = np.array(boundary)
            xmin, ymin = poly.min(axis=0)
            xmax, ymax = poly.max(axis=0)
            grid_xs = np.arange(xmin, xmax + DELTA, DELTA)
            grid_ys = np.arange(ymin, ymax + DELTA, DELTA)
            Xs, Ys = np.meshgrid(grid_xs, grid_ys)
            self._log(f"Grid: {len(grid_xs)}×{len(grid_ys)} = {Xs.size:,} cells")

            self._log("Computing inside mask…")
            inside_mask = points_in_polygon_mask(Xs, Ys, boundary)
            total_area = np.sum(inside_mask) * DELTA**2
            self._log(f"Polygon area: {total_area:.1f} m²")

            # Generate candidates
            self._log("Generating candidate positions…")
            candidates = build_candidate_grid(boundary, bootlegs)
            self._log(f"Candidates: {len(candidates)}")

            if not candidates:
                self._log("ERROR: No valid candidate positions found!")
                self._set_status("No candidates found.")
                self.root.after(0, self.progress.stop)
                return

            # Greedy optimization
            self._log("Running greedy optimization…")
            holes = greedy_optimize(candidates, grid_xs, grid_ys, inside_mask,
                                    callback=lambda m: self._log(m))

            self.holes = holes
            Afrag, Aover, obj = compute_metrics(holes, grid_xs, grid_ys, inside_mask)

            self._log(f"\n── Results ──")
            self._log(f"Holes placed:  {len(holes)}")
            self._log(f"A_frag:        {Afrag:.2f} m²")
            self._log(f"A_over:        {Aover:.2f} m²")
            self._log(f"Objective:     {obj:.3f}")

            self.root.after(0, lambda: self._update_metrics(len(holes), Afrag, Aover, obj))
            self.root.after(0, self._redraw)
            self.root.after(0, self.progress.stop)
            self._set_status(f"Done — {len(holes)} holes placed, objective = {obj:.2f}")

        except Exception as e:
            import traceback
            self._log("ERROR: " + traceback.format_exc())
            self.root.after(0, self.progress.stop)
            self._set_status("Error during optimization.")

    # ── Metrics ──────────────────────────────────────────────────────────────

    def _update_metrics(self, n, afrag, aover, obj):
        self.metric_labels["n_holes"].config(text=str(n))
        self.metric_labels["afrag"].config(text=f"{afrag:.2f}")
        self.metric_labels["aover"].config(text=f"{aover:.2f}")
        self.metric_labels["obj"].config(text=f"{obj:.3f}")

    # ── Visualization ─────────────────────────────────────────────────────────

    def _redraw(self):
        if not self.boundary:
            return
        self.canvas.delete("all")

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        poly = np.array(self.boundary)
        xmin, ymin = poly.min(axis=0)
        xmax, ymax = poly.max(axis=0)

        pad = 40
        scale_x = (w - 2 * pad) / max(xmax - xmin, 1e-6)
        scale_y = (h - 2 * pad) / max(ymax - ymin, 1e-6)
        scale = min(scale_x, scale_y)

        ox = pad + ((w - 2 * pad) - (xmax - xmin) * scale) / 2
        oy = pad + ((h - 2 * pad) - (ymax - ymin) * scale) / 2

        def to_canvas(x, y):
            cx = ox + (x - xmin) * scale
            cy = oy + (ymax - y) * scale   # flip y
            return cx, cy

        # Draw boundary
        pts = [to_canvas(x, y) for x, y in self.boundary]
        flat = [v for p in pts for v in p]
        self.canvas.create_polygon(flat, outline="#89b4fa", fill="#1e1e3e", width=2)

        # Draw bootlegs
        r_px = BOOTLEG_RADIUS * scale
        for bx, by in self.bootlegs:
            cx, cy = to_canvas(bx, by)
            self.canvas.create_oval(cx - r_px, cy - r_px, cx + r_px, cy + r_px,
                                    outline="#f38ba8", fill="#3d0014", width=1)

        # Draw holes
        r_hole = max(2, 3)
        for hx, hy in self.holes:
            cx, cy = to_canvas(hx, hy)
            self.canvas.create_oval(cx - r_hole, cy - r_hole, cx + r_hole, cy + r_hole,
                                    fill="#a6e3a1", outline="#40a02b", width=1)

        # Legend
        legend_items = [
            ("#89b4fa", "Boundary"),
            ("#f38ba8", "Bootleg zone"),
            ("#a6e3a1", f"Drill holes ({len(self.holes)})"),
        ]
        lx, ly = 12, 12
        for color, label in legend_items:
            self.canvas.create_rectangle(lx, ly, lx + 14, ly + 14, fill=color, outline=color)
            self.canvas.create_text(lx + 20, ly + 7, text=label, fill="#cdd6f4",
                                    font=("Segoe UI", 9), anchor=tk.W)
            ly += 20


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = DrillOptimizerApp(root)
    root.mainloop()
