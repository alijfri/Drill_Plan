# Drill Plan Optimizer

A desktop tool (Python + Tkinter) that generates an optimized 2D drill plan inside a bench boundary polygon while avoiding bootleg exclusion zones.

## Requirements

### Basic Optimizer
- Python 3.8+
- numpy
- scipy (optional)

### MILP Optimizer (Recommended for optimal solutions)
- Python 3.8+
- numpy>=1.20.0
- pyomo>=6.0.0
- gurobipy>=9.0.0 
- tkinter (bundled with Python on macOS/Windows; install separately on Linux)

Install MILP dependencies:
```bash
pip install -r requirements.txt
```


## Running

### MILP Optimizer (Recommended)
```bash
python drill_plan_optimizer_milp.py
```

This uses a **Mixed Integer Linear Programming (MILP)** formulation with Pyomo + Gurobi for globally optimal solutions.

### Basic Optimizer
```bash
python drill_plan_optimizer.py
```

Usage for both:
1. Click **Load JSON** and select your input file (e.g., `input.json`)
2. Click **Run Optimization** — progress is logged in the right panel
3. View the visualization: boundary (blue), bootleg exclusion circles (red), drill holes (green)
4. Click **Export Results** to save `output.json`

## Input Format

```json
{
  "boundary": [[x1, y1], [x2, y2], ...],
  "bootlegs": [[bx1, by1], [bx2, by2], ...]
}
```

- **boundary** — closed polygon (simple, no self-intersections)
- **bootlegs** — points defining circular exclusion zones of radius 0.75 m

## Output Format

```json
{
  "holes": [[x1, y1], [x2, y2], ...]
}
```



## MILP Optimization Model

The MILP optimizer formulates the drill plan problem as a **Mixed Integer Linear Program** to find globally optimal solutions.

### Formulation Summary

**Decision Variables:**
- $x_k \in \{0,1\}$ — whether candidate hole $k$ is selected

**Objective:** Maximize fragmented area while penalizing over-fragmentation:
$$\text{Maximize: } \sum_g z_g - 0.5 \sum_g v_g$$

where:
- $z_g \in \{0,1\}$ — cell $g$ is fragmented ($F_g > T$)
- $v_g \in \{0,1\}$ — cell $g$ is over-fragmented ($F_g \geq T_{over}$)

**Key Features:**
- Pre-computes fragmentation coefficients for efficiency
- Grid reduction (coarse optimization grid, fine evaluation grid) for tractability
- Scalable to large problems via Gurobi solver


  
Alternative solvers can be configured in the code (CBC, CPLEX, etc.) via Pyomo.

## Files

```
├── drill_plan_optimizer_milp.py   # MILP optimizer (Pyomo + Gurobi)
├── requirements.txt               # MILP dependencies
├── output.json            # Example optimized hole positions
├── input.json                     # Sample input file
├── README.md                      # This file
```
