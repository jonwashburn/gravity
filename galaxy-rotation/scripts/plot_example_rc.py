#!/usr/bin/env python3
"""
Plot a representative SPARC rotation curve with ILG model overlay.

Inputs:
- sparc_master.pkl produced by build_sparc_master_table.py

Output:
- active/results/example_rc.pdf
"""

from __future__ import annotations

import os
import sys
import math
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


def find_master_table(project_root: Path) -> Optional[Path]:
    """Locate sparc_master.pkl in common locations."""
    candidates = [
        project_root / "active" / "results" / "sparc_master.pkl",
        project_root / "sparc_master.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def n_analytic(r_kpc: np.ndarray, A: float = 7.0, r0_kpc: float = 8.0, p: float = 1.6) -> np.ndarray:
    return 1.0 + A * (1.0 - np.exp(-np.power(np.maximum(r_kpc, 0.0) / r0_kpc, p)))


def compute_xi(f_gas: float, sigma0: float, sigma_star: float = 1.0e8,
               C0: float = 5.064, gamma: float = 2.953, delta: float = 0.216) -> float:
    f = max(0.0, min(1.0, float(f_gas)))
    ratio = float(sigma0) / float(sigma_star) if sigma_star > 0 else 0.0
    return 1.0 + C0 * (f ** gamma) * (ratio ** delta)


def compute_w(r_kpc: np.ndarray,
              T_dyn_years: np.ndarray,
              f_gas: float,
              sigma0: float,
              alpha: float = 0.191,
              tau0_s: float = 7.33e-15,
              lam: float = 0.118,
              zeta: float = 1.0) -> np.ndarray:
    xi = compute_xi(f_gas, sigma0)
    n_r = n_analytic(r_kpc)
    T_dyn_s = np.asarray(T_dyn_years, dtype=float) * SECONDS_PER_YEAR
    # Guard against zero/negative times
    T_dyn_s = np.where(T_dyn_s > 0, T_dyn_s, np.nan)
    w = lam * xi * n_r * np.power(T_dyn_s / tau0_s, alpha) * zeta
    return w


def pick_representative_galaxy(master: Dict[str, Any]) -> str:
    """Pick the galaxy with the most data points for a clean plot."""
    best_name = None
    best_len = -1
    for name, g in master.items():
        try:
            n = len(g.get('r', []))
        except Exception:
            n = 0
        if n > best_len:
            best_len = n
            best_name = name
    if best_name is None:
        # Fallback to arbitrary
        best_name = next(iter(master.keys()))
    return best_name


def main() -> int:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    out_dir = project_root / "active" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    master_path = find_master_table(project_root)
    if master_path is None:
        print("Error: sparc_master.pkl not found. Run build_sparc_master_table.py first.", file=sys.stderr)
        return 1

    with open(master_path, "rb") as f:
        master = pickle.load(f)

    if not isinstance(master, dict) or len(master) == 0:
        print("Error: sparc_master.pkl is empty or invalid.", file=sys.stderr)
        return 1

    name = pick_representative_galaxy(master)
    g = master[name]

    r = np.asarray(g.get('r', []), dtype=float)
    v_obs = np.asarray(g.get('v_obs', []), dtype=float)
    v_baryon = np.asarray(g.get('v_baryon', []), dtype=float)
    T_dyn = np.asarray(g.get('T_dyn', []), dtype=float)  # years
    f_gas_true = float(g.get('f_gas_true', 0.0))
    sigma0 = float(g.get('Sigma_0', 0.0))

    if any(len(x) == 0 for x in (r, v_obs, v_baryon, T_dyn)):
        print(f"Error: selected galaxy '{name}' missing required arrays.", file=sys.stderr)
        return 1

    # Compute ILG model
    w = compute_w(r, T_dyn, f_gas_true, sigma0)
    w = np.where(np.isfinite(w) & (w >= 0.0), w, np.nan)
    v_model = np.sqrt(w) * v_baryon

    # Plot
    plt.figure(figsize=(5.0, 3.2), dpi=150)
    order = np.argsort(r)
    plt.plot(r[order], v_obs[order], color='k', lw=1.6, label='v_obs')
    plt.plot(r[order], v_baryon[order], color='k', lw=1.2, ls='--', label='v_baryon')
    plt.plot(r[order], v_model[order], color='#1f77b4', lw=1.6, label='v_model (ILG)')
    plt.xlabel('r [kpc]')
    plt.ylabel('v [km/s]')
    plt.title(name)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out_path = out_dir / 'example_rc.pdf'
    plt.savefig(out_path)
    plt.close()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


