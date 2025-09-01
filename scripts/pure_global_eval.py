#!/usr/bin/env python3
"""
Pure-global evaluation (non-fitted) for ILG and MOND baselines.

Inputs:
  - active/results/sparc_master.pkl (built previously)

Outputs (repo root 'results/' directory):
  - results/bench_rs_per_galaxy.csv
  - results/bench_mond_per_galaxy.csv
  - results/bench_global_summary.csv

Policy (global-only):
  - Single global stellar M/L = 1.0 applied to v_disk
  - Fixed analytic n(r) with (A, r0[kpc], p) = (7, 8, 1.6)
  - Fixed α = 0.191, C_lag = φ^{-5}, a0 = 1.2e-10 m/s^2, g_ext = 0
  - ξ = 1 + φ^{-5} * f_gas_true**0.5 (global-only proxy)
  - ζ(r) = 1 (geometric factor disabled for simplicity)
  - Error model: σ_eff^2 = σ_obs^2 + σ0^2 + (f v_obs)^2
"""

from __future__ import annotations

import os
import math
import pickle
import pathlib
from typing import Dict, Any, List

import numpy as np
import pandas as pd


# Constants
KPC_TO_M = 3.086e19
KM_TO_M = 1.0e3
PHI = (1.0 + 5 ** 0.5) / 2.0
C_LAG = PHI ** (-5)  # ≈ 0.090
A0 = 1.2e-10  # m/s^2
ALPHA = 0.191

# Global-only settings
GLOBAL_ML = 1.0
N_A, N_R0, N_P = 7.0, 8.0, 1.6  # analytic n(r)

# Error-model settings (global-only, mirroring paper constants)
SIGMA0 = 10.0           # km/s
F_FLOOR = 0.05          # fractional
ALPHA_BEAM = 0.3        # dimensionless
ASYM_DWARF = 0.10       # fractional of v_obs
ASYM_SPIRAL = 0.05      # fractional of v_obs
K_TURB = 0.07           # dimensionless
P_TURB = 1.3            # dimensionless


def n_analytic(r_kpc: np.ndarray) -> np.ndarray:
    return 1.0 + N_A * (1.0 - np.exp(-((r_kpc / N_R0) ** N_P)))


def xi_global(f_gas_true: float) -> float:
    return 1.0 + (PHI ** -5) * (max(f_gas_true, 0.0) ** 0.5)


# Threads-informed complexity (discrete, global thresholds)
def compute_fgas_edges(master_table: Dict[str, Any], num_bins: int = 5) -> np.ndarray:
    """Compute global quantile edges for f_gas_true across the sample.

    Returns an array of length (num_bins-1) with quantile cut points.
    Deterministic and global-only.
    """
    fvals: List[float] = []
    for g in master_table.values():
        try:
            fvals.append(float(g.get('f_gas_true', 0.0)))
        except Exception:
            fvals.append(0.0)
    if len(fvals) == 0:
        return np.array([])
    # Use evenly spaced quantiles excluding 0 and 1
    qs = np.linspace(0, 1, num_bins + 1)[1:-1]
    return np.quantile(np.array(fvals), qs)


def xi_threads(f_gas_true: float, edges: np.ndarray) -> float:
    """Discrete threads factor from global f_gas quantile bins.

    - Assign bin index b \in {0,..,B-1} by global edges
    - Map to a unit interval center u = (b + 0.5)/B
    - Return xi = 1 + C_LAG * sqrt(u)
    This preserves global-only policy and introduces a discrete, auditable
    complexity proxy aligned with the Ribbons/Threads perspective.
    """
    if edges.size == 0:
        return xi_global(f_gas_true)
    # Bin index
    b = int(np.searchsorted(edges, max(f_gas_true, 0.0), side='right'))
    B = edges.size + 1
    u = (b + 0.5) / B
    return 1.0 + C_LAG * (u ** 0.5)


def compute_global_n_norm(master_table: Dict[str, Any]) -> float:
    """Compute a single global normalization c_n so that the exponential-disc
    weighted mean of n(r) equals 1 for a representative Rd.

    We use Rd_ref = median of available R_d; weight w(r) ∝ r exp(-r/Rd_ref).
    """
    Rd_vals: List[float] = []
    for g in master_table.values():
        try:
            Rd = float(g.get('R_d', np.nan))
            if np.isfinite(Rd) and Rd > 0:
                Rd_vals.append(Rd)
        except Exception:
            continue
    Rd_ref = float(np.median(Rd_vals)) if Rd_vals else 2.0
    # radial grid up to 6 scale lengths
    r = np.linspace(0.05 * Rd_ref, 6.0 * Rd_ref, 600)
    w = r * np.exp(-r / Rd_ref)
    mean_n = float(np.trapz(n_analytic(r) * w, r) / np.trapz(w, r))
    if mean_n <= 0:
        return 1.0
    return 1.0 / mean_n


def w_g_kernel(g_bar_si: np.ndarray) -> np.ndarray:
    # g_ext = 0
    term = np.maximum(g_bar_si / A0, 1e-30)  # avoid div/underflow
    return 1.0 + C_LAG * (term ** (-ALPHA) - 1.0)


def compute_ilg_velocity(r_kpc: np.ndarray,
                         v_gas: np.ndarray,
                         v_disk: np.ndarray,
                         v_bul: np.ndarray,
                         v_obs: np.ndarray,
                         f_gas_true: float,
                         R_d_guess: float | None = None,
                         xi_value: float | None = None,
                         c_n: float = 1.0) -> np.ndarray:
    # Apply global M/L to stellar disk
    v_baryon = np.sqrt(v_gas ** 2 + (math.sqrt(GLOBAL_ML) * v_disk) ** 2 + v_bul ** 2)

    # g_bar in SI
    r_m = r_kpc * KPC_TO_M
    v_baryon_mps = v_baryon * KM_TO_M
    g_bar = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)

    w_g = w_g_kernel(g_bar)
    n_r = c_n * n_analytic(r_kpc)
    xi = xi_global(f_gas_true) if xi_value is None else float(xi_value)
    # Geometric factor: simple thickness proxy using R_d if available
    if R_d_guess is None or R_d_guess <= 0:
        zeta = 1.0
    else:
        hz_over_Rd = 0.25
        hz = hz_over_Rd * R_d_guess
        # clip to [0.8, 1.2] as in paper
        zeta = np.clip(1.0 + 0.5 * (hz / (r_kpc + 0.1 * R_d_guess)), 0.8, 1.2)
    w = w_g * n_r * xi * zeta

    v_model = np.sqrt(np.maximum(w, 0.0)) * v_baryon
    return v_model


def mond_simple_nu_velocity(r_kpc: np.ndarray,
                            v_gas: np.ndarray,
                            v_disk: np.ndarray,
                            v_bul: np.ndarray) -> np.ndarray:
    # Global M/L = 1.0 for fair baseline
    v_baryon = np.sqrt(v_gas ** 2 + (math.sqrt(GLOBAL_ML) * v_disk) ** 2 + v_bul ** 2)
    r_m = r_kpc * KPC_TO_M
    v_baryon_mps = v_baryon * KM_TO_M
    g_N = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)
    y = np.maximum(g_N / A0, 1e-30)
    # Simple nu: g = nu(y) * g_N with nu(y) = 0.5 + sqrt(0.25 + 1/y)
    nu = 0.5 + np.sqrt(0.25 + 1.0 / y)
    g = nu * g_N
    v_mps = np.sqrt(np.maximum(g, 0.0) * r_m)
    return v_mps / KM_TO_M


def sigma_total_kms(r_kpc: np.ndarray,
                    v_obs: np.ndarray,
                    v_err: np.ndarray,
                    R_d_guess: float | None = None,
                    is_dwarf: bool | None = None,
                    beam_kpc: float | None = None) -> np.ndarray:
    # Base floors
    sigma = v_err ** 2 + SIGMA0 ** 2 + (F_FLOOR * v_obs) ** 2
    # Beam smearing: prefer catalog beam_kpc if provided; else heuristic from R_d
    if beam_kpc is None:
        if R_d_guess is not None and R_d_guess > 0:
            beam_kpc = 0.3 * R_d_guess
        else:
            beam_kpc = 0.6  # conservative fallback
    sigma_beam = ALPHA_BEAM * beam_kpc * v_obs / (r_kpc + beam_kpc)
    sigma += sigma_beam ** 2
    # Asymmetry term
    if is_dwarf is None:
        is_dwarf = bool(np.nanmax(v_obs) < 80.0)
    asym_frac = ASYM_DWARF if is_dwarf else ASYM_SPIRAL
    sigma += (asym_frac * v_obs) ** 2
    # Turbulence/warp proxy
    if R_d_guess is None or R_d_guess <= 0:
        Rd = 2.0
    else:
        Rd = R_d_guess
    sigma_turb = K_TURB * v_obs * (1.0 - np.exp(-r_kpc / Rd)) ** P_TURB
    sigma += sigma_turb ** 2
    return np.sqrt(np.maximum(sigma, 0.0))


def eval_pure_global(master_table: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    min_points = 5
    beam_mask_factor = 0.3  # heuristic when catalog beam is unavailable
    # Global threads thresholds (discrete complexity proxy)
    fgas_edges = compute_fgas_edges(master_table, num_bins=5)
    c_n = compute_global_n_norm(master_table)

    for name, g in master_table.items():
        # Optional Q=1 gate when available
        try:
            if int(g.get('Q', 1)) != 1:
                continue
        except Exception:
            pass
        df = g['data']
        r = df['rad'].values.astype(float)
        v_obs = df['vobs'].values.astype(float)
        v_err = df['verr'].values.astype(float)
        v_gas = df['vgas'].values.astype(float)
        v_disk = df['vdisk'].values.astype(float)
        v_bul = df['vbul'].values.astype(float)

        f_gas_true = float(g.get('f_gas_true', 0.0))

        R_d_guess = float(g.get('R_d', 2.0))
        # Apply inner-beam mask: prefer catalog beam if present; else heuristic r >= 0.3 R_d
        beam_from_cat = g.get('beam_kpc', None)
        try:
            beam_from_cat = float(beam_from_cat) if beam_from_cat is not None else None
        except Exception:
            beam_from_cat = None
        b_kpc = (beam_from_cat if (beam_from_cat is not None and beam_from_cat > 0) else
                 beam_mask_factor * R_d_guess)
        mask = np.isfinite(r) & np.isfinite(v_obs) & np.isfinite(v_err) & (r >= b_kpc)
        r = r[mask]
        v_obs = v_obs[mask]
        v_err = v_err[mask]
        v_gas = v_gas[mask]
        v_disk = v_disk[mask]
        v_bul = v_bul[mask]
        if r.size < min_points:
            continue

        # Threads-informed xi (discrete, global thresholds)
        xi_val = xi_threads(f_gas_true, fgas_edges)
        v_ilg = compute_ilg_velocity(r, v_gas, v_disk, v_bul, v_obs, f_gas_true,
                                     R_d_guess, xi_value=xi_val, c_n=c_n)
        v_mond = mond_simple_nu_velocity(r, v_gas, v_disk, v_bul)

        is_dwarf = bool(np.nanmax(v_obs) < 80.0)
        sig = sigma_total_kms(r, v_obs, v_err, R_d_guess=R_d_guess, is_dwarf=is_dwarf,
                              beam_kpc=beam_from_cat)
        chi2_ilg = float(np.sum(((v_obs - v_ilg) / sig) ** 2))
        chi2_mond = float(np.sum(((v_obs - v_mond) / sig) ** 2))

        n_pts = int(r.size)
        rows.append({
            'name': name,
            'N': n_pts,
            'chi2_ilg': chi2_ilg,
            'chi2N_ilg': chi2_ilg / max(n_pts, 1),
            'chi2_mond': chi2_mond,
            'chi2N_mond': chi2_mond / max(n_pts, 1),
        })

    return pd.DataFrame(rows).sort_values('name')


def main() -> None:
    base = pathlib.Path(__file__).resolve().parents[2]
    pkl_path = base / 'active' / 'results' / 'sparc_master.pkl'
    out_dir = base / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing master table: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        master_table: Dict[str, Any] = pickle.load(f)

    df = eval_pure_global(master_table)

    # Write per-galaxy CSVs
    df_rs = df[['name', 'N', 'chi2_ilg', 'chi2N_ilg']].rename(columns={
        'chi2_ilg': 'chi2', 'chi2N_ilg': 'chi2N'
    })
    df_mond = df[['name', 'N', 'chi2_mond', 'chi2N_mond']].rename(columns={
        'chi2_mond': 'chi2', 'chi2N_mond': 'chi2N'
    })

    df_rs.to_csv(out_dir / 'bench_rs_per_galaxy.csv', index=False)
    df_mond.to_csv(out_dir / 'bench_mond_per_galaxy.csv', index=False)

    # Summary
    policy = (
        'pure-global (no per-galaxy tuning); analytic n(r) with global disc-weight normalization; '
        'global M/L=1.0; ξ_threads: f_gas discretized by global quintiles; ζ with h_z/R_d=0.25; '
        'errors σ0, fractional floor, beam-proxy/cat-beam when available, asymmetry, turbulence; '
        'inner-beam mask uses catalog beam when available, else r>=0.3 R_d'
    )
    summary = {
        'N_ilg': int(len(df_rs)),
        'median_chi2N_ilg': float(np.median(df_rs['chi2N'])),
        'mean_chi2N_ilg': float(np.mean(df_rs['chi2N'])),
        'N_mond': int(len(df_mond)),
        'median_chi2N_mond': float(np.median(df_mond['chi2N'])),
        'mean_chi2N_mond': float(np.mean(df_mond['chi2N'])),
        'policy': policy,
    }
    pd.DataFrame([summary]).to_csv(out_dir / 'bench_global_summary.csv', index=False)

    print("Wrote:")
    for p in [
        out_dir / 'bench_rs_per_galaxy.csv',
        out_dir / 'bench_mond_per_galaxy.csv',
        out_dir / 'bench_global_summary.csv',
    ]:
        print(f"  - {p}")


if __name__ == '__main__':
    main()


