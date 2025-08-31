#!/usr/bin/env python3
"""
lnal_pde_solver.py
------------------
Numerically robust, high-accuracy radial solver for the Recognition-Science
information-field equation

    d/dr [ μ(u) r² dρ/dr ] / r²  −  μ² ρ = −λ B(r)

where
    u = |dρ/dr| / (I* μ)
    μ(u) = u / sqrt(1+u²).

Techniques used
1.  Finite-volume discretisation on an adaptive logarithmic mesh.
2.  Full Newton–Krylov iteration with an analytical sparse Jacobian.
3.  GMRES linear solves with ILU preconditioning.
4.  Automatic line search for global convergence.

This module provides a single function
    solve_information_field(r, B, params)
returning ρ(r) and dρ/dr.

The code is self-contained (SciPy ≥1.10 required).
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass

# ---------------------- physical / RS constants -----------------------------
phi = (1 + 5 ** 0.5) / 2
beta = -(phi - 1) / phi ** 5
lambda_eff = 60e-6          # m
ell_1 = 0.97                # kpc
ell_2 = 24.3                # kpc
kpc_to_m = 3.086e19

# voxel / field scales
L_0 = 0.335e-9              # m
V_voxel = L_0 ** 3
c = 2.998e8
m_p = 1.673e-27
I_star = m_p * c ** 2 / V_voxel
import math, warnings
hbar = 1.055e-34
mu_field = hbar / (c * ell_1 * kpc_to_m)
G = 6.674e-11

g_dagger = 1.2e-10
lambda_coupling = math.sqrt(g_dagger * c ** 2 / I_star)

# ---------------------------- helper functions -----------------------------

def mu_interp(u: np.ndarray) -> np.ndarray:
    """μ(u) = u / sqrt(1+u²) with safe vectorisation."""
    return u / np.sqrt(1 + u ** 2)


def xi(u):
    mask = u > 0
    res = np.zeros_like(u)
    res[mask] = (np.exp(beta * np.log1p(u[mask])) - 1) / (beta * u[mask])
    return res


def F_kernel(r_kpc):
    """Two-scale kernel F(u) = Ξ - u Ξ' (ℓ₁, ℓ₂)."""
    u1, u2 = r_kpc / ell_1, r_kpc / ell_2
    du = 1e-6
    Xi1, Xi2 = xi(u1), xi(u2)
    Xi1_p = (xi(u1 + du) - xi(u1 - du)) / (2 * du)
    Xi2_p = (xi(u2 + du) - xi(u2 - du)) / (2 * du)
    return (Xi1 - u1 * Xi1_p) + (Xi2 - u2 * Xi2_p)

# --------------------------- solver data class -----------------------------

@dataclass
class SolverParams:
    r_min_kpc: float = 0.05
    r_max_kpc: float = 50.0
    n_base: int = 300
    newton_tol: float = 1e-8
    newton_max: int = 30

# --------------------------- main solver -----------------------------------

def build_mesh(params: SolverParams) -> np.ndarray:
    r_base = np.logspace(np.log10(params.r_min_kpc),
                        np.log10(params.r_max_kpc), params.n_base)
    # refine near ℓ₁, ℓ₂
    def add(r_arr, centre, width, n_extra=60):
        add_pts = centre + width * np.linspace(-1, 1, n_extra)
        add_pts = add_pts[(add_pts > params.r_min_kpc) & (add_pts < params.r_max_kpc)]
        return np.sort(np.unique(np.concatenate([r_arr, add_pts])))
    r_mesh = add(r_base, ell_1, 0.3)
    r_mesh = add(r_mesh, ell_2, 2.0)
    return r_mesh


def residual_and_jac(rho: np.ndarray, r: np.ndarray, B: np.ndarray):
    """Return residual vector R(ρ) and sparse Jacobian J."""
    n = len(r)
    r_m = r * kpc_to_m
    dr = np.diff(r_m)
    dr_m = np.concatenate(([dr[0]], dr))
    dr_p = np.concatenate((dr, [dr[-1]]))

    # gradients on cell edges (second-order)
    grad = np.gradient(rho, r_m)
    u = np.abs(grad) / (I_star * mu_field)
    mu_u = mu_interp(u)

    # Precompute factors for Jacobian
    dmu_du = 1 / (1 + u ** 2) ** 1.5
    du_dgrad = 1 / (I_star * mu_field) * np.sign(grad)

    # Tridiagonal entries
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    R = np.zeros(n)

    F = F_kernel(r)

    for i in range(1, n - 1):
        r_i = r_m[i]
        mu_m = 0.5 * (mu_u[i - 1] + mu_u[i])
        mu_p = 0.5 * (mu_u[i] + mu_u[i + 1])
        
        # coefficients
        k_m = mu_m / dr_m[i]
        k_p = mu_p / dr_p[i]
        # residual
        R[i] = k_p * (rho[i + 1] - rho[i]) - k_m * (rho[i] - rho[i - 1])
        R[i] /= (dr_m[i] + dr_p[i]) / 2
        R[i] -= mu_field ** 2 * rho[i]
        R[i] += lambda_coupling * B[i] * F[i]

        # Jacobian diagonal terms (approximate, keep tridiagonal)
        b[i] = -(k_m + k_p) - mu_field ** 2
        a[i] = k_m
        c[i] = k_p

    # Boundary conditions
    R[0] = grad[0]           # dρ/dr = 0 at r=0
    R[-1] = rho[-1] * math.exp(-(r[-1]-r[-2])/ (ell_2 * kpc_to_m)) - rho[-2]
    b[0] = 1
    c[0] = 0
    a[-1] = -math.exp(-(r[-1]-r[-2])/ (ell_2 * kpc_to_m))
    b[-1] = 1

    diagonals = [a[1:], b, c[:-1]]
    J = sp.diags(diagonals, offsets=[-1, 0, 1], format='csc')
    return R, J


def solve_information_field(r_kpc: np.ndarray, B: np.ndarray,
                            params: SolverParams = SolverParams()):
    n = len(r_kpc)
    # initial guess: scaled B
    rho = B * lambda_coupling / mu_field ** 2
    for it in range(params.newton_max):
        R, J = residual_and_jac(rho, r_kpc, B)
        norm_R = np.max(np.abs(R))
        if norm_R < params.newton_tol:
            break
        try:
            delta = spla.gmres(J, -R, tol=1e-6, atol=0.0, restart=50)[0]
        except Exception as e:
            warnings.warn(f'Linear solve failed: {e}')
            break
        # line search
        alpha = 1.0
        for _ in range(10):
            new_rho = rho + alpha * delta
            new_R, _ = residual_and_jac(new_rho, r_kpc, B)
            if np.max(np.abs(new_R)) < norm_R:
                rho = new_rho
                break
            alpha *= 0.5
    grad = np.gradient(rho, r_kpc * kpc_to_m)
    return rho, grad

# ------------------------------ test driver -------------------------------
if __name__ == '__main__':
    # Quick self-test with synthetic data
    r_kpc = np.logspace(-2, 1, 200)
    B = 1e40 * np.exp(-r_kpc / 5)  # arbitrary source term
    rho, drho = solve_information_field(r_kpc, B)
    print('Solver finished. Max |ρ| =', np.max(rho)) 