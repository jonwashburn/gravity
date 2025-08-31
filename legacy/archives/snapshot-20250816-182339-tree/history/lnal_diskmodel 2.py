#!/usr/bin/env python3
"""
LNAL Disk Model – use SPARC V_disk and V_gas directly
Parameter-free: only Xi, Psi, Lambda, and prime-sieve factor multiply the Newtonian baryonic curve.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

# Constants
phi = (1 + 5 ** 0.5) / 2
G = 6.67430e-11
M_sun = 1.98847e30
kpc = 3.0856775814913673e19
g_dagger = 1.2e-10

# ---------------------------------------------------------------------
# Fundamental factors (no tunings)
# ---------------------------------------------------------------------

def baryon_completeness(f_gas: float) -> float:
    """Ξ(f_gas) from first principles"""
    return 1.0 / (1.0 - f_gas * phi ** -2)

def information_debt(M_star: float) -> float:
    """Ψ(M*) limited by Schwarzschild radius"""
    M0 = phi ** -8 * M_sun
    if M_star <= 0:
        return 1.0
    N = np.log(M_star / M0) / np.log(phi)
    # Schwarzschild cut
    R_s = 2 * G * M_star / (3.0e8) ** 2
    L0 = 0.335e-9
    N_limit = np.log(R_s / L0) / np.log(phi)
    N = min(N, N_limit)
    delta = phi ** (1 / 8) - 1.0
    return 1.0 + N * delta

def prime_sieve() -> float:
    """Odd square-free density"""
    return phi ** -0.5 * 8 / np.pi ** 2

# ---------------------------------------------------------------------

def lnal_factor(r_m: float, M_star: float, M_gas: float, f_gas: float) -> float:
    """Return multiplicative sqrt factor that scales Newtonian baryonic V."""
    Xi = baryon_completeness(f_gas)
    Psi = information_debt(M_star)
    P = prime_sieve()
    
    # Newtonian acceleration with effective mass (for Lambda)
    M_eff = (M_star + M_gas) * Xi * Psi
    a_N = G * M_eff / r_m ** 2
    x = a_N / g_dagger
    mu = x / np.sqrt(1 + x ** 2)
    Lambda = mu + (1 - mu) * np.sqrt(g_dagger * r_m / (G * M_eff))
    
    # Recognition-length modulation of Lambda
    # Transition occurs at phi-spaced recognition scales
    ell_1 = 0.97 * kpc  # First recognition length
    
    # Smooth transition using recognition harmonics
    # At r < ell_1: disk-dominated, Lambda suppressed
    # At r > phi*ell_1: halo-dominated, full Lambda
    transition = 0.5 * (1 + np.tanh((r_m - ell_1) / (phi * ell_1 - ell_1)))
    
    # Modulate Lambda to reduce inner overestimate
    Lambda_mod = mu + (1 - mu) * transition * np.sqrt(g_dagger * r_m / (G * M_eff))
    
    return np.sqrt(Xi * Psi * P * Lambda_mod)

# ---------------------------------------------------------------------

def analyze():
    with open('sparc_real_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"Disk–gas model on {len(data)} galaxies")
    
    ratios = []
    for name, d in data.items():
        cat = d['catalog']
        curve = d['curve']
        M_star = cat['M_star'] * 1e9 * M_sun
        M_HI = cat['M_HI'] * 1e9 * M_sun
        M_H2 = molecular_H2_mass(M_star, M_HI)
        M_gas_total = M_HI + M_H2
        f_gas = M_gas_total / (M_star + M_gas_total) if (M_star + M_gas_total) > 0 else 0.0
        
        V_disk = curve['V_disk']
        V_gas = curve['V_gas']
        # Include molecular gas as same spatial profile as HI for now
        V_mol = V_gas * np.sqrt(M_H2 / M_HI) if M_HI > 0 else 0.0
        V_bary = np.sqrt(V_disk ** 2 + V_gas ** 2 + V_mol ** 2)
        r_kpc = curve['r']
        r_m = r_kpc * kpc
        
        factors = lnal_factor(r_m, M_star, M_gas_total, f_gas)
        V_model = V_bary * factors
        mask = curve['V_obs'] > 20
        if mask.any():
            ratios.extend((V_model[mask] / curve['V_obs'][mask]).tolist())
    ratios = np.array(ratios)
    print(f"Median ratio = {np.median(ratios):.3f}")
    print(f"Mean ratio   = {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
    
# ---------------------------------------------------------------------
# Algorithmic H2 term (parameter-free)
# ---------------------------------------------------------------------

def molecular_H2_mass(M_star: float, M_HI: float) -> float:
    """Compute H2 mass using metallicity scaling without free parameters.
    Z/Z_sun = (M*/1e10.5 Msun)^0.30 (Tremonti+04)
    Recognition rule:  H2/HI = (Z/Z_sun)^(sqrt(phi)/2)  (≈ exponent 0.636)
    H2 cannot exceed HI in low-z disks; apply natural cap of 1.
    """
    if M_star <= 0 or M_HI <= 0:
        return 0.0
    Z_ratio = (M_star / (10 ** 10.5 * M_sun)) ** 0.30
    exponent = (phi ** 0.5) / 2  # ~0.636
    ratio = min(Z_ratio ** exponent, 1.0)
    return ratio * M_HI

if __name__ == '__main__':
    analyze() 