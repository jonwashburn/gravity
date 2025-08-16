#!/usr/bin/env python3
"""
LNAL Unified Final - Recognition Science Galaxy Rotation
Parameter-free with all physical effects
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from scipy.special import iv, kv

# Physical constants
phi = (1 + np.sqrt(5)) / 2
G = 6.67430e-11
c = 2.99792458e8
M_sun = 1.98847e30
kpc = 3.0856775814913673e19
g_dagger = 1.2e-10

def baryon_completeness(f_gas):
    """Ξ(f_gas) = 1 / (1 - f_gas φ^{-2})"""
    return 1.0 / (1.0 - f_gas * phi**-2)

def information_debt(M_star, V_rot=None):
    """Ψ(M*) with optional velocity-dependent kinetic term"""
    M0 = phi**-8 * M_sun
    if M_star <= 0:
        return 1.0
    
    # Hierarchical depth limited by Schwarzschild radius
    N_raw = np.log(M_star / M0) / np.log(phi)
    R_s = 2.0 * G * M_star / c**2
    L_0 = 0.335e-9
    N_limit = np.log(R_s / L_0) / np.log(phi) if R_s > 0 else N_raw
    N = min(max(0.0, N_raw), max(0.0, N_limit))
    
    # Base information debt
    delta = phi**(1/8.0) - 1.0
    psi_base = 1.0 + N * delta
    
    # Add velocity-dependent kinetic debt if provided
    if V_rot is not None:
        # From Jeans equation: σ ≈ 0.7 * V_rot for disks
        sigma = 0.7 * V_rot
        psi_kinetic = 0.5 * (sigma / c)**2
        return psi_base * (1 + psi_kinetic)
    else:
        return psi_base

def lnal_unified(r_kpc, M_star, M_HI, iterate_kinetic=True):
    """
    Complete LNAL model with all effects
    If iterate_kinetic=True, self-consistently solve for V including kinetic debt
    """
    r = r_kpc * kpc
    
    # Basic parameters
    f_gas = M_HI / (M_star + M_HI) if (M_star + M_HI) > 0 else 0.0
    Xi = baryon_completeness(f_gas)
    
    # Prime sieve (8/π² for odd square-free)
    P = phi**(-0.5) * 8 / np.pi**2
    
    # Recognition lengths
    ell_1 = 0.97 * kpc
    ell_2 = 24.3 * kpc
    
    # Initial calculation without kinetic term
    Psi_0 = information_debt(M_star)
    M_eff_0 = (M_star + M_HI) * Xi * Psi_0
    
    # MOND interpolation with recognition bumps
    a_N = G * M_eff_0 / r**2
    x = a_N / g_dagger
    mu = x / np.sqrt(1 + x**2)
    
    # Recognition enhancement
    bump_1 = 0.1 * np.exp(-(np.log(r/ell_1))**2 / 2)
    bump_2 = 0.03 * np.exp(-(np.log(r/ell_2))**2 / 2)
    Lambda = mu + (1 - mu) * np.sqrt(g_dagger * r / (G * M_eff_0)) * (1 + bump_1 + bump_2)
    
    # First estimate of velocity
    V2_0 = (G * M_eff_0 / r) * Lambda * P
    V_0 = np.sqrt(max(V2_0, 0.0))
    
    if iterate_kinetic:
        # Self-consistent iteration with kinetic debt
        V_prev = V_0
        for _ in range(3):  # Usually converges in 2-3 iterations
            Psi = information_debt(M_star, V_prev)
            M_eff = (M_star + M_HI) * Xi * Psi
            
            # Recalculate with new mass
            a_N = G * M_eff / r**2
            x = a_N / g_dagger
            mu = x / np.sqrt(1 + x**2)
            Lambda = mu + (1 - mu) * np.sqrt(g_dagger * r / (G * M_eff)) * (1 + bump_1 + bump_2)
            
            V2 = (G * M_eff / r) * Lambda * P
            V_new = np.sqrt(max(V2, 0.0))
            
            # Check convergence
            if abs(V_new - V_prev) / V_prev < 0.01:
                break
            V_prev = V_new
        
        return V_new / 1000  # m/s to km/s
    else:
        return V_0 / 1000

def analyze_sparc_unified():
    """Analyze SPARC with unified model"""
    
    # Load data
    with open('sparc_curves.pkl', 'rb') as f:
        curves = pickle.load(f)
    
    print("=== LNAL UNIFIED ANALYSIS ===")
    print(f"Analyzing {len(curves)} galaxies")
    
    ratios_flat = []
    ratios_curve = []
    
    for i, (name, data) in enumerate(curves.items()):
        galaxy = data['galaxy_data']
        curve = data['curve']
        
        M_star = galaxy['M_star'] * 1e9 * M_sun
        M_HI = galaxy['M_HI'] * 1e9 * M_sun
        
        # Test at flat velocity radius (3 R_disk)
        V_model_flat = lnal_unified(3 * galaxy['R_disk'], M_star, M_HI)
        ratio_flat = V_model_flat / galaxy['V_flat']
        ratios_flat.append(ratio_flat)
        
        # Full curve
        V_model_curve = np.array([lnal_unified(r, M_star, M_HI) for r in curve['r']])
        ratio_curve = np.median(V_model_curve[curve['V_obs'] > 10] / curve['V_obs'][curve['V_obs'] > 10])
        ratios_curve.append(ratio_curve)
        
        if i < 3:
            print(f"\n{name}:")
            print(f"  At 3 R_disk: V_model={V_model_flat:.1f} km/s, ratio={ratio_flat:.3f}")
            print(f"  Full curve median ratio: {ratio_curve:.3f}")
    
    ratios_flat = np.array(ratios_flat)
    ratios_curve = np.array(ratios_curve)
    
    print(f"\n=== RESULTS ===")
    print(f"Flat velocity (3 R_disk):")
    print(f"  Mean ratio = {np.mean(ratios_flat):.3f} ± {np.std(ratios_flat):.3f}")
    print(f"  Median ratio = {np.median(ratios_flat):.3f}")
    print(f"\nFull curve:")
    print(f"  Mean ratio = {np.mean(ratios_curve):.3f} ± {np.std(ratios_curve):.3f}")
    print(f"  Median ratio = {np.median(ratios_curve):.3f}")
    
    # Save results
    results = {
        'ratios_flat': ratios_flat,
        'ratios_curve': ratios_curve,
        'galaxy_names': list(curves.keys())
    }
    
    with open('lnal_unified_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.hist(ratios_flat, bins=30, alpha=0.5, label='Flat velocity (3 R_disk)', color='blue')
    plt.hist(ratios_curve, bins=30, alpha=0.5, label='Full curve median', color='red')
    plt.axvline(1.0, color='black', linestyle='--', label='Perfect')
    plt.xlabel('V_model / V_obs')
    plt.ylabel('Count')
    plt.title('LNAL Unified Model Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lnal_unified_comparison.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    analyze_sparc_unified() 