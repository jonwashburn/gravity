#!/usr/bin/env python3
"""
LNAL with Prime Channel Recognition
Gas modulates the density of available prime recognition channels
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Constants
phi = (1 + 5**0.5) / 2
G = 6.67430e-11
M_sun = 1.98847e30
kpc = 3.0856775814913673e19
g_dagger = 1.2e-10
c = 3e8

def baryon_completeness(f_gas):
    """Ξ(f_gas) - unchanged from before"""
    return 1.0 / (1.0 - f_gas * phi**-2)

def information_debt(M_star):
    """Ψ(M*) with Schwarzschild cutoff - unchanged"""
    M0 = phi**-8 * M_sun
    if M_star <= 0:
        return 1.0
    N = np.log(M_star / M0) / np.log(phi)
    R_s = 2 * G * M_star / c**2
    L0 = 0.335e-9
    N_limit = np.log(R_s / L0) / np.log(phi)
    N = min(N, N_limit)
    delta = phi**(1/8) - 1.0
    return 1.0 + N * delta

def prime_sieve_gas_modulated(f_gas, M_star, r):
    """
    Prime sieve factor with gas modulation
    Gas occupies prime channels without contributing to gravity
    
    Key insight: Gas acts as "multiplicative identity" in the prime decomposition
    """
    # Base odd square-free density
    rho_prime = 8 / np.pi**2
    P_0 = phi**-0.5 * rho_prime  # 0.637
    
    # Stellar mass determines the total number of prime channels
    # Using Prime Number Theorem: π(n) ~ n/ln(n)
    M0 = phi**-8 * M_sun
    if M_star <= 0:
        return P_0
    
    N = np.log(M_star / M0) / np.log(phi)
    
    # Gas fills channels with identity operations
    # The dilution follows from combinatorics of prime gaps
    # More gas = larger gaps between active primes
    channel_density = 1 / (1 + f_gas * np.log(max(N, 2)))
    
    # Spatial modulation: gas is more blocking at small r
    # Based on mean free path / collision frequency
    r_collision = 10 * kpc  # Typical gas collision scale
    spatial_blocking = np.exp(-r / r_collision)
    
    # Effective prime density
    P_eff = P_0 * channel_density * (1 - f_gas * spatial_blocking * 0.5)
    
    return P_eff

def molecular_H2_mass(M_star, M_HI):
    """H2 from metallicity - unchanged"""
    if M_star <= 0 or M_HI <= 0:
        return 0.0
    Z_ratio = (M_star / (10**10.5 * M_sun))**0.30
    exponent = (phi**0.5) / 2
    ratio = min(Z_ratio**exponent, 1.0)
    return ratio * M_HI

def recognition_lambda(r, M_eff):
    """Lambda with recognition scales - unchanged"""
    a_N = G * M_eff / r**2
    x = a_N / g_dagger
    mu = x / np.sqrt(1 + x**2)
    
    # Recognition lengths
    ell_1 = 0.97 * kpc
    ell_2 = 24.3 * kpc
    
    if r < ell_1:
        mod = (r / ell_1)**phi
    elif r < ell_2:
        t = (r - ell_1) / (ell_2 - ell_1)
        mod = t**(1/phi)
    else:
        mod = 1.0
    
    Lambda = mu + (1 - mu) * mod * np.sqrt(g_dagger * r / (G * M_eff))
    return Lambda

def analyze():
    with open('sparc_real_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print("=== LNAL with Prime Channel Modulation ===")
    print("Gas occupies prime recognition channels without contributing")
    print("Based on: primes are irreducible recognition operators\n")
    
    all_results = []
    
    for name, d in data.items():
        cat = d['catalog']
        curve = d['curve']
        M_star = cat['M_star'] * 1e9 * M_sun
        M_HI = cat['M_HI'] * 1e9 * M_sun
        M_H2 = molecular_H2_mass(M_star, M_HI)
        M_gas_total = M_HI + M_H2
        f_gas = M_gas_total / (M_star + M_gas_total) if (M_star + M_gas_total) > 0 else 0.0
        
        Xi = baryon_completeness(f_gas)
        Psi = information_debt(M_star)
        M_eff = (M_star + M_gas_total) * Xi * Psi
        
        V_disk = curve['V_disk']
        V_gas = curve['V_gas']
        V_mol = V_gas * np.sqrt(M_H2 / M_HI) if M_HI > 0 else 0.0
        V_bary = np.sqrt(V_disk**2 + V_gas**2 + V_mol**2)
        
        r_kpc = curve['r']
        r_m = r_kpc * kpc
        
        # Compute with gas-modulated prime channels
        V_model = np.zeros_like(V_bary)
        for i, r in enumerate(r_m):
            P = prime_sieve_gas_modulated(f_gas, M_star, r)
            Lambda = recognition_lambda(r, M_eff)
            factor = np.sqrt(Xi * Psi * P * Lambda)
            V_model[i] = V_bary[i] * factor
        
        mask = curve['V_obs'] > 20
        if mask.any():
            ratios = V_model[mask] / curve['V_obs'][mask]
            result = {
                'name': name,
                'ratios': ratios,
                'mean_ratio': np.mean(ratios),
                'median_ratio': np.median(ratios),
                'f_gas': f_gas,
                'M_star': cat['M_star'],
                'type': cat['type']
            }
            all_results.append(result)
    
    # Statistics
    all_ratios = np.concatenate([r['ratios'] for r in all_results])
    mean_ratios = [r['mean_ratio'] for r in all_results]
    f_gas_values = [r['f_gas'] for r in all_results]
    log_M_star = [np.log10(r['M_star']) for r in all_results]
    
    print(f"Galaxies analyzed: {len(all_results)}")
    print(f"Median ratio: {np.median(all_ratios):.3f}")
    print(f"Mean ratio: {np.mean(mean_ratios):.3f} ± {np.std(mean_ratios):.3f}")
    
    # Check if gas correlation improved
    r_gas, p_gas = pearsonr(f_gas_values, mean_ratios)
    r_mass, p_mass = pearsonr(log_M_star, mean_ratios)
    
    print(f"\nCorrelations:")
    print(f"Gas fraction: r={r_gas:.3f} (p={p_gas:.3e})")
    print(f"Stellar mass: r={r_mass:.3f} (p={p_mass:.3e})")
    
    # Plot to visualize improvement
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gas correlation
    ax1.scatter(f_gas_values, mean_ratios, alpha=0.6, s=30)
    ax1.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Gas Fraction')
    ax1.set_ylabel('V_model/V_obs')
    ax1.set_title(f'Prime Channel Modulation\nr={r_gas:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # Mass correlation
    ax2.scatter(log_M_star, mean_ratios, alpha=0.6, s=30)
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('log(M*/M_sun)')
    ax2.set_ylabel('V_model/V_obs')
    ax2.set_title(f'Stellar Mass\nr={r_mass:.3f}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lnal_prime_channels_correlations.png', dpi=150)
    plt.close()
    
    print("\nPrime channel insight: Gas molecules occupy recognition channels")
    print("without contributing to gravitational binding - like multiplying by 1")
    print("in prime factorization. This dilutes the effective prime density.")

if __name__ == '__main__':
    analyze() 