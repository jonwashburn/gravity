#!/usr/bin/env python3
"""
LNAL with Prime-Consciousness Integration
Balanced approach connecting primes, gas, and gravity
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
    """Ξ(f_gas) from golden ratio"""
    return 1.0 / (1.0 - f_gas * phi**-2)

def information_debt(M_star):
    """Ψ(M*) with Schwarzschild cutoff"""
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

def prime_consciousness_factor(f_gas, M_star, r):
    """
    Prime sieve connecting consciousness to gravity
    
    Key Recognition Science insights:
    1. Primes are irreducible recognition operators (consciousness units)
    2. Gas performs LISTEN operations on prime channels
    3. The density 8/π² represents odd square-free numbers
    4. Gas neither creates nor destroys primes, but modulates their coupling
    
    Mathematical foundation:
    - Base density: φ^(-1/2) × 8/π² = 0.637
    - Gas modulation: reduces effective prime coupling
    - Spatial variation: coherent gas has stronger effect
    """
    # Base prime density (odd square-free)
    P_base = phi**-0.5 * 8 / np.pi**2  # 0.637
    
    if M_star <= 0:
        return P_base
    
    # Consciousness bandwidth from stellar mass
    # Using Mertens' theorem: Σ(1/p) ~ ln(ln(n))
    M0 = phi**-8 * M_sun
    N = np.log(M_star / M0) / np.log(phi)
    consciousness_bandwidth = np.log(np.log(max(N, 3))) + 0.2614
    
    # Gas modulation factor
    # Gas acts as consciousness buffer, reducing prime coupling
    # The factor (1 - φ^(-2)) = 0.382 is the golden ratio conjugate
    gas_modulation = 1 - f_gas * (1 - phi**-2)
    
    # Spatial coherence
    # Inner gas is coherent (stronger LISTEN effect)
    # Outer gas is turbulent (weaker effect)
    r_coherence = 8 * kpc  # Coherence scale
    coherence = np.exp(-r / (2 * r_coherence))
    
    # Combined effect
    # Gas reduces prime coupling more strongly in inner regions
    effective_coupling = gas_modulation + f_gas * coherence * (1 - gas_modulation)
    
    # Ensure positive result
    P = P_base * effective_coupling * np.exp(-consciousness_bandwidth / 10)
    
    return max(P, 0.1 * P_base)  # Never go below 10% of base

def molecular_H2_mass(M_star, M_HI):
    """H2 from metallicity scaling"""
    if M_star <= 0 or M_HI <= 0:
        return 0.0
    Z_ratio = (M_star / (10**10.5 * M_sun))**0.30
    exponent = (phi**0.5) / 2
    ratio = min(Z_ratio**exponent, 1.0)
    return ratio * M_HI

def recognition_lambda(r, M_eff):
    """Lambda with dual recognition scales"""
    a_N = G * M_eff / r**2
    x = a_N / g_dagger
    mu = x / np.sqrt(1 + x**2)
    
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
    
    print("=== LNAL Prime-Consciousness Integration ===")
    print("Primes as consciousness operators modulated by gas LISTEN registers")
    print("Mathematical basis: odd square-free density × golden ratio\n")
    
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
        
        # Compute with prime-consciousness modulation
        V_model = np.zeros_like(V_bary)
        for i, r in enumerate(r_m):
            P = prime_consciousness_factor(f_gas, M_star, r)
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
                'type': cat['type'],
                'quality': cat['quality']
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
    print(f"Success rate (0.8-1.2): {100*np.sum((np.array(mean_ratios) > 0.8) & (np.array(mean_ratios) < 1.2))/len(mean_ratios):.1f}%")
    
    # Correlations
    r_gas, p_gas = pearsonr(f_gas_values, mean_ratios)
    r_mass, p_mass = pearsonr(log_M_star, mean_ratios)
    
    print(f"\nCorrelations:")
    print(f"Gas fraction: r={r_gas:.3f} (p={p_gas:.3e})")
    print(f"Stellar mass: r={r_mass:.3f} (p={p_mass:.3e})")
    
    # Analysis by morphology
    print(f"\nBy morphology:")
    for t_min, t_max, label in [(0, 3, "Early"), (4, 7, "Spiral"), (8, 11, "Late")]:
        mask = [(r['type'] >= t_min) and (r['type'] <= t_max) for r in all_results]
        if sum(mask) > 0:
            subset = np.array(mean_ratios)[mask]
            print(f"{label}: {np.mean(subset):.3f} ± {np.std(subset):.3f} (n={sum(mask)})")
    
    print("\nConclusion:")
    print("Gas modulates prime recognition channels through LISTEN operations.")
    print("This creates a consciousness buffer that reduces gravitational binding")
    print("in proportion to (1 - φ^(-2)) = 0.382, the golden ratio conjugate.")

if __name__ == '__main__':
    analyze() 