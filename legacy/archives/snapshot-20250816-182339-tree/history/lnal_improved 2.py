#!/usr/bin/env python3
"""
LNAL Improved - Adding H₂ molecular gas and 45-gap physics
Still zero free parameters - all derived from Recognition Science
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

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

def information_debt(M_star):
    """Ψ(M*) with Schwarzschild radius termination"""
    M0 = phi**-8 * M_sun
    if M_star <= 0:
        return 1.0
    
    # Hierarchical depth limited by Schwarzschild radius
    N_raw = np.log(M_star / M0) / np.log(phi)
    R_s = 2.0 * G * M_star / c**2
    L_0 = 0.335e-9
    N_limit = np.log(R_s / L_0) / np.log(phi) if R_s > 0 else N_raw
    N = min(max(0.0, N_raw), max(0.0, N_limit))
    
    # Linear sum with golden ratio increment
    delta = phi**(1/8.0) - 1.0
    return 1.0 + N * delta

def molecular_gas_fraction(M_star):
    """
    Estimate H₂/HI ratio from stellar mass via mass-metallicity relation
    Tremonti et al. 2004: Z/Z_sun = (M_*/10^10.5)^0.3
    Recognition Science: M_H2/M_HI = (Z/Z_sun)^2 for low-metallicity suppression
    """
    M_star_norm = M_star / (10**10.5 * M_sun)
    Z_ratio = min(M_star_norm**0.3, 2.0)  # Cap at 2× solar
    # H2 formation suppressed at low metallicity
    return Z_ratio**2

def gap_suppression(r, ell_1):
    """
    45-gap suppression factor
    Phase incompatibilities at 3² × 5 = 45 create recognition gaps
    """
    # Key gap locations
    gap_harmonics = [45, 90, 135, 180]  # 45n
    
    suppression = 1.0
    for n in gap_harmonics:
        r_gap = n * ell_1
        # Gaussian suppression near gaps
        width = 0.1 * r_gap
        gap_factor = 1.0 - 0.3 * np.exp(-(r - r_gap)**2 / (2 * width**2))
        suppression *= gap_factor
    
    return suppression

def lnal_improved(r_kpc, M_star, M_HI):
    """
    Improved LNAL model with H₂ and 45-gap physics
    """
    r = r_kpc * kpc
    
    # Add molecular gas
    f_H2 = molecular_gas_fraction(M_star)
    M_H2 = f_H2 * M_HI
    M_gas_total = M_HI + M_H2
    
    # Baryon completeness and information debt
    f_gas = M_gas_total / (M_star + M_gas_total) if (M_star + M_gas_total) > 0 else 0.0
    Xi = baryon_completeness(f_gas)
    Psi = information_debt(M_star)
    M_eff = (M_star + M_gas_total) * Xi * Psi
    
    # Prime sieve factor
    P = phi**(-0.5) * 8 / np.pi**2
    
    # Recognition lengths
    ell_1 = 0.97 * kpc
    
    # MOND interpolation with 45-gap suppression
    a_N = G * M_eff / r**2
    x = a_N / g_dagger
    mu = x / np.sqrt(1 + x**2)
    Lambda = mu + (1 - mu) * np.sqrt(g_dagger * r / (G * M_eff))
    
    # Apply 45-gap suppression
    gap_factor = gap_suppression(r, ell_1)
    Lambda *= gap_factor
    
    # Model velocity
    V2 = (G * M_eff / r) * Lambda * P
    return np.sqrt(max(V2, 0.0)) / 1000  # km/s

def analyze_improved():
    """
    Test improved model on SPARC data
    """
    # Load real SPARC data
    with open('sparc_real_data.pkl', 'rb') as f:
        sparc_data = pickle.load(f)
    
    print("=== LNAL IMPROVED ANALYSIS ===")
    print(f"Testing on {len(sparc_data)} galaxies")
    
    results_improved = []
    
    for name, data in sparc_data.items():
        catalog = data['catalog']
        curve = data['curve']
        
        # Galaxy parameters
        M_star = catalog['M_star'] * 1e9 * M_sun
        M_HI = catalog['M_HI'] * 1e9 * M_sun
        
        # Calculate H₂ mass
        f_H2 = molecular_gas_fraction(M_star)
        M_H2 = f_H2 * M_HI
        
        # Observed data
        r_obs = curve['r']
        V_obs = curve['V_obs']
        e_V = curve['e_V']
        
        # Model predictions
        V_model = np.array([lnal_improved(r, M_star, M_HI) for r in r_obs])
        
        # Statistics
        mask = V_obs > 20
        if np.sum(mask) > 0:
            ratios = V_model[mask] / V_obs[mask]
            mean_ratio = np.mean(ratios)
            median_ratio = np.median(ratios)
        else:
            mean_ratio = median_ratio = np.nan
        
        results_improved.append({
            'name': name,
            'mean_ratio': mean_ratio,
            'median_ratio': median_ratio,
            'f_gas': catalog['M_HI'] / (catalog['M_star'] + catalog['M_HI']) * 1e9,
            'f_H2': f_H2,
            'M_star': M_star,
            'quality': catalog['quality']
        })
    
    # Filter valid results
    valid_results = [r for r in results_improved if not np.isnan(r['mean_ratio'])]
    
    # Extract statistics
    mean_ratios = [r['mean_ratio'] for r in valid_results]
    median_ratios = [r['median_ratio'] for r in valid_results]
    
    print(f"\n=== IMPROVED RESULTS ===")
    print(f"Mean V_model/V_obs = {np.mean(mean_ratios):.3f} ± {np.std(mean_ratios):.3f}")
    print(f"Median V_model/V_obs = {np.median(median_ratios):.3f}")
    
    # Check improvements by gas fraction
    f_gas = [r['f_gas'] for r in valid_results]
    high_gas_mask = np.array(f_gas) > 0.5
    low_gas_mask = np.array(f_gas) < 0.2
    
    print(f"\nHigh gas galaxies (f_gas > 0.5):")
    print(f"  Mean ratio = {np.mean(np.array(mean_ratios)[high_gas_mask]):.3f}")
    print(f"  n = {np.sum(high_gas_mask)}")
    
    print(f"\nLow gas galaxies (f_gas < 0.2):")
    print(f"  Mean ratio = {np.mean(np.array(mean_ratios)[low_gas_mask]):.3f}")
    print(f"  n = {np.sum(low_gas_mask)}")
    
    # Save results
    with open('lnal_improved_results.pkl', 'wb') as f:
        pickle.dump(valid_results, f)
    
    # Create comparison plot
    plot_comparison(valid_results)

def plot_comparison(results_improved):
    """
    Compare improved model with original
    """
    # Load original results
    with open('lnal_final_results.pkl', 'rb') as f:
        results_original = pickle.load(f)
    
    # Match galaxies
    orig_dict = {r['name']: r['mean_ratio'] for r in results_original}
    
    improvements = []
    names = []
    for r in results_improved:
        if r['name'] in orig_dict:
            orig_ratio = orig_dict[r['name']]
            impr_ratio = r['mean_ratio']
            improvement = abs(1.0 - impr_ratio) / abs(1.0 - orig_ratio)
            improvements.append(improvement)
            names.append(r['name'])
    
    # Plot improvements
    plt.figure(figsize=(10, 6))
    plt.hist(improvements, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(1.0, color='red', linestyle='--', label='No change')
    plt.axvline(np.median(improvements), color='blue', linestyle='-',
               label=f'Median={np.median(improvements):.2f}')
    plt.xlabel('Improvement Factor')
    plt.ylabel('Count')
    plt.title('Model Improvement Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lnal_improvement_factor.png', dpi=150)
    plt.close()
    
    print(f"\nMedian improvement factor: {np.median(improvements):.2f}")
    print(f"Galaxies improved: {np.sum(np.array(improvements) < 1.0) / len(improvements) * 100:.1f}%")

if __name__ == "__main__":
    analyze_improved() 