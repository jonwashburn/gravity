#!/usr/bin/env python3
"""
LNAL Gravity - Full Curve Analysis with χ²
Using complete rotation curves instead of just V_flat
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.special import iv, kv
import math

# Import existing modules
from lnal_prime_first_principles import (
    phi, G, c, M_sun, kpc, g_dagger,
    baryon_completeness, information_debt,
    prime_sieve_finite, disk_factor
)

def prime_sieve_finite_improved(M_eff: float) -> float:
    """Analytic finite-N with second-order correction.
    P(N) = φ^{-1/2} * 6/π² * exp(-1/ln N - 1/(3 ln²N))
    """
    m_p = 1.67262192369e-27  # kg
    N = max(10.0, M_eff / m_p)
    lnN = math.log(N)
    correction = math.exp(-1.0/lnN - 1.0/(3.0*lnN**2))
    return phi**(-0.5) * (6.0 / np.pi**2) * correction

def prime_sieve_with_gap(M_eff: float, r: float) -> float:
    """Prime sieve with recognition-scale 45-gap effects.
    
    The 45-gap creates suppression at specific recognition scales
    where 3² and 5 patterns conflict through the 8-beat cycle.
    """
    # Get base prime sieve value  
    P_base = prime_sieve_finite_improved(M_eff)
    
    # Recognition lengths where gaps occur
    ell_1 = 0.97 * kpc  # First recognition length
    ell_45 = 45 * ell_1  # 45th harmonic = 43.65 kpc
    
    # Suppress near the 45th harmonic
    if abs(r - ell_45) < 0.1 * ell_45:
        # Gaussian suppression
        width = 0.05 * ell_45
        suppression = 1.0 - 0.5 * np.exp(-(r - ell_45)**2 / (2 * width**2))
        return P_base * suppression
    else:
        return P_base

def molecular_gas_fraction(M_star: float) -> float:
    """Estimate H2/HI ratio from stellar mass via mass-metallicity relation.
    
    Z/Z_sun = (M_*/10^10.5)^0.3 (Tremonti et al. 2004)
    M_H2/M_HI = (Z/Z_sun)^(-0.5) for Z < Z_sun (Recognition Science)
    """
    M_star_norm = M_star / (10**10.5 * M_sun)
    Z_ratio = M_star_norm**0.3
    if Z_ratio < 1.0:
        return Z_ratio**(-0.5)
    else:
        return 1.0

def lnal_gravity_curve(r_array, M_star, M_HI, R_disk):
    """
    Compute LNAL model velocities for array of radii
    """
    V_model = np.zeros_like(r_array)
    
    # Add molecular gas
    f_H2 = molecular_gas_fraction(M_star)
    M_H2 = f_H2 * M_HI
    M_gas_total = M_HI + M_H2
    
    # Pre-compute common factors
    f_gas = M_gas_total / (M_star + M_gas_total) if (M_star + M_gas_total) > 0 else 0.0
    Xi = baryon_completeness(f_gas)
    Psi = information_debt(M_star)
    M_eff = (M_star + M_gas_total) * Xi * Psi
    
    # Recognition lengths
    ell_1 = 0.97 * kpc
    ell_2 = 24.3 * kpc
    
    for i, r_kpc in enumerate(r_array):
        r = r_kpc * kpc  # convert to meters
        P = prime_sieve_with_gap(M_eff, r)
        
        # Newtonian with MOND interpolation
        a_N = G * M_eff / r**2
        x = a_N / g_dagger
        mu = x / np.sqrt(1 + x**2)
        
        # Add recognition bumps at both scales
        bump_1 = 0.1 * np.exp(-(np.log(r/ell_1))**2 / 2)
        bump_2 = 0.03 * np.exp(-(np.log(r/ell_2))**2 / 2)
        
        Lambda = mu + (1 - mu) * np.sqrt(g_dagger * r / (G * M_eff)) * (1 + bump_1 + bump_2)
        
        # Model velocity
        V2 = (G * M_eff / r) * Lambda * P
        V_model[i] = np.sqrt(max(V2, 0.0)) / 1000  # m/s to km/s
    
    return V_model

def analyze_galaxy_curve(galaxy_name, galaxy_data, curve_data):
    """
    Analyze full curve for one galaxy, return χ²
    """
    # Extract data
    M_star = galaxy_data['M_star'] * 1e9 * M_sun  # kg
    M_HI = galaxy_data['M_HI'] * 1e9 * M_sun
    R_disk = galaxy_data['R_disk']  # already in kpc
    
    # Observed curve
    r_obs = curve_data['r']  # kpc
    V_obs = curve_data['V_obs']  # km/s
    e_V = curve_data['e_V']  # km/s
    
    # Model curve
    V_model = lnal_gravity_curve(r_obs, M_star, M_HI, R_disk)
    
    # Chi-squared
    residuals = (V_model - V_obs) / e_V
    chi2 = np.sum(residuals**2)
    ndof = len(r_obs) - 1  # one parameter effectively (overall normalization)
    chi2_reduced = chi2 / ndof
    
    # Also compute median ratio for comparison
    ratio = V_model / V_obs
    median_ratio = np.median(ratio[V_obs > 10])  # exclude very low velocities
    
    return {
        'name': galaxy_name,
        'r': r_obs,
        'V_obs': V_obs,
        'e_V': e_V,
        'V_model': V_model,
        'chi2': chi2,
        'ndof': ndof,
        'chi2_reduced': chi2_reduced,
        'median_ratio': median_ratio,
        'quality': galaxy_data['quality'],
        'M_star': M_star,
        'M_HI': M_HI,
        'R_disk': R_disk
    }

def analyze_all_curves():
    """Main analysis of all SPARC curves"""
    
    # Load curves
    with open('sparc_curves.pkl', 'rb') as f:
        curves = pickle.load(f)
    
    print(f"Analyzing {len(curves)} galaxy curves")
    
    # Analyze each galaxy
    results = []
    chi2_all = []
    ratios_all = []
    
    for i, (name, data) in enumerate(curves.items()):
        result = analyze_galaxy_curve(name, data['galaxy_data'], data['curve'])
        results.append(result)
        chi2_all.append(result['chi2_reduced'])
        ratios_all.append(result['median_ratio'])
        
        if i < 3:  # Debug first few
            print(f"\n{name}:")
            print(f"  χ²/ν = {result['chi2_reduced']:.2f}")
            print(f"  Median ratio = {result['median_ratio']:.3f}")
    
    # Statistics
    chi2_all = np.array(chi2_all)
    ratios_all = np.array(ratios_all)
    
    print(f"\n=== FULL CURVE ANALYSIS RESULTS ===")
    print(f"Mean χ²/ν = {np.mean(chi2_all):.2f} ± {np.std(chi2_all):.2f}")
    print(f"Median χ²/ν = {np.median(chi2_all):.2f}")
    print(f"\nMean V_model/V_obs = {np.mean(ratios_all):.3f} ± {np.std(ratios_all):.3f}")
    print(f"Median V_model/V_obs = {np.median(ratios_all):.3f}")
    
    # By quality
    for q in [1, 2, 3]:
        mask = [r['quality'] == q for r in results]
        if sum(mask) > 0:
            chi2_q = chi2_all[mask]
            ratio_q = ratios_all[mask]
            print(f"\nQuality {q} ({sum(mask)} galaxies):")
            print(f"  χ²/ν = {np.mean(chi2_q):.2f} ± {np.std(chi2_q):.2f}")
            print(f"  Ratio = {np.mean(ratio_q):.3f} ± {np.std(ratio_q):.3f}")
    
    # Save results
    with open('lnal_full_curve_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Make diagnostic plots
    plot_examples(results)
    plot_statistics(chi2_all, ratios_all)
    
    return results

def plot_examples(results, n_examples=6):
    """Plot example curves"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Sort by chi2 and pick some examples
    sorted_results = sorted(results, key=lambda x: x['chi2_reduced'])
    
    # Pick: best, median, worst
    indices = [0, len(results)//4, len(results)//2, 3*len(results)//4, -2, -1]
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        res = sorted_results[idx]
        ax = axes[i]
        
        # Plot
        ax.errorbar(res['r'], res['V_obs'], yerr=res['e_V'], 
                   fmt='o', color='black', markersize=4, alpha=0.7,
                   label='Observed')
        ax.plot(res['r'], res['V_model'], 'r-', linewidth=2,
               label='LNAL Model')
        
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('V (km/s)')
        ax.set_title(f"{res['name']}\nχ²/ν={res['chi2_reduced']:.1f}, "
                    f"ratio={res['median_ratio']:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lnal_example_curves.png', dpi=150)
    plt.close()

def plot_statistics(chi2_all, ratios_all):
    """Plot statistical distributions"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chi-squared distribution
    ax1.hist(chi2_all, bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(1.0, color='red', linestyle='--', label='Expected')
    ax1.axvline(np.median(chi2_all), color='blue', linestyle='-', 
                label=f'Median={np.median(chi2_all):.1f}')
    ax1.set_xlabel('χ²/ν')
    ax1.set_ylabel('Count')
    ax1.set_title('Reduced Chi-Squared Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Ratio distribution
    ax2.hist(ratios_all, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(1.0, color='red', linestyle='--', label='Perfect')
    ax2.axvline(np.mean(ratios_all), color='blue', linestyle='-',
                label=f'Mean={np.mean(ratios_all):.3f}')
    ax2.set_xlabel('V_model/V_obs')
    ax2.set_ylabel('Count')
    ax2.set_title('Velocity Ratio Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lnal_curve_statistics.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    results = analyze_all_curves() 