#!/usr/bin/env python3
"""
LNAL Final Analysis - Real SPARC Rotation Curves
Zero free parameters, complete Recognition Science implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from scipy.stats import pearsonr

# Physical constants from Recognition Science
phi = (1 + np.sqrt(5)) / 2
G = 6.67430e-11
c = 2.99792458e8
M_sun = 1.98847e30
kpc = 3.0856775814913673e19
g_dagger = 1.2e-10  # From eight-beat recognition

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
    L_0 = 0.335e-9  # Recognition voxel size
    N_limit = np.log(R_s / L_0) / np.log(phi) if R_s > 0 else N_raw
    N = min(max(0.0, N_raw), max(0.0, N_limit))
    
    # Linear sum with golden ratio increment
    delta = phi**(1/8.0) - 1.0
    return 1.0 + N * delta

def lnal_velocity(r_kpc, M_star, M_HI):
    """
    Complete LNAL model velocity at radius r
    """
    r = r_kpc * kpc  # Convert to meters
    
    # Baryon completeness and information debt
    f_gas = M_HI / (M_star + M_HI) if (M_star + M_HI) > 0 else 0.0
    Xi = baryon_completeness(f_gas)
    Psi = information_debt(M_star)
    M_eff = (M_star + M_HI) * Xi * Psi
    
    # Prime sieve factor (odd square-free density)
    P = phi**(-0.5) * 8 / np.pi**2
    
    # MOND interpolation function
    a_N = G * M_eff / r**2
    x = a_N / g_dagger
    mu = x / np.sqrt(1 + x**2)
    Lambda = mu + (1 - mu) * np.sqrt(g_dagger * r / (G * M_eff))
    
    # Model velocity
    V2 = (G * M_eff / r) * Lambda * P
    return np.sqrt(max(V2, 0.0)) / 1000  # m/s to km/s

def analyze_galaxy(galaxy_name, data):
    """
    Analyze one galaxy's rotation curve
    """
    catalog = data['catalog']
    curve = data['curve']
    
    # Galaxy parameters
    M_star = catalog['M_star'] * 1e9 * M_sun  # kg
    M_HI = catalog['M_HI'] * 1e9 * M_sun      # kg
    
    # Observed data
    r_obs = curve['r']     # kpc
    V_obs = curve['V_obs'] # km/s
    e_V = curve['e_V']     # km/s
    
    # Model predictions
    V_model = np.array([lnal_velocity(r, M_star, M_HI) for r in r_obs])
    
    # Statistics
    residuals = (V_model - V_obs) / e_V
    chi2 = np.sum(residuals**2)
    ndof = len(r_obs)
    chi2_reduced = chi2 / ndof if ndof > 0 else 0
    
    # Velocity ratios (exclude very low velocities)
    mask = V_obs > 20  # km/s
    if np.sum(mask) > 0:
        ratios = V_model[mask] / V_obs[mask]
        mean_ratio = np.mean(ratios)
        median_ratio = np.median(ratios)
    else:
        mean_ratio = median_ratio = np.nan
    
    return {
        'name': galaxy_name,
        'M_star': M_star,
        'M_HI': M_HI,
        'f_gas': M_HI / (M_star + M_HI) if (M_star + M_HI) > 0 else 0,
        'quality': catalog['quality'],
        'V_flat': catalog['V_flat'],
        'r': r_obs,
        'V_obs': V_obs,
        'e_V': e_V,
        'V_model': V_model,
        'chi2_reduced': chi2_reduced,
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        'n_points': len(r_obs)
    }

def main_analysis():
    """
    Main analysis of all SPARC galaxies
    """
    # Load real SPARC data
    with open('sparc_real_data.pkl', 'rb') as f:
        sparc_data = pickle.load(f)
    
    print("=== LNAL FINAL ANALYSIS - REAL SPARC DATA ===")
    print(f"Analyzing {len(sparc_data)} galaxies with real rotation curves")
    
    # Analyze each galaxy
    results = []
    for name, data in sparc_data.items():
        result = analyze_galaxy(name, data)
        results.append(result)
    
    # Filter out invalid results
    valid_results = [r for r in results if not np.isnan(r['mean_ratio'])]
    print(f"Valid results: {len(valid_results)} galaxies")
    
    # Extract statistics
    chi2_all = [r['chi2_reduced'] for r in valid_results]
    mean_ratios = [r['mean_ratio'] for r in valid_results]
    median_ratios = [r['median_ratio'] for r in valid_results]
    f_gas_all = [r['f_gas'] for r in valid_results]
    M_star_all = [r['M_star'] / M_sun for r in valid_results]
    quality_all = [r['quality'] for r in valid_results]
    
    # Overall statistics
    print(f"\n=== OVERALL RESULTS ===")
    print(f"Mean χ²/ν = {np.mean(chi2_all):.2f} ± {np.std(chi2_all):.2f}")
    print(f"Median χ²/ν = {np.median(chi2_all):.2f}")
    print(f"\nMean V_model/V_obs = {np.mean(mean_ratios):.3f} ± {np.std(mean_ratios):.3f}")
    print(f"Median V_model/V_obs = {np.median(median_ratios):.3f} ± {np.std(median_ratios):.3f}")
    
    # By quality flag
    for q in [1, 2, 3]:
        mask = [r['quality'] == q for r in valid_results]
        if sum(mask) > 0:
            chi2_q = np.array(chi2_all)[mask]
            ratio_q = np.array(mean_ratios)[mask]
            print(f"\nQuality {q} ({sum(mask)} galaxies):")
            print(f"  χ²/ν = {np.mean(chi2_q):.2f} ± {np.std(chi2_q):.2f}")
            print(f"  V_ratio = {np.mean(ratio_q):.3f} ± {np.std(ratio_q):.3f}")
    
    # Correlations
    r_gas, p_gas = pearsonr(f_gas_all, mean_ratios)
    r_mass, p_mass = pearsonr(np.log10(M_star_all), mean_ratios)
    
    print(f"\n=== CORRELATIONS ===")
    print(f"Gas fraction vs ratio: r = {r_gas:.3f} (p = {p_gas:.3f})")
    print(f"log(M*) vs ratio: r = {r_mass:.3f} (p = {p_mass:.3f})")
    
    # Save results
    with open('lnal_final_results.pkl', 'wb') as f:
        pickle.dump(valid_results, f)
    
    # Create plots
    plot_results(valid_results)
    
    return valid_results

def plot_results(results):
    """
    Create diagnostic plots
    """
    # Extract data for plotting
    mean_ratios = [r['mean_ratio'] for r in results]
    chi2_all = [r['chi2_reduced'] for r in results]
    f_gas_all = [r['f_gas'] for r in results]
    M_star_all = [r['M_star'] / M_sun for r in results]
    quality_all = [r['quality'] for r in results]
    
    # Main results plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Velocity ratio histogram
    ax = axes[0, 0]
    ax.hist(mean_ratios, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(1.0, color='red', linestyle='--', label='Perfect')
    ax.axvline(np.mean(mean_ratios), color='blue', linestyle='-',
               label=f'Mean={np.mean(mean_ratios):.3f}')
    ax.set_xlabel('V_model / V_obs')
    ax.set_ylabel('Count')
    ax.set_title('Velocity Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Chi-squared histogram
    ax = axes[0, 1]
    ax.hist(chi2_all, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(1.0, color='red', linestyle='--', label='Expected')
    ax.axvline(np.median(chi2_all), color='blue', linestyle='-',
               label=f'Median={np.median(chi2_all):.1f}')
    ax.set_xlabel('χ²/ν')
    ax.set_ylabel('Count')
    ax.set_title('Reduced Chi-Squared Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gas fraction correlation
    ax = axes[1, 0]
    colors = ['red', 'orange', 'green']
    for q in [1, 2, 3]:
        mask = np.array(quality_all) == q
        if np.sum(mask) > 0:
            ax.scatter(np.array(f_gas_all)[mask], np.array(mean_ratios)[mask], 
                      c=colors[q-1], label=f'Quality {q}', alpha=0.7)
    ax.set_xlabel('Gas Fraction')
    ax.set_ylabel('V_model / V_obs')
    ax.set_title('Gas Fraction vs Velocity Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stellar mass correlation
    ax = axes[1, 1]
    for q in [1, 2, 3]:
        mask = np.array(quality_all) == q
        if np.sum(mask) > 0:
            ax.scatter(np.log10(np.array(M_star_all)[mask]), np.array(mean_ratios)[mask],
                      c=colors[q-1], label=f'Quality {q}', alpha=0.7)
    ax.set_xlabel('log(M* / M_sun)')
    ax.set_ylabel('V_model / V_obs')
    ax.set_title('Stellar Mass vs Velocity Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lnal_final_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Example curves
    plot_example_curves(results)

def plot_example_curves(results, n_examples=6):
    """
    Plot example rotation curves
    """
    # Sort by chi2 and pick representative examples
    sorted_results = sorted(results, key=lambda x: x['chi2_reduced'])
    indices = [0, len(results)//4, len(results)//2, 3*len(results)//4, -2, -1]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i >= len(axes) or idx >= len(sorted_results):
            break
            
        result = sorted_results[idx]
        ax = axes[i]
        
        # Plot observed data with error bars
        ax.errorbar(result['r'], result['V_obs'], yerr=result['e_V'],
                   fmt='o', color='black', markersize=4, alpha=0.7,
                   label='Observed')
        
        # Plot LNAL model
        ax.plot(result['r'], result['V_model'], 'r-', linewidth=2,
               label='LNAL Model')
        
        # Formatting
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Velocity (km/s)')
        ax.set_title(f"{result['name']}\nχ²/ν={result['chi2_reduced']:.1f}, "
                    f"ratio={result['mean_ratio']:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lnal_example_curves_real.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    results = main_analysis() 