#!/usr/bin/env python3
"""
LNAL Refined - Conservative improvements based on analysis
Focus on reducing scatter while maintaining median accuracy
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

def information_debt_refined(M_star, V_rot=None):
    """
    Refined Ψ with optional velocity-dependent term
    Addresses overestimate in massive galaxies
    """
    M0 = phi**-8 * M_sun
    if M_star <= 0:
        return 1.0
    
    # Base hierarchical depth
    N_raw = np.log(M_star / M0) / np.log(phi)
    
    # Apply saturation for massive galaxies (reduces overestimate)
    # Ψ saturates at high mass following tanh profile
    N_eff = N_raw * np.tanh(N_raw / 30.0)
    
    # Linear sum with golden ratio increment
    delta = phi**(1/8.0) - 1.0
    psi_base = 1.0 + N_eff * delta
    
    # Optional kinetic correction (small effect)
    if V_rot is not None:
        sigma = 0.7 * V_rot
        psi_kinetic = 1.0 + 0.5 * (sigma / c)**2
        return psi_base * psi_kinetic
    else:
        return psi_base

def molecular_gas_refined(M_star, M_HI):
    """
    Refined H₂ estimate - only for massive galaxies
    Low-mass galaxies have negligible H₂
    """
    if M_star < 1e9 * M_sun:
        return 0  # No H₂ in dwarfs
    
    # Simple linear scaling for massive galaxies
    # M_H2 / M_HI = 0.1 * (M_star / 10^10 M_sun)
    M_H2_ratio = 0.1 * (M_star / (1e10 * M_sun))
    M_H2_ratio = min(M_H2_ratio, 0.3)  # Cap at 30% of HI
    
    return M_H2_ratio * M_HI

def radial_correction(r, R_disk):
    """
    Radius-dependent correction to address inner overestimate
    Based on disk-halo transition
    """
    # Smooth transition from disk to halo
    x = r / (2.0 * R_disk)
    return 1.0 / (1.0 + 0.3 * np.exp(-x))

def lnal_refined(r_kpc, M_star, M_HI, R_disk_kpc):
    """
    Refined LNAL model with conservative improvements
    """
    r = r_kpc * kpc
    R_disk = R_disk_kpc * kpc
    
    # Add molecular gas only for massive galaxies
    M_H2 = molecular_gas_refined(M_star, M_HI)
    M_gas_total = M_HI + M_H2
    
    # Baryon completeness
    f_gas = M_gas_total / (M_star + M_gas_total) if (M_star + M_gas_total) > 0 else 0.0
    Xi = baryon_completeness(f_gas)
    
    # Refined information debt with saturation
    Psi = information_debt_refined(M_star)
    
    # Effective mass
    M_eff = (M_star + M_gas_total) * Xi * Psi
    
    # Prime sieve factor
    P = phi**(-0.5) * 8 / np.pi**2
    
    # MOND interpolation
    a_N = G * M_eff / r**2
    x = a_N / g_dagger
    mu = x / np.sqrt(1 + x**2)
    Lambda = mu + (1 - mu) * np.sqrt(g_dagger * r / (G * M_eff))
    
    # Apply radial correction
    Lambda *= radial_correction(r_kpc, R_disk_kpc)
    
    # Model velocity
    V2 = (G * M_eff / r) * Lambda * P
    return np.sqrt(max(V2, 0.0)) / 1000  # km/s

def analyze_refined():
    """
    Test refined model on SPARC data
    """
    # Load real SPARC data
    with open('sparc_real_data.pkl', 'rb') as f:
        sparc_data = pickle.load(f)
    
    print("=== LNAL REFINED ANALYSIS ===")
    print(f"Testing conservative improvements on {len(sparc_data)} galaxies")
    
    results_refined = []
    example_curves = []
    
    for i, (name, data) in enumerate(sparc_data.items()):
        catalog = data['catalog']
        curve = data['curve']
        
        # Galaxy parameters
        M_star = catalog['M_star'] * 1e9 * M_sun
        M_HI = catalog['M_HI'] * 1e9 * M_sun
        R_disk = catalog['R_disk']  # kpc
        
        # Observed data
        r_obs = curve['r']
        V_obs = curve['V_obs']
        e_V = curve['e_V']
        
        # Model predictions
        V_model = np.array([lnal_refined(r, M_star, M_HI, R_disk) for r in r_obs])
        
        # Statistics
        mask = V_obs > 20
        if np.sum(mask) > 0:
            ratios = V_model[mask] / V_obs[mask]
            mean_ratio = np.mean(ratios)
            median_ratio = np.median(ratios)
            
            # Chi-squared
            residuals = (V_model[mask] - V_obs[mask]) / e_V[mask]
            chi2_reduced = np.sum(residuals**2) / len(residuals)
        else:
            mean_ratio = median_ratio = chi2_reduced = np.nan
        
        result = {
            'name': name,
            'mean_ratio': mean_ratio,
            'median_ratio': median_ratio,
            'chi2_reduced': chi2_reduced,
            'f_gas': M_HI / (M_star + M_HI),
            'M_star': M_star,
            'quality': catalog['quality'],
            'V_model': V_model,
            'V_obs': V_obs,
            'r': r_obs
        }
        results_refined.append(result)
        
        # Save examples
        if i < 6:
            example_curves.append(result)
    
    # Filter valid results
    valid_results = [r for r in results_refined if not np.isnan(r['mean_ratio'])]
    
    # Extract statistics
    mean_ratios = [r['mean_ratio'] for r in valid_results]
    median_ratios = [r['median_ratio'] for r in valid_results]
    chi2_all = [r['chi2_reduced'] for r in valid_results]
    
    print(f"\n=== REFINED RESULTS ===")
    print(f"Mean V_model/V_obs = {np.mean(mean_ratios):.3f} ± {np.std(mean_ratios):.3f}")
    print(f"Median V_model/V_obs = {np.median(median_ratios):.3f}")
    print(f"Mean χ²/ν = {np.mean(chi2_all):.1f}")
    print(f"Success rate (0.8 < ratio < 1.2): {np.sum((np.array(mean_ratios) > 0.8) & (np.array(mean_ratios) < 1.2)) / len(mean_ratios) * 100:.1f}%")
    
    # By quality
    for q in [1, 2, 3]:
        mask = [r['quality'] == q for r in valid_results]
        if sum(mask) > 0:
            ratios_q = np.array(mean_ratios)[mask]
            print(f"\nQuality {q} ({sum(mask)} galaxies):")
            print(f"  Mean ratio = {np.mean(ratios_q):.3f} ± {np.std(ratios_q):.3f}")
    
    # Save results
    with open('lnal_refined_results.pkl', 'wb') as f:
        pickle.dump(valid_results, f)
    
    # Plot examples
    plot_example_curves(example_curves)
    
    return valid_results

def plot_example_curves(examples):
    """
    Plot example rotation curves
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(examples[:6]):
        ax = axes[i]
        
        # Plot data
        ax.scatter(result['r'], result['V_obs'], color='black', s=20, alpha=0.7, label='Observed')
        ax.plot(result['r'], result['V_model'], 'r-', linewidth=2, label='LNAL Refined')
        
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Velocity (km/s)')
        ax.set_title(f"{result['name']}\nratio={result['mean_ratio']:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lnal_refined_examples.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    results = analyze_refined() 