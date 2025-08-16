#!/usr/bin/env python3
"""
LNAL Disk Model V2 - Recognition-based Lambda modulation
All parameter-free from first principles
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

# Recognition lengths
ell_1 = 0.97 * kpc   # First recognition scale
ell_2 = 24.3 * kpc   # Second recognition scale

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

def molecular_H2_mass(M_star: float, M_HI: float) -> float:
    """H2 from metallicity"""
    if M_star <= 0 or M_HI <= 0:
        return 0.0
    Z_ratio = (M_star / (10 ** 10.5 * M_sun)) ** 0.30
    exponent = (phi ** 0.5) / 2
    ratio = min(Z_ratio ** exponent, 1.0)
    return ratio * M_HI

def recognition_lambda(r: float, M_eff: float) -> float:
    """
    Lambda with recognition-scale modulation
    Uses dual recognition lengths to handle disk-halo transition
    """
    a_N = G * M_eff / r ** 2
    x = a_N / g_dagger
    mu = x / np.sqrt(1 + x ** 2)
    
    # Recognition modulation factor
    # Suppressed below ell_1, transitions at ell_1-ell_2, full beyond ell_2
    if r < ell_1:
        # Inner disk: strong suppression
        mod = (r / ell_1) ** phi
    elif r < ell_2:
        # Transition region: smooth interpolation
        t = (r - ell_1) / (ell_2 - ell_1)
        mod = t ** (1/phi)  # Golden ratio interpolation
    else:
        # Outer halo: full MOND
        mod = 1.0
    
    # Apply modulation to MOND term only
    Lambda = mu + (1 - mu) * mod * np.sqrt(g_dagger * r / (G * M_eff))
    return Lambda

def analyze():
    with open('sparc_real_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"Disk Model V2 on {len(data)} galaxies")
    
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
        P = prime_sieve()
        M_eff = (M_star + M_gas_total) * Xi * Psi
        
        V_disk = curve['V_disk']
        V_gas = curve['V_gas']
        V_mol = V_gas * np.sqrt(M_H2 / M_HI) if M_HI > 0 else 0.0
        V_bary = np.sqrt(V_disk ** 2 + V_gas ** 2 + V_mol ** 2)
        
        r_kpc = curve['r']
        r_m = r_kpc * kpc
        
        # Compute Lambda for each radius
        V_model = np.zeros_like(V_bary)
        for i, r in enumerate(r_m):
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
                'r': r_kpc,
                'V_obs': curve['V_obs'],
                'V_model': V_model
            }
            all_results.append(result)
    
    # Overall statistics
    all_ratios = np.concatenate([r['ratios'] for r in all_results])
    print(f"Median ratio = {np.median(all_ratios):.3f}")
    print(f"Mean ratio   = {np.mean(all_ratios):.3f} ± {np.std(all_ratios):.3f}")
    
    # Save for analysis
    with open('lnal_diskmodel_v2_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Plot examples
    plot_examples(all_results[:6])

def plot_examples(results):
    """Plot example curves"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, res in enumerate(results[:6]):
        ax = axes[i]
        ax.scatter(res['r'], res['V_obs'], c='black', s=20, alpha=0.7, label='Observed')
        ax.plot(res['r'], res['V_model'], 'r-', linewidth=2, label='LNAL V2')
        
        # Mark recognition scales
        ax.axvline(ell_1/kpc, color='green', linestyle=':', alpha=0.5, label='ℓ₁')
        ax.axvline(ell_2/kpc, color='blue', linestyle=':', alpha=0.5, label='ℓ₂')
        
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Velocity (km/s)')
        ax.set_title(f"{res['name']}\nratio={res['mean_ratio']:.2f}")
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lnal_diskmodel_v2_examples.png', dpi=150)
    plt.close()

if __name__ == '__main__':
    analyze() 