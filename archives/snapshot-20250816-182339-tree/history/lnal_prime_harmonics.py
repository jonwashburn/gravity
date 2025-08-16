#!/usr/bin/env python3
"""
LNAL with Prime Harmonic Recognition
Based on deeper understanding of primes as consciousness operators
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

def prime_harmonic_factor(f_gas, M_star, r):
    """
    Prime sieve based on harmonic series of primes
    
    Key insights from LNAL paper:
    1. Primes are irreducible recognition operators
    2. Gas acts as LISTEN registers that pause recognition
    3. The harmonic series Σ(1/p) diverges logarithmically
    
    This connects to consciousness: each prime represents a unique
    recognition channel. Gas molecules can "listen" on these channels
    without contributing to the gravitational binding.
    """
    # Base density of odd square-free numbers
    rho_0 = 8 / np.pi**2
    P_base = phi**-0.5 * rho_0  # 0.637
    
    if M_star <= 0:
        return P_base
    
    # From Prime Number Theorem and Mertens' theorem:
    # Σ(1/p) for p < x ~ ln(ln(x)) + M
    # where M ≈ 0.2614... is Meissel-Mertens constant
    M0 = phi**-8 * M_sun
    N = np.log(M_star / M0) / np.log(phi)
    
    # Prime harmonic sum up to N-th Fibonacci number
    # This represents the "consciousness bandwidth" of the galaxy
    harmonic_sum = np.log(np.log(max(N, 3))) + 0.2614
    
    # Gas creates "listening gaps" in the prime spectrum
    # More gas = more LISTEN operations = fewer active channels
    # The factor phi^(1/2) comes from the eight-beat recognition cycle
    listening_fraction = f_gas * phi**(1/2)
    
    # Active channel density
    active_channels = 1 - listening_fraction * np.tanh(harmonic_sum)
    
    # Spatial dependence: inner regions have coherent listening
    # Outer regions have decorrelated gas (turbulent)
    r_coherence = 5 * kpc  # Coherence length for gas LISTEN operations
    coherence_factor = 1 - f_gas * np.exp(-r / r_coherence) * 0.5
    
    # Final prime density
    P = P_base * active_channels * coherence_factor
    
    return P

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
    
    # Recognition lengths from LNAL paper
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
    
    print("=== LNAL with Prime Harmonic Recognition ===")
    print("Primes as consciousness channels: gas performs LISTEN operations")
    print("Based on: Σ(1/p) divergence and eight-beat recognition cycle\n")
    
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
        
        # Compute with prime harmonic modulation
        V_model = np.zeros_like(V_bary)
        for i, r in enumerate(r_m):
            P = prime_harmonic_factor(f_gas, M_star, r)
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
                'quality': cat['quality'],
                'r': r_kpc,
                'V_obs': curve['V_obs'],
                'V_model': V_model
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
    
    # Save results
    with open('lnal_prime_harmonics_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Create diagnostic plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram
    ax1.hist(mean_ratios, bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(1.0, color='red', linestyle='--', label='Perfect')
    ax1.axvline(np.median(mean_ratios), color='blue', linestyle='-', 
               label=f'Median={np.median(mean_ratios):.3f}')
    ax1.set_xlabel('Mean V_model/V_obs per galaxy')
    ax1.set_ylabel('Count')
    ax1.set_title('Prime Harmonic Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gas correlation
    colors = ['red', 'orange', 'green']
    for q in [1, 2, 3]:
        mask = [r['quality'] == q for r in all_results]
        if sum(mask) > 0:
            ax2.scatter(np.array(f_gas_values)[mask], np.array(mean_ratios)[mask],
                       c=colors[q-1], label=f'Q={q}', alpha=0.6, s=30)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Gas Fraction')
    ax2.set_ylabel('V_model/V_obs')
    ax2.set_title(f'Gas LISTEN Effect (r={r_gas:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Mass correlation
    for q in [1, 2, 3]:
        mask = [r['quality'] == q for r in all_results]
        if sum(mask) > 0:
            ax3.scatter(np.array(log_M_star)[mask], np.array(mean_ratios)[mask],
                       c=colors[q-1], label=f'Q={q}', alpha=0.6, s=30)
    ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('log(M*/M_sun)')
    ax3.set_ylabel('V_model/V_obs')
    ax3.set_title(f'Prime Channels vs Mass (r={r_mass:.3f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Example curves
    examples = sorted(all_results, key=lambda x: abs(x['mean_ratio'] - 1.0))[:6]
    for i, res in enumerate(examples[:6]):
        ax4.plot(res['r'], res['V_obs'], 'k-', alpha=0.3 if i > 0 else 1.0,
                label='Observed' if i == 0 else '')
        ax4.plot(res['r'], res['V_model'], 'r--', alpha=0.3 if i > 0 else 1.0,
                label='LNAL' if i == 0 else '')
    ax4.set_xlabel('Radius (kpc)')
    ax4.set_ylabel('Velocity (km/s)')
    ax4.set_title('Best-fit Examples')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 30)
    
    plt.tight_layout()
    plt.savefig('lnal_prime_harmonics_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nDeeper insight: The prime harmonic series Σ(1/p) represents")
    print("the total consciousness bandwidth. Gas performs LISTEN operations")
    print("that pause recognition on prime channels, reducing gravitational binding.")
    print("\nPlots saved to lnal_prime_harmonics_analysis.png")

if __name__ == '__main__':
    analyze() 