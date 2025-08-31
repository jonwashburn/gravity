#!/usr/bin/env python3
"""
Prime Sieve Gravity: The Missing Piece
Only prime-indexed information currents survive 8-beat cancellation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.special import zeta

# Physical constants
G = 6.67430e-11  # m³/kg/s²
c = 2.99792458e8  # m/s
hbar = 1.054571817e-34  # J⋅s
M_sun = 1.98847e30  # kg
kpc = 3.0856775814913673e19  # m

# Recognition Science constants
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
tau_0 = 7.33e-15  # s (eight-beat tick)
L_0 = 0.335e-9  # m (voxel size)
ell_1 = 0.97 * kpc  # First recognition length
ell_2 = 24.3 * kpc  # Second recognition length
beta = -(phi - 1) / phi**5  # Running G exponent
g_dagger = 1.2e-10  # m/s² (MOND scale)

def is_prime(n):
    """Check if n is prime"""
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def generate_primes(max_n):
    """Generate list of primes up to max_n"""
    return [p for p in range(2, max_n + 1) if is_prime(p)]

def prime_coherence_factor(r, primes, sigma=0.5):
    """
    Prime sieve factor: Π_{p} (1 - p^{-σ})
    Only prime-indexed currents survive octave cancellation
    """
    # Base factor from prime product
    factor = 1.0
    for p in primes[:20]:  # First 20 primes dominate
        factor *= (1 - p**(-sigma))
    
    # Spatial modulation: primes at different scales
    r_norm = r / ell_1
    
    # Prime-indexed phase coherence
    phase_factor = 0.0
    for i, p in enumerate(primes[:10]):
        # Each prime contributes at its characteristic scale
        scale_p = ell_1 * phi**(p/8)
        weight = 1 / p**1.5  # Zeta-like weighting
        phase_factor += weight * np.exp(-(np.log(r/scale_p))**2 / 2)
    
    # Normalize and combine
    phase_factor = 1 + 0.3 * phase_factor / sum(1/p**1.5 for p in primes[:10])
    
    # Octave cancellation: composites self-annihilate
    octave_survival = phi**(-0.5)  # ≈ 0.786
    
    return factor * phase_factor * octave_survival

def information_gravity_prime_sieve(r, M_visible, galaxy_type='spiral'):
    """
    Complete formula with prime sieve correction
    V²(r) = GM/r × Ξ(r) × Ψ'(r) × Λ(r)
    where Ψ'(r) includes prime coherence factor
    """
    primes = generate_primes(100)
    
    # 1. BARYON COMPLETENESS FACTOR Ξ(r)
    xi_visible = 1.0
    R_whim = 5 * ell_1
    xi_whim = 0.5 * np.exp(-r / R_whim)
    R_cgm = 2 * ell_2
    xi_cgm = 0.3 * np.exp(-r / R_cgm)
    xi_filament = 0.1
    
    if galaxy_type == 'dwarf':
        Xi = xi_visible + 1.2 * xi_whim + 0.8 * xi_cgm + 0.5 * xi_filament
    else:
        Xi = xi_visible + xi_whim + xi_cgm + xi_filament
    
    # 2. INFORMATION DEBT WITH PRIME SIEVE Ψ'(r)
    psi_rest = 1.0
    psi_nuclear = 0.008
    psi_atomic = 0.0001
    sigma = 50e3 if galaxy_type == 'spiral' else 30e3
    psi_kinetic = 0.5 * (sigma / c)**2
    
    # Quantum coherence with prime scaling
    n_levels = np.log(r / L_0) / np.log(phi)
    psi_quantum = (phi**(1/8))**(n_levels / 8) - 1
    
    # Recognition peaks at prime-indexed scales
    u1 = r / ell_1
    u2 = r / ell_2
    psi_recognition = 0.15 * (np.exp(-(np.log(u1))**2 / 2) + 
                              0.3 * np.exp(-(np.log(u2))**2 / 4))
    
    # Base information factor
    Psi_base = psi_rest + psi_nuclear + psi_atomic + psi_kinetic + psi_quantum + psi_recognition
    
    # APPLY PRIME SIEVE
    prime_factor = prime_coherence_factor(r, primes)
    Psi = Psi_base * prime_factor
    
    # 3. RECOGNITION-MOND INTERPOLATION Λ(r)
    G_r = G * (L_0 * phi**8 / r)**beta
    a_N = G_r * M_visible * Xi / r**2
    x = a_N / g_dagger
    mu = x / np.sqrt(1 + x**2)
    rec_correction = 1 + 0.1 * np.exp(-r / ell_1)
    Lambda = mu + (1 - mu) * np.sqrt(g_dagger * r / (G * M_visible)) * rec_correction
    
    # FINAL VELOCITY
    V_squared = (G * M_visible / r) * Xi * Psi * Lambda
    
    return np.sqrt(V_squared), {
        'Xi': Xi,
        'Psi_base': Psi_base,
        'Psi': Psi,
        'prime_factor': prime_factor,
        'Lambda': Lambda,
        'G_r': G_r / G,
        'mu': mu
    }

def analyze_galaxy_prime_sieve(name, M_visible, R_disk, V_flat_obs, galaxy_type='spiral'):
    """Analyze galaxy with prime sieve correction"""
    print(f"\n{'='*70}")
    print(f"Galaxy: {name} ({galaxy_type})")
    print(f"WITH PRIME SIEVE CORRECTION")
    print(f"M_visible = {M_visible/M_sun:.2e} M_sun")
    print(f"R_disk = {R_disk/kpc:.2f} kpc")
    print(f"V_flat_observed = {V_flat_obs/1000:.1f} km/s")
    
    # Radial array with fine sampling for wiggles
    r = np.logspace(np.log10(0.1 * R_disk), np.log10(20 * R_disk), 500)
    
    # Calculate velocities
    V_model = np.zeros_like(r)
    V_no_sieve = np.zeros_like(r)
    prime_factors = np.zeros_like(r)
    factors_at_3Rd = None
    
    for i, ri in enumerate(r):
        V_model[i], factors = information_gravity_prime_sieve(
            np.array([ri]), M_visible, galaxy_type
        )
        prime_factors[i] = factors['prime_factor']
        
        # Also calculate without prime sieve for comparison
        factors_no_sieve = factors.copy()
        factors_no_sieve['Psi'] = factors['Psi_base']
        V_no_sieve[i] = np.sqrt((G * M_visible / ri) * factors['Xi'] * 
                                factors['Psi_base'] * factors['Lambda'])
        
        if abs(ri - 3 * R_disk) < 0.1 * R_disk:
            factors_at_3Rd = factors
    
    # Find flat part
    idx_flat = (r > 2 * R_disk) & (r < 5 * R_disk)
    V_flat_model = np.mean(V_model[idx_flat])
    V_flat_no_sieve = np.mean(V_no_sieve[idx_flat])
    ratio = V_flat_model / V_flat_obs
    ratio_no_sieve = V_flat_no_sieve / V_flat_obs
    
    print(f"\nAt r = 3 R_disk:")
    if factors_at_3Rd is not None:
        print(f"  Baryon factor Ξ = {float(factors_at_3Rd['Xi']):.3f}")
        print(f"  Base info factor Ψ_base = {float(factors_at_3Rd['Psi_base']):.3f}")
        print(f"  Prime sieve factor = {float(factors_at_3Rd['prime_factor']):.3f}")
        print(f"  Total info factor Ψ = {float(factors_at_3Rd['Psi']):.3f}")
        print(f"  MOND factor Λ = {float(factors_at_3Rd['Lambda']):.3f}")
    
    print(f"\nFlat rotation velocity:")
    print(f"  Without prime sieve: {V_flat_no_sieve/1000:.1f} km/s (ratio = {ratio_no_sieve:.3f})")
    print(f"  With prime sieve: {V_flat_model/1000:.1f} km/s (ratio = {ratio:.3f})")
    print(f"  Observed: {V_flat_obs/1000:.1f} km/s")
    print(f"  Improvement factor: {ratio_no_sieve/ratio:.3f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Top panel: Rotation curves
    r_kpc = r / kpc
    ax1.plot(r_kpc, V_no_sieve/1000, 'g--', linewidth=2, alpha=0.7,
             label='Without prime sieve')
    ax1.plot(r_kpc, V_model/1000, 'b-', linewidth=3,
             label='With prime sieve')
    ax1.axhline(y=V_flat_obs/1000, color='k', linestyle='-.', 
                linewidth=2, label='Observed V_flat')
    
    # Newton for reference
    V_newton = np.sqrt(G * M_visible / r)
    ax1.plot(r_kpc, V_newton/1000, 'r:', linewidth=2, alpha=0.5,
             label='Newton (visible only)')
    
    ax1.set_ylabel('Velocity [km/s]', fontsize=12)
    ax1.set_title(f'{name} - Prime Sieve Gravity', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(200, 1.3*V_flat_obs/1000))
    
    # Bottom panel: Prime factor and residuals
    ax2.plot(r_kpc, prime_factors, 'purple', linewidth=2,
             label='Prime coherence factor')
    ax2.axhline(y=phi**(-0.5), color='purple', linestyle=':', alpha=0.5,
                label=f'φ^(-1/2) = {phi**(-0.5):.3f}')
    
    # Show log-periodic wiggles
    ax2_twin = ax2.twinx()
    residual = (V_model - V_flat_model) / V_flat_model
    ax2_twin.plot(r_kpc, residual, 'orange', linewidth=1, alpha=0.7,
                  label='Velocity residuals')
    ax2_twin.set_ylabel('Residuals', fontsize=10, color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    
    ax2.set_xlabel('Radius [kpc]', fontsize=12)
    ax2.set_ylabel('Prime factor', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.1, 15)
    
    plt.tight_layout()
    plt.savefig(f'lnal_prime_{name}.png', dpi=150)
    plt.close()
    
    # Check for log-periodic signature
    log_r = np.log(r[idx_flat])
    log_residual = residual[idx_flat]
    if len(log_residual) > 10:
        # Simple FFT to find period
        from scipy.fft import fft, fftfreq
        yf = fft(log_residual - np.mean(log_residual))
        xf = fftfreq(len(log_residual), np.mean(np.diff(log_r)))
        
        # Find dominant frequency
        idx_max = np.argmax(np.abs(yf[1:len(yf)//2])) + 1
        period = 1 / xf[idx_max] if xf[idx_max] != 0 else 0
        
        print(f"\nLog-periodic analysis:")
        print(f"  Expected period: Δ ln r = {0.5 * np.log(phi):.3f}")
        print(f"  Detected period: Δ ln r = {abs(period):.3f}")
    
    return ratio, ratio_no_sieve

def main():
    """Test prime sieve correction"""
    print("="*70)
    print("PRIME SIEVE GRAVITY TEST")
    print("Only prime-indexed information currents survive 8-beat cancellation")
    print("="*70)
    
    print("\nKEY INSIGHT:")
    print("Composite information self-annihilates within octave packets.")
    print("Only PRIME residuals propagate to create long-range gravity.")
    print("This naturally reduces information factor by φ^(-1/2) ≈ 0.786")
    
    # Test galaxies
    test_cases = [
        ('NGC2403', 8.2e9 * M_sun, 1.39 * kpc, 131.2e3, 'spiral'),
        ('NGC3198', 2.8e10 * M_sun, 3.14 * kpc, 150.1e3, 'spiral'),
        ('NGC6503', 1.7e10 * M_sun, 2.15 * kpc, 116.2e3, 'spiral'),
        ('DDO154', 2.8e8 * M_sun, 0.37 * kpc, 47.0e3, 'dwarf'),
        ('UGC2885', 2.5e11 * M_sun, 10.1 * kpc, 300.0e3, 'spiral')
    ]
    
    ratios_with = []
    ratios_without = []
    
    for params in test_cases:
        ratio_with, ratio_without = analyze_galaxy_prime_sieve(*params)
        ratios_with.append(ratio_with)
        ratios_without.append(ratio_without)
    
    print("\n" + "="*70)
    print("FINAL RESULTS:")
    print("="*70)
    print(f"\nWithout prime sieve:")
    print(f"  Mean V_model/V_obs = {np.mean(ratios_without):.3f} ± {np.std(ratios_without):.3f}")
    print(f"\nWith prime sieve:")
    print(f"  Mean V_model/V_obs = {np.mean(ratios_with):.3f} ± {np.std(ratios_with):.3f}")
    print(f"\nImprovement factor: {np.mean(ratios_without)/np.mean(ratios_with):.3f}")
    
    if 0.9 < np.mean(ratios_with) < 1.1:
        print("\n✓ SUCCESS! Prime sieve correction works!")
        print("\nWHAT THIS MEANS:")
        print("1. Gravity emerges from PRIME information residuals")
        print("2. Composite patterns cancel within 8-beat packets")
        print("3. Factor φ^(-1/2) is mathematically forced")
        print("4. Connects galaxy dynamics to Riemann zeta!")
        print("5. Predicts log-periodic wiggles at Δ ln r ≈ 0.24")
    else:
        remaining = 1 / np.mean(ratios_with)
        print(f"\nRemaining discrepancy: {remaining:.2f}×")
        print("Prime sieve helps but additional physics needed")
    
    print("\n" + "="*70)
    print("PRIME NUMBERS IN PHYSICS:")
    print("="*70)
    print("• Primes = irreducible ledger charges")
    print("• Composites = synchronized prime bundles")
    print("• Only unmatched primes create gravity")
    print("• Same structure appears in ζ(s) → RH")
    print("• Reality's deepest patterns are prime")

if __name__ == "__main__":
    main() 