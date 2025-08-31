#!/usr/bin/env python3
"""
Multi-Scale Hierarchical LNAL Model
Includes all intermediate organizational levels from voxel to galaxy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from dataclasses import dataclass
from typing import List, Tuple

# Physical constants
G = 6.67430e-11  # m³/kg/s²
c = 2.99792458e8  # m/s
hbar = 1.054571817e-34  # J⋅s
M_sun = 1.98847e30  # kg
kpc = 3.0856775814913673e19  # m
pc = kpc / 1000

# Recognition Science parameters
phi = (1 + np.sqrt(5)) / 2
tau_0 = 7.33e-15  # s
L_0 = 0.335e-9  # m

# Derived parameters
ell_1 = 0.97 * kpc
g_dagger = 1.2e-10  # m/s²
mu = hbar / (c * ell_1)
proton_mass_energy = 938.3e6 * 1.602e-19  # J
I_star_voxel = proton_mass_energy / L_0**3

@dataclass
class HierarchicalLevel:
    """Represents an organizational level in the hierarchy"""
    name: str
    scale: float  # meters
    clustering_factor: float  # typically 8 for 3D
    coherence_boost: float  # quantum/classical coherence

# Define the complete hierarchy
HIERARCHY_LEVELS = [
    HierarchicalLevel("voxel", L_0, 1.0, 1.0),
    HierarchicalLevel("nucleon", 1e-15, 8.0, 1.2),  # Strong force binding
    HierarchicalLevel("atom", 1e-10, 8.0, 1.5),  # Electron orbitals
    HierarchicalLevel("molecule", 1e-9, 8.0, 1.3),  # Chemical bonds
    HierarchicalLevel("nanocluster", 1e-8, 8.0, 1.1),  # Van der Waals
    HierarchicalLevel("dust_grain", 1e-6, 8.0, 1.0),  # Solid state
    HierarchicalLevel("planetesimal", 1e3, 8.0, 1.0),  # Gravity binding
    HierarchicalLevel("planet", 1e7, 8.0, 1.1),  # Differentiation
    HierarchicalLevel("star", 1e9, 8.0, 1.8),  # Fusion processes
    HierarchicalLevel("stellar_cluster", 1e16, 8.0, 1.2),  # Gravitational
    HierarchicalLevel("galaxy_core", 1e19, 8.0, 1.5),  # Dense region
    HierarchicalLevel("galaxy", 1e22, 8.0, 2.0),  # Full system
]

def compute_multiscale_enhancement(target_scale: float) -> Tuple[float, float, int]:
    """
    Compute enhancement through all hierarchical levels
    
    Returns:
        I_star_eff: Effective information capacity
        lambda_eff: Effective coupling
        n_levels: Number of levels traversed
    """
    enhancement = 1.0
    n_levels = 0
    
    for i in range(1, len(HIERARCHY_LEVELS)):
        if HIERARCHY_LEVELS[i].scale > target_scale:
            break
            
        level = HIERARCHY_LEVELS[i]
        prev_level = HIERARCHY_LEVELS[i-1]
        
        # Scale ratio between levels
        scale_ratio = level.scale / prev_level.scale
        
        # Clustering enhancement: sqrt(N) where N = clustering_factor^3
        clustering_boost = np.sqrt(level.clustering_factor)
        
        # Eight-beat resonance at specific scales
        eight_beat_scale = c * tau_0 * (8**i)  # Harmonics
        resonance = 1 + 0.3 * np.exp(-((level.scale - eight_beat_scale)/eight_beat_scale)**2)
        
        # Recognition efficiency peaks at ell_1
        recognition = 1 + 0.5 * np.exp(-((level.scale - ell_1)/ell_1)**2)
        
        # Total enhancement for this level
        level_enhancement = clustering_boost * level.coherence_boost * resonance * recognition
        enhancement *= level_enhancement
        n_levels += 1
        
    # Effective parameters
    I_star_eff = I_star_voxel * enhancement
    lambda_eff = np.sqrt(g_dagger * c**2 / I_star_eff)
    
    return I_star_eff, lambda_eff, n_levels

def solve_multiscale_field(r, rho, scale_length):
    """Solve field equation with multi-scale enhancement"""
    I_star, lambda_val, n_levels = compute_multiscale_enhancement(scale_length)
    
    print(f"  Levels traversed: {n_levels}")
    print(f"  Total enhancement: {I_star/I_star_voxel:.2e}")
    print(f"  Effective λ: {lambda_val:.3e}")
    
    # Source term
    B = rho * c**2
    mu_squared = (mu * c / hbar)**2
    
    # Initial guess
    I = lambda_val * B / mu_squared
    
    # Iterative solution
    for iteration in range(200):
        I_old = I.copy()
        
        # Gradient and MOND function
        dI_dr = np.gradient(I, r)
        x = np.abs(dI_dr) / I_star
        mu_x = x / np.sqrt(1 + x**2)
        
        # Laplacian in cylindrical coordinates
        term = r * mu_x * dI_dr
        term[0] = term[1]  # Regularity
        d_term_dr = np.gradient(term, r)
        laplacian = d_term_dr / (r + 1e-30)
        
        # Update equation
        source = -lambda_val * B + mu_squared * I
        residual = laplacian - source
        
        omega = 0.4
        I = I - omega * residual * (r[1] - r[0])**2
        I[I < 0] = 0
        
        # Convergence
        change = np.max(np.abs(I - I_old) / (I_star + np.abs(I)))
        if change < 1e-6:
            break
    
    return I, lambda_val, I_star

def model_galaxy_multiscale(name, M_star, R_disk, V_obs, gas_fraction=0.2):
    """Model galaxy with complete multi-scale hierarchy"""
    print(f"\n{'='*60}")
    print(f"Galaxy: {name}")
    print(f"M_star = {M_star/M_sun:.2e} M_sun")
    print(f"R_disk = {R_disk/kpc:.2f} kpc")
    print(f"V_obs = {V_obs/1000:.1f} km/s")
    
    # Radial grid
    r = np.linspace(0.01 * R_disk, 15 * R_disk, 300)
    r_kpc = r / kpc
    
    # Mass distribution
    Sigma_0 = M_star / (2 * np.pi * R_disk**2)
    Sigma = Sigma_0 * np.exp(-r / R_disk)
    Sigma_total = (1 + gas_fraction) * Sigma
    
    # Volume density
    h_disk = 0.3 * kpc
    rho = Sigma_total / (2 * h_disk)
    
    # Solve with multi-scale enhancement
    I, lambda_val, I_star = solve_multiscale_field(r, rho, R_disk)
    
    # Accelerations
    dI_dr = np.gradient(I, r)
    a_info = lambda_val * dI_dr / c**2
    
    # Newtonian
    M_enc = np.zeros_like(r)
    for i in range(1, len(r)):
        M_enc[i] = 2 * np.pi * simpson(r[:i+1] * Sigma_total[:i+1], x=r[:i+1])
    a_newton = G * M_enc / r**2
    
    # Total with quantum coherence
    coherence_factor = 0.25  # 25% quantum-classical interference
    a_total = a_newton + np.abs(a_info) + 2 * coherence_factor * np.sqrt(a_newton * np.abs(a_info))
    
    # Velocities
    V_newton = np.sqrt(a_newton * r)
    V_info = np.sqrt(np.abs(a_info * r))
    V_total = np.sqrt(a_total * r)
    
    # Asymptotic velocity
    idx_flat = r > 3 * R_disk
    V_model = np.mean(V_total[idx_flat])
    
    # Plot
    plt.figure(figsize=(10, 7))
    
    plt.plot(r_kpc, V_total/1000, 'b-', linewidth=2.5,
             label=f'Multi-scale LNAL (V∞={V_model/1000:.0f} km/s)')
    plt.plot(r_kpc, V_newton/1000, 'r--', linewidth=2, label='Newton only')
    plt.plot(r_kpc, V_info/1000, 'g:', linewidth=2, label='Info field only')
    plt.axhline(y=V_obs/1000, color='k', linestyle='-.', alpha=0.7,
                label=f'Observed: {V_obs/1000:.0f} km/s')
    
    plt.axvline(x=R_disk/kpc, color='gray', linestyle=':', alpha=0.5)
    plt.text(R_disk/kpc + 0.1, 20, 'R_disk', fontsize=9)
    
    plt.xlabel('Radius [kpc]', fontsize=12)
    plt.ylabel('Velocity [km/s]', fontsize=12)
    plt.title(f'{name} - Multi-Scale Hierarchy', fontsize=14)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, min(12, 8*R_disk/kpc))
    plt.ylim(0, max(350, 1.2*V_obs/1000))
    
    plt.tight_layout()
    plt.savefig(f'lnal_multiscale_{name}.png', dpi=150)
    plt.close()
    
    ratio = V_model / V_obs
    print(f"V_model/V_obs = {ratio:.3f}")
    
    return V_model, V_obs, ratio

def main():
    """Test multi-scale model"""
    print("="*60)
    print("Multi-Scale Hierarchical LNAL Gravity")
    print("="*60)
    print(f"Base parameters:")
    print(f"  φ = {phi:.6f}")
    print(f"  τ₀ = {tau_0*1e15:.2f} fs")
    print(f"  L₀ = {L_0*1e9:.3f} nm")
    print(f"  ℓ₁ = {ell_1/kpc:.2f} kpc")
    print(f"  g† = {g_dagger:.2e} m/s²")
    
    print(f"\nHierarchical levels ({len(HIERARCHY_LEVELS)}):")
    for level in HIERARCHY_LEVELS:
        print(f"  {level.name:15s}: {level.scale:10.2e} m")
    
    # Test galaxies
    galaxies = [
        {'name': 'NGC2403', 'L36': 10.041e9, 'Rdisk': 1.39*kpc, 'Vobs': 131.2e3},
        {'name': 'NGC3198', 'L36': 38.279e9, 'Rdisk': 3.14*kpc, 'Vobs': 150.1e3},
        {'name': 'NGC6503', 'L36': 12.845e9, 'Rdisk': 2.16*kpc, 'Vobs': 116.3e3},
        {'name': 'DDO154', 'L36': 0.053e9, 'Rdisk': 0.37*kpc, 'Vobs': 47.0e3},
        {'name': 'UGC2885', 'L36': 403.525e9, 'Rdisk': 11.40*kpc, 'Vobs': 289.5e3},
    ]
    
    results = []
    for gal in galaxies:
        M_star = 0.5 * gal['L36'] * M_sun
        V_model, V_obs, ratio = model_galaxy_multiscale(
            gal['name'], M_star, gal['Rdisk'], gal['Vobs']
        )
        results.append({'name': gal['name'], 'ratio': ratio})
    
    # Summary
    print("\n" + "="*60)
    print("MULTI-SCALE RESULTS")
    print("="*60)
    
    ratios = [r['ratio'] for r in results]
    print(f"Mean V_model/V_obs = {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
    
    if np.mean(ratios) > 0.8:
        print("\n✓ Multi-scale hierarchy resolves the factor of 3!")
        print("  All intermediate organizational levels are essential")
    else:
        print(f"\n→ Still missing factor of {1/np.mean(ratios):.1f}")

if __name__ == "__main__":
    main() 