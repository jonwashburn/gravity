#!/usr/bin/env python3
"""
Full SPARC Analysis with Multi-Scale LNAL Model
Analyzes all 175 SPARC galaxies with complete hierarchical theory
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pickle

# Physical constants
G = 6.67430e-11  # m³/kg/s²
c = 2.99792458e8  # m/s
hbar = 1.054571817e-34  # J⋅s
M_sun = 1.98847e30  # kg
kpc = 3.0856775814913673e19  # m

# Recognition Science parameters
phi = (1 + np.sqrt(5)) / 2
tau_0 = 7.33e-15  # s
L_0 = 0.335e-9  # m
ell_1 = 0.97 * kpc
g_dagger = 1.2e-10  # m/s²

# Base parameters
mu = hbar / (c * ell_1)
proton_mass_energy = 938.3e6 * 1.602e-19  # J
I_star_voxel = proton_mass_energy / L_0**3

@dataclass
class HierarchicalLevel:
    name: str
    scale: float
    clustering: float
    coherence: float

# Complete hierarchy with fine-tuned parameters
HIERARCHY = [
    HierarchicalLevel("voxel", L_0, 1.0, 1.0),
    HierarchicalLevel("nucleon", 1e-15, 8.0, 1.5),
    HierarchicalLevel("atom", 1e-10, 8.0, 2.0),
    HierarchicalLevel("molecule", 1e-9, 8.0, 1.8),
    HierarchicalLevel("nanocluster", 1e-8, 8.0, 1.5),
    HierarchicalLevel("dust", 1e-6, 8.0, 1.3),
    HierarchicalLevel("rock", 1e-1, 8.0, 1.2),
    HierarchicalLevel("planetesimal", 1e3, 8.0, 1.1),
    HierarchicalLevel("planet", 1e7, 8.0, 1.3),
    HierarchicalLevel("star", 1e9, 8.0, 2.5),
    HierarchicalLevel("cluster", 1e16, 8.0, 1.8),
    HierarchicalLevel("core", 1e19, 8.0, 2.2),
    HierarchicalLevel("galaxy", 1e22, 8.0, 3.0),
]

def parse_sparc_data():
    """Parse SPARC data file"""
    with open("SPARC_Lelli2016c.mrt", 'r') as f:
        lines = f.readlines()
    
    # Find data start
    data_start = 0
    for i, line in enumerate(lines):
        if '-------' in line and i > 80:
            data_start = i + 1
            break
    
    galaxies = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or 'Note:' in line:
            continue
            
        parts = line.split()
        if len(parts) < 18:
            continue
            
        try:
            galaxy = {
                'name': parts[0],
                'D': float(parts[2]),  # Mpc
                'L36': float(parts[7]),  # 10^9 L_sun
                'Rdisk': float(parts[11]),  # kpc
                'Vflat': float(parts[15]),  # km/s
                'Q': int(parts[17])
            }
            
            # Keep galaxies with good data and Rdisk > 0
            if (galaxy['Vflat'] > 0 and galaxy['L36'] > 0 and 
                galaxy['Rdisk'] > 0 and galaxy['Q'] >= 1):
                galaxies.append(galaxy)
                
        except (ValueError, IndexError):
            continue
    
    return galaxies

def compute_full_enhancement(scale_length: float) -> Tuple[float, float]:
    """
    Compute total enhancement through all hierarchical levels
    
    Key insight: Each level contributes sqrt(clustering) × coherence
    """
    enhancement = 1.0
    
    for i in range(1, len(HIERARCHY)):
        if HIERARCHY[i].scale > scale_length:
            break
            
        level = HIERARCHY[i]
        
        # Base clustering enhancement
        cluster_boost = np.sqrt(level.clustering)
        
        # Eight-beat resonance
        eight_beat_scale = c * tau_0 * phi**i
        resonance = 1 + 0.4 * np.exp(-((level.scale - eight_beat_scale)/eight_beat_scale)**2)
        
        # Recognition scale boost
        recognition = 1 + 0.6 * np.exp(-((level.scale - ell_1)/ell_1)**2)
        
        # Total for this level
        level_boost = cluster_boost * level.coherence * resonance * recognition
        enhancement *= level_boost
    
    # Effective parameters
    I_star_eff = I_star_voxel * enhancement
    lambda_eff = np.sqrt(g_dagger * c**2 / I_star_eff)
    
    return I_star_eff, lambda_eff

def solve_field_equation(r, rho, scale_length):
    """Solve non-linear field equation"""
    I_star, lambda_val = compute_full_enhancement(scale_length)
    
    B = rho * c**2
    mu_squared = (mu * c / hbar)**2
    
    # Initial guess
    I = lambda_val * B / mu_squared
    
    # Iterative solution
    for _ in range(150):
        I_old = I.copy()
        
        # Gradient and MOND function
        dI_dr = np.gradient(I, r)
        x = np.abs(dI_dr) / I_star
        mu_x = x / np.sqrt(1 + x**2)
        
        # Laplacian
        term = r * mu_x * dI_dr
        term[0] = term[1]
        d_term_dr = np.gradient(term, r)
        laplacian = d_term_dr / (r + 1e-30)
        
        # Update
        source = -lambda_val * B + mu_squared * I
        residual = laplacian - source
        
        I = I - 0.4 * residual * (r[1] - r[0])**2
        I[I < 0] = 0
        
        if np.max(np.abs(I - I_old) / (I_star + np.abs(I))) < 1e-6:
            break
    
    return I, lambda_val

def analyze_galaxy(galaxy: Dict, plot=False) -> Dict:
    """Analyze single galaxy with LNAL model"""
    name = galaxy['name']
    M_star = 0.5 * galaxy['L36'] * 1e9 * M_sun  # M/L = 0.5
    R_disk = galaxy['Rdisk'] * kpc
    V_obs = galaxy['Vflat'] * 1000  # m/s
    
    # Radial grid
    r = np.linspace(0.01 * R_disk, 10 * R_disk, 200)
    
    # Mass distribution (exponential disk)
    Sigma_0 = M_star / (2 * np.pi * R_disk**2)
    Sigma = Sigma_0 * np.exp(-r / R_disk)
    Sigma_total = 1.2 * Sigma  # 20% gas
    
    # Volume density
    h_disk = 0.3 * kpc
    rho = Sigma_total / (2 * h_disk)
    
    # Solve field equation
    I, lambda_val = solve_field_equation(r, rho, R_disk)
    
    # Accelerations
    dI_dr = np.gradient(I, r)
    a_info = lambda_val * dI_dr / c**2
    
    # Newtonian
    M_enc = np.zeros_like(r)
    for i in range(1, len(r)):
        M_enc[i] = 2 * np.pi * simpson(r[:i+1] * Sigma_total[:i+1], x=r[:i+1])
    a_newton = G * M_enc / r**2
    
    # Total with coherence
    coherence = 0.3  # 30% quantum-classical interference
    a_total = a_newton + np.abs(a_info) + 2 * coherence * np.sqrt(a_newton * np.abs(a_info))
    
    # Velocities
    V_total = np.sqrt(a_total * r)
    
    # Asymptotic velocity
    idx_flat = r > 3 * R_disk
    if np.sum(idx_flat) > 5:
        V_model = np.mean(V_total[idx_flat])
    else:
        V_model = V_total[-1]
    
    ratio = V_model / V_obs
    
    if plot:
        plt.figure(figsize=(8, 6))
        r_kpc = r / kpc
        plt.plot(r_kpc, V_total/1000, 'b-', linewidth=2, 
                 label=f'LNAL (V∞={V_model/1000:.0f} km/s)')
        plt.axhline(y=V_obs/1000, color='k', linestyle='--', 
                    label=f'Observed: {V_obs/1000:.0f} km/s')
        plt.xlabel('Radius [kpc]')
        plt.ylabel('Velocity [km/s]')
        plt.title(f'{name} - Ratio = {ratio:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, min(10, 8*galaxy['Rdisk']))
        plt.tight_layout()
        plt.savefig(f'lnal_{name}.png', dpi=120)
        plt.close()
    
    return {
        'name': name,
        'V_obs': V_obs/1000,
        'V_model': V_model/1000,
        'ratio': ratio,
        'quality': galaxy['Q']
    }

def main():
    """Analyze all SPARC galaxies"""
    print("="*60)
    print("Full SPARC Analysis with Multi-Scale LNAL")
    print("="*60)
    
    # Parse data
    print("Parsing SPARC data...")
    galaxies = parse_sparc_data()
    print(f"Found {len(galaxies)} galaxies with good data")
    
    # Analyze all galaxies
    print("\nAnalyzing galaxies...")
    results = []
    
    # Test galaxies to plot
    test_names = ['NGC2403', 'NGC3198', 'NGC6503', 'DDO154', 'UGC02885']
    
    for i, galaxy in enumerate(galaxies):
        plot = galaxy['name'] in test_names
        result = analyze_galaxy(galaxy, plot=plot)
        results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(galaxies)} galaxies...")
    
    # Save results
    with open('lnal_sparc_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Analysis by quality
    print("\n" + "="*60)
    print("RESULTS BY QUALITY")
    print("="*60)
    
    for q in [3, 2, 1]:
        q_results = [r for r in results if r['quality'] == q]
        if q_results:
            ratios = [r['ratio'] for r in q_results]
            print(f"\nQuality {q} ({len(q_results)} galaxies):")
            print(f"  Mean ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
            print(f"  Median: {np.median(ratios):.3f}")
            print(f"  Range: [{np.min(ratios):.3f}, {np.max(ratios):.3f}]")
    
    # Overall statistics
    all_ratios = [r['ratio'] for r in results]
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Total galaxies: {len(results)}")
    print(f"Mean V_model/V_obs: {np.mean(all_ratios):.3f} ± {np.std(all_ratios):.3f}")
    print(f"Median: {np.median(all_ratios):.3f}")
    
    # Success rate
    good_fits = sum(1 for r in all_ratios if 0.8 < r < 1.2)
    print(f"\nGood fits (0.8-1.2): {good_fits}/{len(results)} = {100*good_fits/len(results):.1f}%")
    
    # Show outliers
    print("\nOutliers (ratio < 0.5 or > 2.0):")
    outliers = [r for r in results if r['ratio'] < 0.5 or r['ratio'] > 2.0]
    for out in outliers[:5]:
        print(f"  {out['name']:12s}: ratio = {out['ratio']:.3f}")
    
    if len(outliers) > 5:
        print(f"  ... and {len(outliers)-5} more")
    
    # Create summary plot
    plt.figure(figsize=(10, 6))
    plt.hist(all_ratios, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=1.0, color='red', linestyle='--', label='Perfect agreement')
    plt.axvline(x=np.mean(all_ratios), color='blue', linestyle='-', 
                label=f'Mean = {np.mean(all_ratios):.3f}')
    plt.xlabel('V_model / V_obs')
    plt.ylabel('Number of galaxies')
    plt.title('LNAL Model Performance on SPARC Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lnal_sparc_histogram.png', dpi=150)
    plt.close()
    
    print("\nPlots saved for test galaxies and histogram")

if __name__ == "__main__":
    main() 