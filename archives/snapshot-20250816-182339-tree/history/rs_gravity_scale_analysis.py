#!/usr/bin/env python3
"""
Scale Analysis of Recognition Science Gravity
Understanding the behavior at different scales and system types
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Physical constants
G_SI = 6.67430e-11  # m^3/kg/s^2
c = 299792458.0     # m/s
pc = 3.0857e16      # meters
kpc = 1000 * pc
M_sun = 1.989e30    # kg

# Recognition Science constants
phi = (1 + np.sqrt(5)) / 2
beta_0 = -(phi - 1) / phi**5
lambda_micro = 7.23e-36  # meters
lambda_eff = 50.8e-6     # meters (optimized)
ell_1 = 0.97 * kpc
ell_2 = 24.3 * kpc

# Optimized scale factors from SPARC analysis
beta_scale = 1.492
mu_scale = 1.644
coupling_scale = 1.326

def Xi_kernel(x):
    """Xi kernel for scale transitions"""
    x = np.atleast_1d(x)
    result = np.zeros_like(x)
    
    # Low-x expansion
    low_mask = x < 0.01
    if np.any(low_mask):
        x_low = x[low_mask]
        result[low_mask] = (3/5)*x_low**2 - (3/7)*x_low**4 + (9/35)*x_low**6
    
    # High-x expansion
    high_mask = x > 100
    if np.any(high_mask):
        x_high = x[high_mask]
        result[high_mask] = (1 - 6/x_high**2 + 30/x_high**4 - 140/x_high**6)
    
    # Middle range - direct calculation
    mid_mask = ~(low_mask | high_mask)
    if np.any(mid_mask):
        x_mid = x[mid_mask]
        result[mid_mask] = 3 * (np.sin(x_mid) - x_mid * np.cos(x_mid)) / x_mid**3
    
    return result

def F_kernel(r):
    """Recognition kernel F(r)"""
    r = np.atleast_1d(r)
    F1 = Xi_kernel(r / ell_1)
    F2 = Xi_kernel(r / ell_2)
    return F1 + F2

def G_of_r(r, use_optimization=True):
    """Scale-dependent Newton constant"""
    G_inf = G_SI
    beta = beta_scale * beta_0 if use_optimization else beta_0
    
    # Power law component
    power_factor = (lambda_eff / r) ** beta
    
    # Recognition kernel
    F = F_kernel(r)
    
    return G_inf * power_factor * F

def analyze_scale_behavior():
    """Analyze G(r) behavior at different scales"""
    
    # Create scale array from nano to cosmic
    r = np.logspace(np.log10(1e-9), np.log10(100*kpc), 1000)
    
    # Calculate G(r) with and without optimization
    G_opt = G_of_r(r, use_optimization=True)
    G_base = G_of_r(r, use_optimization=False)
    
    # Find characteristic scales
    r_nano = 20e-9  # 20 nm
    r_micro = lambda_eff
    r_dwarf = 0.25 * kpc  # Typical dwarf spheroidal scale
    r_galaxy = 10 * kpc   # Typical galaxy scale
    
    # Calculate G at these scales
    G_nano_opt = float(G_of_r(np.array([r_nano]), True)[0])
    G_micro_opt = float(G_of_r(np.array([r_micro]), True)[0])
    G_dwarf_opt = float(G_of_r(np.array([r_dwarf]), True)[0])
    G_galaxy_opt = float(G_of_r(np.array([r_galaxy]), True)[0])
    
    G_nano_base = float(G_of_r(np.array([r_nano]), False)[0])
    G_micro_base = float(G_of_r(np.array([r_micro]), False)[0])
    G_dwarf_base = float(G_of_r(np.array([r_dwarf]), False)[0])
    G_galaxy_base = float(G_of_r(np.array([r_galaxy]), False)[0])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. G(r) profile
    ax = axes[0, 0]
    ax.loglog(r/kpc, G_opt/G_SI, 'b-', linewidth=2, label='Optimized (SPARC)')
    ax.loglog(r/kpc, G_base/G_SI, 'r--', linewidth=2, label='Base theory')
    
    # Mark characteristic scales
    ax.axvline(r_nano/kpc, color='gray', linestyle=':', alpha=0.5, label='20 nm')
    ax.axvline(r_micro/kpc, color='green', linestyle=':', alpha=0.5, label='λ_eff')
    ax.axvline(r_dwarf/kpc, color='orange', linestyle=':', alpha=0.5, label='Dwarf scale')
    ax.axvline(r_galaxy/kpc, color='purple', linestyle=':', alpha=0.5, label='Galaxy scale')
    ax.axvline(ell_1/kpc, color='cyan', linestyle=':', alpha=0.5, label='ℓ₁')
    ax.axvline(ell_2/kpc, color='magenta', linestyle=':', alpha=0.5, label='ℓ₂')
    
    ax.axhline(1, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('G(r)/G₀')
    ax.set_title('Scale-dependent Newton Constant')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.1, 1e6])
    
    # 2. Enhancement ratio
    ax = axes[0, 1]
    enhancement = G_opt / G_base
    ax.semilogx(r/kpc, enhancement, 'g-', linewidth=2)
    ax.axhline(beta_scale, color='red', linestyle='--', label=f'β_scale = {beta_scale:.3f}')
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('G_optimized / G_base')
    ax.set_title('Optimization Enhancement Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. System comparison
    ax = axes[1, 0]
    
    # Calculate effective G enhancement for different systems
    systems = ['Nanoscale\n(20 nm)', 'Microscale\n(λ_eff)', 'Dwarf Sph.\n(0.25 kpc)', 'Galaxy\n(10 kpc)']
    G_opt_values = [G_nano_opt/G_SI, G_micro_opt/G_SI, G_dwarf_opt/G_SI, G_galaxy_opt/G_SI]
    G_base_values = [G_nano_base/G_SI, G_micro_base/G_SI, G_dwarf_base/G_SI, G_galaxy_base/G_SI]
    
    x = np.arange(len(systems))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, G_opt_values, width, label='Optimized', alpha=0.7)
    bars2 = ax.bar(x + width/2, G_base_values, width, label='Base theory', alpha=0.7)
    
    ax.set_yscale('log')
    ax.set_xlabel('System Type')
    ax.set_ylabel('G/G₀')
    ax.set_title('G Enhancement by System')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (opt, base) in enumerate(zip(G_opt_values, G_base_values)):
        ax.text(i - width/2, opt*1.5, f'{opt:.1e}', ha='center', va='bottom', fontsize=8, rotation=45)
        ax.text(i + width/2, base*1.5, f'{base:.1e}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    # 4. Analysis text
    ax = axes[1, 1]
    ax.axis('off')
    
    text = f"""Scale Analysis Results:

Nanoscale (20 nm):
  Base: G/G₀ = {G_nano_base/G_SI:.2f}
  Optimized: G/G₀ = {G_nano_opt/G_SI:.2f}
  
Microscale (λ_eff = {lambda_eff*1e6:.1f} μm):
  Base: G/G₀ = {G_micro_base/G_SI:.1e}
  Optimized: G/G₀ = {G_micro_opt/G_SI:.1e}
  
Dwarf Spheroidal (0.25 kpc):
  Base: G/G₀ = {G_dwarf_base/G_SI:.1e}
  Optimized: G/G₀ = {G_dwarf_opt/G_SI:.1e}
  Enhancement: {G_dwarf_opt/G_dwarf_base:.1f}×
  
Galaxy (10 kpc):
  Base: G/G₀ = {G_galaxy_base/G_SI:.2f}
  Optimized: G/G₀ = {G_galaxy_opt/G_SI:.2f}
  Enhancement: {G_galaxy_opt/G_galaxy_base:.1f}×

Key Insight:
Dwarf spheroidals experience ~{G_dwarf_opt/G_SI:.0e}× stronger
gravity than Newton, explaining the high σ_v predictions.

The optimization (β_scale = {beta_scale:.3f}) enhances this
by an additional factor of ~{G_dwarf_opt/G_dwarf_base:.1f}.
"""
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('rs_gravity_scale_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'G_nano_opt': G_nano_opt,
        'G_dwarf_opt': G_dwarf_opt,
        'G_galaxy_opt': G_galaxy_opt,
        'dwarf_enhancement': G_dwarf_opt / G_SI,
        'optimization_factor': G_dwarf_opt / G_dwarf_base
    }

def theoretical_implications():
    """Explore theoretical implications of dwarf spheroidal results"""
    
    # The key finding: RS gravity is ~700× too strong for dwarfs
    dwarf_overprediction = 16.82  # Mean σ_pred/σ_obs
    required_G_reduction = dwarf_overprediction**2  # Since σ ∝ √G
    
    print("\n=== Theoretical Implications ===\n")
    print(f"1. Dwarf Spheroidal Challenge:")
    print(f"   - RS predicts σ_v ~{dwarf_overprediction:.1f}× too high")
    print(f"   - Implies G is ~{required_G_reduction:.0f}× too strong")
    print(f"   - At r ~ 0.25 kpc, need G ~ G₀/{required_G_reduction:.0f}\n")
    
    print(f"2. Possible Resolutions:")
    print(f"   a) Modified F(r) kernel at sub-kpc scales")
    print(f"   b) Information field screening in low-density systems")
    print(f"   c) Different physics for pressure vs rotation support")
    print(f"   d) Inseparability corrections at dwarf scales\n")
    
    print(f"3. Physical Interpretation:")
    print(f"   - Disk galaxies: rotation → strong information field")
    print(f"   - Dwarf spheroidals: pressure → weak information field")
    print(f"   - Suggests velocity gradient ∇v drives enhancement\n")
    
    print(f"4. Predictions to Test:")
    print(f"   - Globular clusters should show similar overprediction")
    print(f"   - Elliptical galaxies might be intermediate")
    print(f"   - Ultra-diffuse galaxies are key test cases")
    
    # Create a plot showing the required modification
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Current G(r) vs required for dwarfs
    r = np.logspace(np.log10(0.01*kpc), np.log10(100*kpc), 1000)
    G_current = G_of_r(r, use_optimization=True)
    G_required_dwarf = G_SI * np.ones_like(r) / required_G_reduction
    
    ax1.loglog(r/kpc, G_current/G_SI, 'b-', linewidth=2, label='Current RS')
    ax1.loglog(r/kpc, G_required_dwarf/G_SI, 'r--', linewidth=2, 
               label=f'Required for dwarfs (G₀/{required_G_reduction:.0f})')
    ax1.axvline(0.25, color='orange', linestyle=':', alpha=0.5, label='Dwarf scale')
    ax1.axvline(10, color='purple', linestyle=':', alpha=0.5, label='Galaxy scale')
    ax1.axhline(1, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Radius (kpc)')
    ax1.set_ylabel('G(r)/G₀')
    ax1.set_title('Required G(r) Modification')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([1e-4, 1e6])
    
    # Proposed screening function
    ax2.semilogx(r/kpc, np.ones_like(r), 'b-', linewidth=2, label='Disk galaxies')
    
    # Smooth transition for dwarfs
    r_screen = 1.0 * kpc  # Screening scale
    screening = 1 / (1 + (r_screen/r)**2) / required_G_reduction + (1 - 1/(1 + (r_screen/r)**2))
    ax2.semilogx(r/kpc, screening, 'r--', linewidth=2, label='Dwarf spheroidals')
    
    ax2.set_xlabel('Radius (kpc)')
    ax2.set_ylabel('G_effective / G_RS')
    ax2.set_title('Proposed System-Dependent Screening')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.2])
    
    plt.tight_layout()
    plt.savefig('rs_gravity_dwarf_modification.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run the scale analysis"""
    print("=== Recognition Science Gravity Scale Analysis ===\n")
    
    # Analyze scale behavior
    results = analyze_scale_behavior()
    
    print(f"Key Results:")
    print(f"- Nanoscale (20 nm): G/G₀ = {results['G_nano_opt']/G_SI:.2f}")
    print(f"- Dwarf scale (0.25 kpc): G/G₀ = {results['dwarf_enhancement']:.1e}")
    print(f"- Galaxy scale (10 kpc): G/G₀ = {results['G_galaxy_opt']/G_SI:.2f}")
    print(f"- Optimization enhances dwarf G by {results['optimization_factor']:.1f}×")
    
    # Explore theoretical implications
    theoretical_implications()
    
    print("\nAnalysis complete. Plots saved as:")
    print("- rs_gravity_scale_analysis.png")
    print("- rs_gravity_dwarf_modification.png")

if __name__ == "__main__":
    main() 