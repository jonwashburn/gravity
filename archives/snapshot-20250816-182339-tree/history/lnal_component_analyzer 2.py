#!/usr/bin/env python3
"""
LNAL Component Analyzer
=======================
Use SPARC's component decomposition (V_gas, V_disk, V_bulge)
to get better surface density estimates.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline
import json
import os

# Constants
G = 6.67430e-11  # m³/kg/s²
G_DAGGER = 1.2e-10  # m/s² (MOND scale)
kpc = 3.0856775814913673e19  # m
pc = kpc / 1000
M_sun = 1.98847e30  # kg


def velocity_to_surface_density(r, v_component, component_type='disk'):
    """
    Convert component velocity to surface density.
    
    For a thin disk: V² = G * M_enc / r
    where M_enc = 2π ∫ Σ(r') r' dr'
    
    This gives: Σ(r) = (1/2πGr) * d(r V²)/dr
    """
    # Handle zeros
    v_component = np.maximum(v_component, 0)
    r = np.maximum(r, r[0] * 0.1)
    
    # Compute r * V²
    rv2 = r * v_component**2
    
    # Smooth before differentiating
    if len(r) > 5:
        valid = rv2 > 0
        if np.sum(valid) > 5:
            spline = UnivariateSpline(r[valid], rv2[valid], s=0.1, k=3)
            rv2_smooth = spline(r)
        else:
            rv2_smooth = rv2
    else:
        rv2_smooth = rv2
    
    # Compute derivative
    drv2_dr = np.gradient(rv2_smooth, r)
    
    # Surface density
    Sigma = drv2_dr / (2 * np.pi * G * r)
    
    # Ensure positive
    Sigma = np.maximum(Sigma, 0)
    
    # Apply component-specific corrections
    if component_type == 'gas':
        # Gas includes helium
        Sigma *= 1.33
    elif component_type == 'bulge':
        # Bulge is 3D, approximate correction
        Sigma *= 0.6  # Rough spherical-to-disk conversion
    
    return Sigma


def load_sparc_components(galaxy_name, rotmod_dir='Rotmod_LTG'):
    """Load SPARC rotation curve with components."""
    filepath = os.path.join(rotmod_dir, f'{galaxy_name}_rotmod.dat')
    
    if not os.path.exists(filepath):
        return None
    
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 8:
                data.append([float(p) for p in parts[:8]])
    
    if not data:
        return None
    
    data = np.array(data)
    return {
        'r': data[:, 0] * kpc,  # kpc to m
        'v_obs': data[:, 1] * 1000,  # km/s to m/s
        'v_err': data[:, 2] * 1000,
        'v_gas': data[:, 3] * 1000,
        'v_disk': data[:, 4] * 1000,
        'v_bulge': data[:, 5] * 1000,
    }


def analyze_components(galaxy_name):
    """Analyze galaxy using component decomposition."""
    # Load data
    data = load_sparc_components(galaxy_name)
    if data is None:
        print(f"Could not load {galaxy_name}")
        return None
    
    r = data['r']
    
    # Convert each component to surface density
    Sigma_gas = velocity_to_surface_density(r, data['v_gas'], 'gas')
    Sigma_disk = velocity_to_surface_density(r, data['v_disk'], 'disk')
    Sigma_bulge = velocity_to_surface_density(r, data['v_bulge'], 'bulge')
    
    # Total surface density
    Sigma_total = Sigma_gas + Sigma_disk + Sigma_bulge
    
    # Compute LNAL velocity from total Sigma
    v_lnal = lnal_velocity_from_sigma(r, Sigma_total)
    
    # Also compute from summing component velocities in quadrature
    v_components_quad = np.sqrt(data['v_gas']**2 + data['v_disk']**2 + data['v_bulge']**2)
    
    # Compute chi-squared
    chi2 = np.sum(((v_lnal - data['v_obs']) / data['v_err'])**2)
    chi2_reduced = chi2 / len(r)
    
    return {
        'galaxy': galaxy_name,
        'r': r,
        'Sigma_gas': Sigma_gas,
        'Sigma_disk': Sigma_disk,
        'Sigma_bulge': Sigma_bulge,
        'Sigma_total': Sigma_total,
        'v_obs': data['v_obs'],
        'v_err': data['v_err'],
        'v_lnal': v_lnal,
        'v_components': v_components_quad,
        'chi2': chi2,
        'chi2_reduced': chi2_reduced
    }


def lnal_velocity_from_sigma(r, Sigma):
    """Compute LNAL velocity from surface density."""
    # Enclosed mass
    if len(r) > 1:
        M_enc = 2 * np.pi * cumulative_trapezoid(r * Sigma, r, initial=0)
    else:
        M_enc = np.zeros_like(r)
    
    # Newtonian acceleration
    g_newton = G * M_enc / r**2
    g_newton[0] = g_newton[1] if len(g_newton) > 1 else 0
    
    # LNAL modification
    x = g_newton / G_DAGGER
    mu = x / np.sqrt(1 + x**2)
    g_total = g_newton / mu
    
    # Velocity
    v = np.sqrt(r * g_total)
    return v


def plot_component_analysis(result, save_path=None):
    """Plot component analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    r_kpc = result['r'] / kpc
    
    # Surface density components
    ax = axes[0, 0]
    Sigma_scale = (pc/M_sun)**2  # Convert to M_sun/pc^2
    
    ax.semilogy(r_kpc, result['Sigma_gas'] * Sigma_scale, 'g-', 
                label='Gas', linewidth=2)
    ax.semilogy(r_kpc, result['Sigma_disk'] * Sigma_scale, 'b-', 
                label='Disk', linewidth=2)
    if np.any(result['Sigma_bulge'] > 0):
        ax.semilogy(r_kpc, result['Sigma_bulge'] * Sigma_scale, 'r-', 
                    label='Bulge', linewidth=2)
    ax.semilogy(r_kpc, result['Sigma_total'] * Sigma_scale, 'k-', 
                label='Total', linewidth=2.5)
    
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Σ [M⊙/pc²]')
    ax.set_title('Surface Density Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.1, 1e4)
    
    # Rotation curve
    ax = axes[0, 1]
    ax.errorbar(r_kpc, result['v_obs']/1000, yerr=result['v_err']/1000,
                fmt='ko', markersize=4, label='Observed', alpha=0.7)
    ax.plot(r_kpc, result['v_lnal']/1000, 'r-', linewidth=2,
            label='LNAL from Σ_total')
    ax.plot(r_kpc, result['v_components']/1000, 'b--', linewidth=2,
            label='√(ΣV_i²) (Newtonian)')
    
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Velocity [km/s]')
    ax.set_title(f"{result['galaxy']} - Component-based LNAL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residuals
    ax = axes[1, 0]
    residuals = (result['v_lnal'] - result['v_obs']) / result['v_err']
    ax.scatter(r_kpc, residuals, c='purple', s=30)
    ax.axhline(y=0, color='k', linestyle='--')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('(LNAL - Obs) / Error')
    ax.set_title('Normalized Residuals')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 5)
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    M_gas = 2 * np.pi * np.trapz(result['r'] * result['Sigma_gas'], result['r'])
    M_disk = 2 * np.pi * np.trapz(result['r'] * result['Sigma_disk'], result['r'])
    M_bulge = 2 * np.pi * np.trapz(result['r'] * result['Sigma_bulge'], result['r'])
    M_total = M_gas + M_disk + M_bulge
    
    text = f"Component Masses:\n\n"
    text += f"Gas:   {M_gas/M_sun:.2e} M⊙\n"
    text += f"Disk:  {M_disk/M_sun:.2e} M⊙\n"
    text += f"Bulge: {M_bulge/M_sun:.2e} M⊙\n"
    text += f"Total: {M_total/M_sun:.2e} M⊙\n\n"
    text += f"χ²/N = {result['chi2_reduced']:.2f}\n\n"
    text += "Method: Direct component inversion\n"
    text += "No free parameters in gravity law"
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    return fig


def analyze_all_components(galaxy_list, output_dir='lnal_component_results'):
    """Analyze all galaxies using component decomposition."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for galaxy in galaxy_list:
        print(f"\nAnalyzing {galaxy} components...")
        result = analyze_components(galaxy)
        
        if result is not None:
            # Plot
            plot_component_analysis(
                result, 
                save_path=os.path.join(output_dir, f'{galaxy}_components.png')
            )
            
            # Store summary
            results.append({
                'galaxy': galaxy,
                'chi2': result['chi2'],
                'chi2_reduced': result['chi2_reduced'],
                'n_data': len(result['r'])
            })
            
            print(f"  χ²/N = {result['chi2_reduced']:.2f}")
    
    # Save summary
    with open(os.path.join(output_dir, 'component_analysis_summary.json'), 'w') as f:
        json.dump({
            'description': 'LNAL analysis using SPARC component decomposition',
            'method': 'Direct inversion of V_gas, V_disk, V_bulge to get Σ(r)',
            'galaxies': results
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    print("LNAL Component Analyzer")
    print("=" * 60)
    print("Using SPARC's V_gas, V_disk, V_bulge decomposition")
    
    # Test galaxies
    test_galaxies = ['NGC3198', 'NGC2403', 'NGC6503', 'DDO154']
    
    results = analyze_all_components(test_galaxies)
    
    # Summary
    chi2_values = [r['chi2_reduced'] for r in results]
    print(f"\n{'='*60}")
    print(f"Component analysis complete!")
    print(f"\nχ²/N statistics:")
    print(f"  Mean: {np.mean(chi2_values):.2f}")
    print(f"  Min: {np.min(chi2_values):.2f}")
    print(f"  Max: {np.max(chi2_values):.2f}")
    print(f"\nResults saved to lnal_component_results/") 