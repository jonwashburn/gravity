#!/usr/bin/env python3
"""
LNAL Forward Model V2
=====================
Simplified version that loads SPARC data directly.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
import json
import os

# Constants
G = 6.67430e-11  # m³/kg/s²
G_DAGGER = 1.2e-10  # m/s² (MOND scale)
kpc = 3.0856775814913673e19  # m
pc = kpc / 1000
M_sun = 1.98847e30  # kg


def load_sparc_rotmod(galaxy_name, rotmod_dir='Rotmod_LTG'):
    """Load SPARC rotation curve data."""
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


def exponential_disk(r, M_disk, R_d):
    """Exponential disk surface density profile."""
    Sigma_0 = M_disk / (2 * np.pi * R_d**2)
    return Sigma_0 * np.exp(-r / R_d)


def gas_profile(r, M_gas, R_g):
    """Gas surface density profile."""
    Sigma_0 = M_gas / (2 * np.pi * R_g**2)
    return Sigma_0 * np.exp(-r / R_g)


def bulge_profile(r, M_bulge, R_b):
    """Bulge contribution."""
    if M_bulge == 0:
        return np.zeros_like(r)
    Sigma_0 = M_bulge / (2 * np.pi * R_b**2)
    x = r / R_b
    return Sigma_0 * x / (1 + x**2)**2


def beam_smear(r, v, beam_fwhm):
    """Apply beam smearing to velocity profile."""
    if beam_fwhm <= 0 or len(r) < 3:
        return v
    
    # Convert FWHM to sigma
    sigma = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Determine smoothing scale in points
    dr = np.median(np.diff(r))
    sigma_points = sigma / dr
    
    # Apply Gaussian smoothing
    if sigma_points > 0.1:
        v_smeared = gaussian_filter1d(v, sigma_points, mode='nearest')
    else:
        v_smeared = v.copy()
    
    return v_smeared


def forward_model_simple(r, params, obs_params=None):
    """
    Forward model galaxy rotation curve.
    
    params: dict with M_disk, R_d, M_gas, R_g, M_bulge, R_b
    obs_params: dict with beam_fwhm, inclination
    """
    # Generate surface density profiles
    Sigma_disk = exponential_disk(r, params['M_disk'], params['R_d'])
    Sigma_gas = gas_profile(r, params['M_gas'], params['R_g'])
    Sigma_bulge = bulge_profile(r, params.get('M_bulge', 0), params.get('R_b', params['R_d']))
    
    # Total surface density
    Sigma_total = Sigma_disk + Sigma_gas + Sigma_bulge
    
    # Compute enclosed mass
    if len(r) > 1:
        M_enc = 2 * np.pi * cumulative_trapezoid(r * Sigma_total, r, initial=0)
    else:
        M_enc = np.zeros_like(r)
    
    # Newtonian acceleration
    g_newton = G * M_enc / r**2
    g_newton[0] = g_newton[1] if len(g_newton) > 1 else 0
    
    # LNAL modification
    x = g_newton / G_DAGGER
    mu = x / np.sqrt(1 + x**2)
    g_total = g_newton / mu
    
    # Circular velocity
    v_circ = np.sqrt(r * g_total)
    
    # Apply observational effects if specified
    v_model = v_circ
    if obs_params is not None:
        # Inclination projection
        if 'inclination' in obs_params:
            v_model = v_model * np.sin(np.radians(obs_params['inclination']))
        
        # Beam smearing
        if 'beam_fwhm' in obs_params:
            v_model = beam_smear(r, v_model, obs_params['beam_fwhm'])
    
    return v_model, {
        'Sigma_disk': Sigma_disk,
        'Sigma_gas': Sigma_gas,
        'Sigma_bulge': Sigma_bulge,
        'Sigma_total': Sigma_total,
        'v_circ': v_circ
    }


def fit_galaxy(galaxy_name):
    """Fit forward model to galaxy."""
    # Load data
    data = load_sparc_rotmod(galaxy_name)
    if data is None:
        print(f"Could not load {galaxy_name}")
        return None
    
    r = data['r']
    v_obs = data['v_obs']
    v_err = data['v_err']
    
    # Handle missing errors
    v_err[v_err <= 0] = 5000  # 5 km/s default
    
    # Observational parameters
    obs_params = {
        'beam_fwhm': 1.0 * kpc,  # 1 kpc beam
        'inclination': 60  # degrees
    }
    
    # Initial guess
    v_flat = np.median(v_obs[len(v_obs)//2:])
    R_opt = r[np.argmax(v_obs)] / 2
    
    initial_params = {
        'M_disk': 5e10 * M_sun,
        'R_d': R_opt,
        'M_gas': 1e10 * M_sun,
        'R_g': 2 * R_opt,
        'M_bulge': 1e9 * M_sun if galaxy_name.startswith('NGC') else 0,
        'R_b': 0.2 * R_opt
    }
    
    # Define chi2 function
    def chi2(param_array):
        params = {
            'M_disk': param_array[0] * M_sun,
            'R_d': param_array[1] * kpc,
            'M_gas': param_array[2] * M_sun,
            'R_g': param_array[3] * kpc,
        }
        if len(param_array) > 4:
            params['M_bulge'] = param_array[4] * M_sun
            params['R_b'] = param_array[5] * kpc
        
        v_model, _ = forward_model_simple(r, params, obs_params)
        residuals = (v_model - v_obs) / v_err
        return np.sum(residuals**2)
    
    # Initial guess in fitting units
    x0 = [
        initial_params['M_disk'] / M_sun,
        initial_params['R_d'] / kpc,
        initial_params['M_gas'] / M_sun,
        initial_params['R_g'] / kpc
    ]
    
    # Bounds
    bounds = [
        (1e8, 1e12),   # M_disk
        (0.1, 20),     # R_d
        (1e7, 1e11),   # M_gas
        (0.1, 50),     # R_g
    ]
    
    # Include bulge for NGC galaxies
    if galaxy_name.startswith('NGC'):
        x0.extend([
            initial_params['M_bulge'] / M_sun,
            initial_params['R_b'] / kpc
        ])
        bounds.extend([
            (0, 1e11),     # M_bulge
            (0.1, 10)      # R_b
        ])
    
    # Optimize
    print(f"Fitting {galaxy_name}...")
    result = minimize(chi2, x0, bounds=bounds, method='L-BFGS-B')
    
    # Extract best-fit parameters
    best_params = {
        'M_disk': result.x[0] * M_sun,
        'R_d': result.x[1] * kpc,
        'M_gas': result.x[2] * M_sun,
        'R_g': result.x[3] * kpc,
    }
    
    if len(result.x) > 4:
        best_params['M_bulge'] = result.x[4] * M_sun
        best_params['R_b'] = result.x[5] * kpc
    
    # Generate best-fit model
    v_model, components = forward_model_simple(r, best_params, obs_params)
    
    chi2_reduced = result.fun / len(r)
    
    return {
        'galaxy': galaxy_name,
        'r': r,
        'v_obs': v_obs,
        'v_err': v_err,
        'v_model': v_model,
        'components': components,
        'best_params': best_params,
        'obs_params': obs_params,
        'chi2': result.fun,
        'chi2_reduced': chi2_reduced,
        'success': result.success
    }


def plot_fit(result, save_path=None):
    """Plot forward model fit."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    r_kpc = result['r'] / kpc
    
    # Surface densities
    Sigma_scale = (pc/M_sun)**2
    ax1.semilogy(r_kpc, result['components']['Sigma_gas'] * Sigma_scale, 'g-', label='Gas', lw=2)
    ax1.semilogy(r_kpc, result['components']['Sigma_disk'] * Sigma_scale, 'b-', label='Disk', lw=2)
    if np.any(result['components']['Sigma_bulge'] > 0):
        ax1.semilogy(r_kpc, result['components']['Sigma_bulge'] * Sigma_scale, 'r-', label='Bulge', lw=2)
    ax1.semilogy(r_kpc, result['components']['Sigma_total'] * Sigma_scale, 'k-', label='Total', lw=2.5)
    ax1.set_xlabel('Radius [kpc]')
    ax1.set_ylabel('Σ [M⊙/pc²]')
    ax1.set_title('Model Surface Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.1, 1e4)
    
    # Rotation curve
    ax2.errorbar(r_kpc, result['v_obs']/1000, yerr=result['v_err']/1000,
                 fmt='ko', markersize=4, label='Observed', alpha=0.7)
    ax2.plot(r_kpc, result['v_model']/1000, 'r-', linewidth=2.5, label='Forward Model')
    ax2.plot(r_kpc, result['components']['v_circ']/1000, 'b--', linewidth=1.5,
             label='Intrinsic', alpha=0.7)
    ax2.set_xlabel('Radius [kpc]')
    ax2.set_ylabel('Velocity [km/s]')
    ax2.set_title(f"{result['galaxy']} - Forward Model Fit")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Residuals
    residuals = (result['v_model'] - result['v_obs']) / result['v_err']
    ax3.scatter(r_kpc, residuals, c='purple', s=30)
    ax3.axhline(y=0, color='k', linestyle='--')
    ax3.set_xlabel('Radius [kpc]')
    ax3.set_ylabel('(Model - Obs) / Error')
    ax3.set_title('Normalized Residuals')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 5)
    
    # Parameters
    ax4.axis('off')
    params = result['best_params']
    obs = result['obs_params']
    
    text = "Best-fit Parameters:\n\n"
    text += f"M_disk = {params['M_disk']/M_sun:.2e} M⊙\n"
    text += f"R_disk = {params['R_d']/kpc:.2f} kpc\n"
    text += f"M_gas = {params['M_gas']/M_sun:.2e} M⊙\n"
    text += f"R_gas = {params['R_g']/kpc:.2f} kpc\n"
    if 'M_bulge' in params:
        text += f"M_bulge = {params.get('M_bulge', 0)/M_sun:.2e} M⊙\n"
    
    text += f"\nObservational Effects:\n"
    text += f"Beam = {obs['beam_fwhm']/kpc:.1f} kpc\n"
    text += f"Inclination = {obs['inclination']:.0f}°\n"
    
    text += f"\nχ²/N = {result['chi2_reduced']:.2f}"
    
    ax4.text(0.1, 0.9, text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    return fig


def main():
    """Run forward model analysis."""
    print("LNAL Forward Model V2")
    print("=" * 60)
    
    test_galaxies = ['NGC3198', 'NGC2403', 'NGC6503', 'DDO154']
    output_dir = 'lnal_forward_v2_results'
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for galaxy in test_galaxies:
        result = fit_galaxy(galaxy)
        if result is not None:
            results.append(result)
            plot_fit(result, os.path.join(output_dir, f'{galaxy}_forward_v2.png'))
            print(f"  {galaxy}: χ²/N = {result['chi2_reduced']:.2f}")
    
    # Summary
    chi2_values = [r['chi2_reduced'] for r in results]
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Mean χ²/N: {np.mean(chi2_values):.2f}")
    print(f"  Range: {np.min(chi2_values):.2f} - {np.max(chi2_values):.2f}")
    
    # Save summary
    with open(os.path.join(output_dir, 'forward_v2_summary.json'), 'w') as f:
        json.dump({
            'description': 'LNAL forward model with observational effects',
            'galaxies': [{
                'name': r['galaxy'],
                'chi2_reduced': r['chi2_reduced'],
                'success': r['success']
            } for r in results]
        }, f, indent=2)


if __name__ == "__main__":
    main() 