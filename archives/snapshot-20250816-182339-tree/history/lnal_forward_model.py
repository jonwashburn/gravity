#!/usr/bin/env python3
"""
LNAL Forward Model
==================
Forward model observations including beam smearing, 
inclination effects, and vertical structure.
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


def exponential_disk(r, M_disk, R_d):
    """Exponential disk surface density profile."""
    Sigma_0 = M_disk / (2 * np.pi * R_d**2)
    return Sigma_0 * np.exp(-r / R_d)


def gas_profile(r, M_gas, R_g, n=1.0):
    """Gas surface density profile (Sersic-like)."""
    Sigma_0 = M_gas / (2 * np.pi * R_g**2 * n)
    return Sigma_0 * np.exp(-(r / R_g)**(1/n))


def bulge_profile(r, M_bulge, R_b):
    """Bulge contribution (projected from 3D)."""
    # Hernquist profile projected
    Sigma_0 = M_bulge / (2 * np.pi * R_b**2)
    x = r / R_b
    return Sigma_0 * x / (1 + x**2)**2


def beam_smear(r, v, beam_fwhm):
    """Apply beam smearing to velocity profile."""
    # Convert FWHM to sigma in same units as r
    sigma = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Determine smoothing scale in points
    if len(r) > 1:
        dr = np.median(np.diff(r))
        sigma_points = sigma / dr
        
        # Apply Gaussian smoothing
        if sigma_points > 0.1:
            v_smeared = gaussian_filter1d(v, sigma_points, mode='nearest')
        else:
            v_smeared = v.copy()
    else:
        v_smeared = v.copy()
    
    return v_smeared


def inclination_correction(v_los, inclination):
    """Convert line-of-sight velocity to circular velocity."""
    # inclination in degrees
    inc_rad = np.radians(inclination)
    return v_los / np.sin(inc_rad)


def vertical_structure_correction(r, h_z, R_d):
    """Correction for disk vertical structure."""
    # Approximate correction for thick disk
    # h_z is scale height
    z_factor = 1.0 + (h_z / R_d) * np.exp(-r / R_d)
    return z_factor


def forward_model_galaxy(r, params, obs_params):
    """
    Forward model galaxy rotation curve.
    
    params: dict with M_disk, R_d, M_gas, R_g, M_bulge, R_b
    obs_params: dict with beam_fwhm, inclination, h_z
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
    
    # Apply vertical structure correction
    if 'h_z' in obs_params:
        z_corr = vertical_structure_correction(r, obs_params['h_z'], params['R_d'])
        v_circ = v_circ * np.sqrt(z_corr)
    
    # Project to line-of-sight
    if 'inclination' in obs_params:
        v_los = v_circ * np.sin(np.radians(obs_params['inclination']))
    else:
        v_los = v_circ
    
    # Apply beam smearing
    if 'beam_fwhm' in obs_params:
        v_observed = beam_smear(r, v_los, obs_params['beam_fwhm'])
    else:
        v_observed = v_los
    
    return v_observed, {
        'Sigma_disk': Sigma_disk,
        'Sigma_gas': Sigma_gas,
        'Sigma_bulge': Sigma_bulge,
        'Sigma_total': Sigma_total,
        'v_circ': v_circ,
        'v_los': v_los,
        'v_observed': v_observed
    }


def fit_forward_model(r, v_obs, v_err, initial_params, obs_params):
    """Fit forward model to observed data."""
    
    def chi2(param_array):
        # Unpack parameters
        params = {
            'M_disk': param_array[0] * M_sun,
            'R_d': param_array[1] * kpc,
            'M_gas': param_array[2] * M_sun,
            'R_g': param_array[3] * kpc,
            'M_bulge': param_array[4] * M_sun if len(param_array) > 4 else 0,
            'R_b': param_array[5] * kpc if len(param_array) > 5 else param_array[1] * kpc
        }
        
        # Forward model
        v_model, _ = forward_model_galaxy(r, params, obs_params)
        
        # Chi-squared
        residuals = (v_model - v_obs) / v_err
        return np.sum(residuals**2)
    
    # Initial guess in fitting units
    x0 = [
        initial_params['M_disk'] / M_sun,
        initial_params['R_d'] / kpc,
        initial_params['M_gas'] / M_sun,
        initial_params['R_g'] / kpc
    ]
    
    if 'M_bulge' in initial_params and initial_params['M_bulge'] > 0:
        x0.extend([
            initial_params['M_bulge'] / M_sun,
            initial_params.get('R_b', initial_params['R_d']) / kpc
        ])
    
    # Bounds
    bounds = [
        (1e8, 1e12),   # M_disk in M_sun
        (0.1, 20),     # R_d in kpc
        (1e7, 1e11),   # M_gas in M_sun
        (0.1, 50),     # R_g in kpc
    ]
    
    if len(x0) > 4:
        bounds.extend([
            (0, 1e11),     # M_bulge in M_sun
            (0.1, 10)      # R_b in kpc
        ])
    
    # Optimize
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
    
    return best_params, result.fun, result


def analyze_with_forward_model(galaxy_name, rotmod_dir='Rotmod_LTG'):
    """Analyze galaxy with forward modeling."""
    # Load data
    from lnal_sparc_loader_fixed import load_sparc_galaxy_fixed
    galaxy_data = load_sparc_galaxy_fixed(galaxy_name, rotmod_dir)
    
    if galaxy_data is None:
        return None
    
    r = galaxy_data.r
    v_obs = galaxy_data.v_obs
    v_err = galaxy_data.v_err
    
    # Estimate observational parameters
    obs_params = {
        'beam_fwhm': 1.0 * kpc,  # 1 kpc beam
        'inclination': 60,  # degrees, typical
        'h_z': 0.2 * kpc  # scale height
    }
    
    # Initial parameter guess from data
    v_flat = np.median(v_obs[len(v_obs)//2:])
    R_opt = r[np.argmax(v_obs)] / 2  # rough scale
    
    initial_params = {
        'M_disk': 5e10 * M_sun,
        'R_d': R_opt,
        'M_gas': 1e10 * M_sun,
        'R_g': 2 * R_opt,
        'M_bulge': 1e9 * M_sun if galaxy_name.startswith('NGC') else 0,
        'R_b': 0.2 * R_opt
    }
    
    # Fit
    print(f"Fitting {galaxy_name} with forward model...")
    best_params, chi2, opt_result = fit_forward_model(
        r, v_obs, v_err, initial_params, obs_params
    )
    
    # Generate best-fit model
    v_model, components = forward_model_galaxy(r, best_params, obs_params)
    
    chi2_reduced = chi2 / len(r)
    
    return {
        'galaxy': galaxy_name,
        'r': r,
        'v_obs': v_obs,
        'v_err': v_err,
        'v_model': v_model,
        'components': components,
        'best_params': best_params,
        'obs_params': obs_params,
        'chi2': chi2,
        'chi2_reduced': chi2_reduced,
        'success': opt_result.success
    }


def plot_forward_model_fit(result, save_path=None):
    """Plot forward model fit results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    r_kpc = result['r'] / kpc
    components = result['components']
    
    # Surface density profiles
    ax = axes[0, 0]
    Sigma_scale = (pc/M_sun)**2
    
    ax.semilogy(r_kpc, components['Sigma_gas'] * Sigma_scale, 'g-', 
                label='Gas', linewidth=2)
    ax.semilogy(r_kpc, components['Sigma_disk'] * Sigma_scale, 'b-', 
                label='Disk', linewidth=2)
    if np.any(components['Sigma_bulge'] > 0):
        ax.semilogy(r_kpc, components['Sigma_bulge'] * Sigma_scale, 'r-', 
                    label='Bulge', linewidth=2)
    ax.semilogy(r_kpc, components['Sigma_total'] * Sigma_scale, 'k-', 
                label='Total', linewidth=2.5)
    
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Σ [M⊙/pc²]')
    ax.set_title('Model Surface Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.1, 1e4)
    
    # Rotation curves
    ax = axes[0, 1]
    ax.errorbar(r_kpc, result['v_obs']/1000, yerr=result['v_err']/1000,
                fmt='ko', markersize=4, label='Observed', alpha=0.7)
    ax.plot(r_kpc, result['v_model']/1000, 'r-', linewidth=2.5,
            label='Forward Model')
    ax.plot(r_kpc, components['v_circ']/1000, 'b--', linewidth=1.5,
            label='Intrinsic V_circ', alpha=0.7)
    
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Velocity [km/s]')
    ax.set_title(f"{result['galaxy']} - Forward Model Fit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residuals
    ax = axes[1, 0]
    residuals = (result['v_model'] - result['v_obs']) / result['v_err']
    ax.scatter(r_kpc, residuals, c='purple', s=30)
    ax.axhline(y=0, color='k', linestyle='--')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('(Model - Obs) / Error')
    ax.set_title('Normalized Residuals')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 5)
    
    # Parameters and effects
    ax = axes[1, 1]
    ax.axis('off')
    
    params = result['best_params']
    obs = result['obs_params']
    
    text = "Best-fit Parameters:\n\n"
    text += f"M_disk = {params['M_disk']/M_sun:.2e} M⊙\n"
    text += f"R_disk = {params['R_d']/kpc:.2f} kpc\n"
    text += f"M_gas = {params['M_gas']/M_sun:.2e} M⊙\n"
    text += f"R_gas = {params['R_g']/kpc:.2f} kpc\n"
    if 'M_bulge' in params and params['M_bulge'] > 0:
        text += f"M_bulge = {params['M_bulge']/M_sun:.2e} M⊙\n"
    
    text += f"\nObservational Effects:\n"
    text += f"Beam FWHM = {obs['beam_fwhm']/kpc:.1f} kpc\n"
    text += f"Inclination = {obs['inclination']:.0f}°\n"
    text += f"Scale height = {obs['h_z']/kpc:.2f} kpc\n"
    
    text += f"\nχ²/N = {result['chi2_reduced']:.2f}"
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    return fig


def analyze_all_forward_models(galaxy_list, output_dir='lnal_forward_model_results'):
    """Analyze galaxies with forward modeling."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for galaxy in galaxy_list:
        print(f"\nAnalyzing {galaxy} with forward model...")
        result = analyze_with_forward_model(galaxy)
        
        if result is not None:
            # Plot
            plot_forward_model_fit(
                result,
                save_path=os.path.join(output_dir, f'{galaxy}_forward_model.png')
            )
            
            # Store summary
            results.append({
                'galaxy': galaxy,
                'chi2': result['chi2'],
                'chi2_reduced': result['chi2_reduced'],
                'n_data': len(result['r']),
                'success': result['success']
            })
            
            print(f"  χ²/N = {result['chi2_reduced']:.2f} (success: {result['success']})")
    
    # Save summary
    with open(os.path.join(output_dir, 'forward_model_summary.json'), 'w') as f:
        json.dump({
            'description': 'LNAL analysis with forward modeling',
            'method': 'Forward model including beam smearing, inclination, vertical structure',
            'galaxies': results
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    print("LNAL Forward Model Analysis")
    print("=" * 60)
    print("Including observational effects:")
    print("- Beam smearing")
    print("- Inclination projection")
    print("- Vertical disk structure")
    
    # Test galaxies
    test_galaxies = ['NGC3198', 'NGC2403', 'NGC6503', 'DDO154']
    
    results = analyze_all_forward_models(test_galaxies)
    
    # Summary
    chi2_values = [r['chi2_reduced'] for r in results]
    print(f"\n{'='*60}")
    print(f"Forward model analysis complete!")
    print(f"\nχ²/N statistics:")
    print(f"  Mean: {np.mean(chi2_values):.2f}")
    print(f"  Min: {np.min(chi2_values):.2f}")
    print(f"  Max: {np.max(chi2_values):.2f}")
    print(f"\nResults saved to lnal_forward_model_results/") 