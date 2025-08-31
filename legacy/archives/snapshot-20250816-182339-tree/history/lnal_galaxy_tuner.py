#!/usr/bin/env python3
"""
LNAL Galaxy Tuner
=================
Find optimal surface density profiles for SPARC galaxies
using the pure LNAL formula with regularization.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import json

# Constants
G = 6.67430e-11  # m³/kg/s²
G_DAGGER = 1.2e-10  # m/s² (MOND scale)
kpc = 3.0856775814913673e19  # m
pc = kpc / 1000
M_sun = 1.98847e30  # kg


class GalaxyTuner:
    """
    Tune galaxy surface density to match rotation curve.
    Uses parametric model with physical constraints.
    """
    
    def __init__(self, r_data, v_obs, v_err, name="Galaxy"):
        self.r_data = r_data
        self.v_obs = v_obs
        self.v_err = v_err
        self.name = name
        
        # Set up radial grid for model
        self.r_model = np.logspace(
            np.log10(0.1 * r_data.min()),
            np.log10(2 * r_data.max()),
            200
        )
        
    def surface_density_model(self, r, params):
        """
        Flexible surface density model.
        
        Components:
        1. Inner exponential disk
        2. Outer exponential disk  
        3. Central concentration (bulge/bar)
        
        params = [M1, R1, M2, R2, M_c, R_c]
        """
        M1, R1, M2, R2, M_c, R_c = params
        
        # Double exponential disk
        Sigma1 = (M1 / (2 * np.pi * R1**2)) * np.exp(-r / R1)
        Sigma2 = (M2 / (2 * np.pi * R2**2)) * np.exp(-r / R2)
        
        # Central component (Sérsic n=1 for simplicity)
        if M_c > 0 and R_c > 0:
            Sigma_c = (M_c / (2 * np.pi * R_c**2)) * np.exp(-r / R_c)
        else:
            Sigma_c = 0
        
        return Sigma1 + Sigma2 + Sigma_c
    
    def lnal_velocity(self, r, Sigma):
        """Compute velocity from surface density using pure LNAL."""
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
    
    def objective(self, params):
        """Objective function: χ² + regularization."""
        # Unpack and convert to physical units
        M1 = 10**params[0] * M_sun
        R1 = params[1] * kpc
        M2 = 10**params[2] * M_sun
        R2 = params[3] * kpc
        M_c = 10**params[4] * M_sun if params[4] > 0 else 0
        R_c = params[5] * kpc
        
        physical_params = [M1, R1, M2, R2, M_c, R_c]
        
        # Compute model
        Sigma_model = self.surface_density_model(self.r_model, physical_params)
        v_model_full = self.lnal_velocity(self.r_model, Sigma_model)
        
        # Interpolate to data points
        v_model = np.interp(self.r_data, self.r_model, v_model_full)
        
        # Chi-squared
        chi2 = np.sum(((v_model - self.v_obs) / self.v_err)**2)
        
        # Regularization: prefer smooth, physical solutions
        # Penalize very different scale lengths
        reg1 = 10 * (params[1] - params[3])**2 if R2 > R1 else 0
        
        # Penalize unphysical mass ratios
        mass_ratio = M2 / M1
        reg2 = 5 * max(0, mass_ratio - 10)**2  # M2 shouldn't be >10x M1
        
        return chi2 + reg1 + reg2
    
    def optimize(self, method='differential_evolution'):
        """Find optimal surface density parameters."""
        # Parameter bounds [log10(M/M_sun), R/kpc]
        bounds = [
            (8, 12),    # log10(M1/M_sun) - inner disk mass
            (0.1, 10),  # R1 [kpc] - inner scale length
            (7, 11),    # log10(M2/M_sun) - outer disk mass
            (1, 30),    # R2 [kpc] - outer scale length
            (6, 10),    # log10(M_c/M_sun) - central mass (optional)
            (0.01, 2),  # R_c [kpc] - central scale length
        ]
        
        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective, bounds, 
                maxiter=300, seed=42, workers=-1
            )
        else:
            # Initial guess
            x0 = [10, 2, 9, 5, 8, 0.5]
            result = minimize(
                self.objective, x0,
                method='Nelder-Mead',
                bounds=bounds
            )
        
        # Convert back to physical parameters
        M1 = 10**result.x[0] * M_sun
        R1 = result.x[1] * kpc
        M2 = 10**result.x[2] * M_sun
        R2 = result.x[3] * kpc
        M_c = 10**result.x[4] * M_sun if result.x[4] > 0 else 0
        R_c = result.x[5] * kpc
        
        self.best_params = [M1, R1, M2, R2, M_c, R_c]
        self.best_chi2 = result.fun
        
        return self.best_params
    
    def plot_solution(self, save_path=None):
        """Plot the tuned galaxy solution."""
        # Compute best-fit model
        Sigma_best = self.surface_density_model(self.r_model, self.best_params)
        v_best = self.lnal_velocity(self.r_model, Sigma_best)
        v_interp = np.interp(self.r_data, self.r_model, v_best)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Rotation curve
        ax = axes[0, 0]
        ax.errorbar(self.r_data/kpc, self.v_obs/1000, yerr=self.v_err/1000,
                   fmt='ko', markersize=5, label='Data', alpha=0.7)
        ax.plot(self.r_model/kpc, v_best/1000, 'r-', linewidth=2, 
                label='LNAL model')
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel('Velocity [km/s]')
        ax.set_title(f'{self.name} - Tuned LNAL Model')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Surface density
        ax = axes[0, 1]
        Sigma_solar = Sigma_best * (pc/M_sun)**2
        ax.semilogy(self.r_model/kpc, Sigma_solar, 'b-', linewidth=2)
        
        # Show components
        M1, R1, M2, R2, M_c, R_c = self.best_params
        Sigma1 = (M1 / (2 * np.pi * R1**2)) * np.exp(-self.r_model / R1)
        Sigma2 = (M2 / (2 * np.pi * R2**2)) * np.exp(-self.r_model / R2)
        ax.semilogy(self.r_model/kpc, Sigma1 * (pc/M_sun)**2, 'g--', 
                   alpha=0.5, label='Inner disk')
        ax.semilogy(self.r_model/kpc, Sigma2 * (pc/M_sun)**2, 'orange', 
                   linestyle='--', alpha=0.5, label='Outer disk')
        
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel('Σ [M⊙/pc²]')
        ax.set_title('Surface Density Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.1, 1e4)
        
        # Residuals
        ax = axes[1, 0]
        residuals = (v_interp - self.v_obs) / self.v_err
        ax.scatter(self.r_data/kpc, residuals, c='purple', s=30)
        ax.axhline(y=0, color='k', linestyle='--')
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel('(Model - Data) / Error')
        ax.set_title('Normalized Residuals')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 5)
        
        # Parameter summary
        ax = axes[1, 1]
        ax.axis('off')
        
        M_total = M1 + M2 + M_c
        chi2_reduced = self.best_chi2 / len(self.r_data)
        
        text = f"Best-fit Parameters:\n\n"
        text += f"Inner disk:\n"
        text += f"  M₁ = {M1/M_sun:.2e} M⊙\n"
        text += f"  R₁ = {R1/kpc:.2f} kpc\n\n"
        text += f"Outer disk:\n"
        text += f"  M₂ = {M2/M_sun:.2e} M⊙\n"
        text += f"  R₂ = {R2/kpc:.2f} kpc\n\n"
        if M_c > 0:
            text += f"Central:\n"
            text += f"  Mc = {M_c/M_sun:.2e} M⊙\n"
            text += f"  Rc = {R_c/kpc:.2f} kpc\n\n"
        text += f"Total mass: {M_total/M_sun:.2e} M⊙\n"
        text += f"χ²/N = {chi2_reduced:.2f}"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
        
        return fig


def tune_sparc_galaxy(galaxy_name, output_dir='lnal_tuned_galaxies'):
    """Tune a SPARC galaxy and save results."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load rotation curve
    from lnal_pure_pipeline import load_rotation_curve
    rot_data = load_rotation_curve(galaxy_name)
    
    if rot_data is None:
        print(f"Could not load {galaxy_name}")
        return None
    
    print(f"\nTuning {galaxy_name}...")
    
    # Create tuner
    tuner = GalaxyTuner(
        rot_data['r'], 
        rot_data['v_obs'], 
        rot_data['v_err'],
        galaxy_name
    )
    
    # Optimize
    best_params = tuner.optimize()
    
    # Plot
    tuner.plot_solution(
        save_path=os.path.join(output_dir, f'{galaxy_name}_tuned.png')
    )
    
    # Report
    M_total = sum(best_params[i] for i in [0, 2, 4])
    chi2_reduced = tuner.best_chi2 / len(rot_data['r'])
    
    print(f"  Total mass: {M_total/M_sun:.2e} M_sun")
    print(f"  χ²/N = {chi2_reduced:.2f}")
    
    return {
        'galaxy': galaxy_name,
        'params': [p/M_sun if i in [0,2,4] else p/kpc for i, p in enumerate(best_params)],
        'chi2': tuner.best_chi2,
        'chi2_reduced': chi2_reduced,
        'n_data': len(rot_data['r'])
    }


if __name__ == "__main__":
    print("LNAL Galaxy Tuner")
    print("=" * 60)
    print("Finding optimal Σ(r) for SPARC galaxies using pure LNAL")
    
    # Test galaxies
    test_galaxies = ['NGC3198', 'NGC2403', 'NGC6503', 'DDO154']
    
    results = []
    for galaxy in test_galaxies:
        result = tune_sparc_galaxy(galaxy)
        if result:
            results.append(result)
    
    # Save summary
    with open('lnal_tuned_galaxies/tuning_summary.json', 'w') as f:
        json.dump({
            'description': 'LNAL galaxy tuning results',
            'method': 'Multi-component surface density optimization',
            'galaxies': results
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Tuning complete!")
    print("Results saved to lnal_tuned_galaxies/")
    
    # Summary statistics
    chi2_values = [r['chi2_reduced'] for r in results]
    print(f"\nχ²/N statistics:")
    print(f"  Mean: {np.mean(chi2_values):.2f}")
    print(f"  Min: {np.min(chi2_values):.2f}")
    print(f"  Max: {np.max(chi2_values):.2f}") 