#!/usr/bin/env python3
"""
Parameter Optimization for Recognition Science Gravity
======================================================
Grid search to find optimal parameters for SPARC galaxies
"""

import numpy as np
import json
import os
import glob
from rs_gravity_tunable import TunableGravitySolver, GalaxyData
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
DATA_DIR = "Rotmod_LTG"
RESULTS_FILE = "optimization_results.json"
MAX_GALAXIES = 20  # Use subset for faster optimization

# Parameter search ranges
PARAM_GRID = {
    'lambda_eff': np.array([40, 50, 63, 80, 100]) * 1e-6,  # μm to m
    'h_scale': np.array([200, 250, 300, 350, 400]) * 3.086e16,  # pc to m
    'beta_scale': np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
    'mu_scale': np.array([0.5, 0.75, 1.0, 1.5, 2.0]),
    'coupling_scale': np.array([0.5, 0.75, 1.0, 1.5, 2.0])
}

def load_galaxy_from_file(filepath):
    """Load galaxy data from rotmod file"""
    name = os.path.basename(filepath).replace('_rotmod.dat', '')
    
    try:
        data = np.loadtxt(filepath, skiprows=1)
        
        # Extract columns
        R_kpc = data[:, 0]
        v_obs = data[:, 1]
        
        # Check for valid data
        if len(R_kpc) < 5:
            return None
            
        # Velocity errors (3% or 2 km/s minimum)
        v_err = np.maximum(0.03 * v_obs, 2.0)
        
        # Surface densities
        if data.shape[1] >= 7:
            sigma_gas = data[:, 5] * 1.33  # He correction
            sigma_disk = data[:, 6] * 0.5   # M/L ratio
        else:
            # Estimate if columns missing
            sigma_gas = 10 * np.exp(-R_kpc / 2)
            sigma_disk = 100 * np.exp(-R_kpc / 3)
        
        return GalaxyData(
            name=name,
            R_kpc=R_kpc,
            v_obs=v_obs,
            v_err=v_err,
            sigma_gas=sigma_gas,
            sigma_disk=sigma_disk
        )
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def evaluate_parameters(params, galaxies):
    """Evaluate parameter set on galaxy sample"""
    solver = TunableGravitySolver(**params)
    
    chi2_values = []
    for galaxy in galaxies:
        try:
            result = solver.solve_galaxy(galaxy)
            if not np.isnan(result['chi2_reduced']):
                chi2_values.append(result['chi2_reduced'])
        except Exception as e:
            print(f"Error solving {galaxy.name}: {e}")
            continue
    
    if len(chi2_values) > 0:
        return np.mean(chi2_values), np.median(chi2_values), len(chi2_values)
    else:
        return np.inf, np.inf, 0

def grid_search_optimization(galaxies):
    """Perform grid search optimization"""
    print(f"Starting grid search optimization on {len(galaxies)} galaxies...")
    print(f"Total parameter combinations: {np.prod([len(v) for v in PARAM_GRID.values()])}")
    
    best_params = None
    best_mean_chi2 = np.inf
    all_results = []
    
    # Start with coarse grid
    param_names = list(PARAM_GRID.keys())
    param_values = [PARAM_GRID[name] for name in param_names]
    
    # Coarse search - test corners and center
    coarse_indices = [(0, 2, 4), (0, 2, 4), (0, 2, 4), (0, 2, 4), (0, 2, 4)]
    
    print("\nPhase 1: Coarse grid search...")
    for i0 in coarse_indices[0]:
        for i1 in coarse_indices[1]:
            for i2 in coarse_indices[2]:
                for i3 in coarse_indices[3]:
                    for i4 in coarse_indices[4]:
                        params = {
                            param_names[0]: param_values[0][i0],
                            param_names[1]: param_values[1][i1],
                            param_names[2]: param_values[2][i2],
                            param_names[3]: param_values[3][i3],
                            param_names[4]: param_values[4][i4]
                        }
                        
                        mean_chi2, median_chi2, n_success = evaluate_parameters(params, galaxies)
                        
                        result = {
                            'params': params,
                            'mean_chi2': mean_chi2,
                            'median_chi2': median_chi2,
                            'n_success': n_success
                        }
                        all_results.append(result)
                        
                        if mean_chi2 < best_mean_chi2:
                            best_mean_chi2 = mean_chi2
                            best_params = params.copy()
                            print(f"New best: χ²/N = {mean_chi2:.3f}")
    
    # Fine search around best point
    if best_params:
        print("\nPhase 2: Fine search around best parameters...")
        
        # Search ±1 index around best
        best_indices = []
        for i, name in enumerate(param_names):
            best_val = best_params[name]
            idx = np.argmin(np.abs(param_values[i] - best_val))
            best_indices.append(idx)
        
        for di0 in [-1, 0, 1]:
            for di1 in [-1, 0, 1]:
                for di2 in [-1, 0, 1]:
                    i0 = np.clip(best_indices[0] + di0, 0, len(param_values[0])-1)
                    i1 = np.clip(best_indices[1] + di1, 0, len(param_values[1])-1)
                    i2 = np.clip(best_indices[2] + di2, 0, len(param_values[2])-1)
                    
                    params = {
                        param_names[0]: param_values[0][i0],
                        param_names[1]: param_values[1][i1],
                        param_names[2]: param_values[2][i2],
                        param_names[3]: best_params[param_names[3]],  # Keep fixed
                        param_names[4]: best_params[param_names[4]]   # Keep fixed
                    }
                    
                    mean_chi2, median_chi2, n_success = evaluate_parameters(params, galaxies)
                    
                    if mean_chi2 < best_mean_chi2:
                        best_mean_chi2 = mean_chi2
                        best_params = params.copy()
                        print(f"Improved: χ²/N = {mean_chi2:.3f}")
    
    return best_params, best_mean_chi2, all_results

def main():
    """Main optimization routine"""
    print("Recognition Science Gravity Parameter Optimization")
    print("="*60)
    
    # Load galaxies
    print(f"\nLoading galaxies from {DATA_DIR}...")
    galaxy_files = glob.glob(os.path.join(DATA_DIR, "*_rotmod.dat"))[:MAX_GALAXIES]
    
    galaxies = []
    for filepath in galaxy_files:
        galaxy = load_galaxy_from_file(filepath)
        if galaxy is not None:
            galaxies.append(galaxy)
    
    print(f"Loaded {len(galaxies)} galaxies successfully")
    
    if len(galaxies) == 0:
        print("No galaxies loaded! Check data directory.")
        return
    
    # Run optimization
    start_time = datetime.now()
    best_params, best_chi2, all_results = grid_search_optimization(galaxies)
    end_time = datetime.now()
    
    print(f"\nOptimization completed in {(end_time - start_time).total_seconds():.1f} seconds")
    
    # Save results
    results_data = {
        'best_parameters': best_params,
        'best_mean_chi2': best_chi2,
        'n_galaxies': len(galaxies),
        'timestamp': datetime.now().isoformat(),
        'all_results': sorted(all_results, key=lambda x: x['mean_chi2'])[:10]  # Top 10
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_FILE}")
    
    # Display best parameters
    print("\n" + "="*60)
    print("OPTIMAL PARAMETERS:")
    print("="*60)
    if best_params:
        print(f"λ_eff = {best_params['lambda_eff']*1e6:.1f} μm")
        print(f"h_scale = {best_params['h_scale']/3.086e16:.0f} pc")
        print(f"β_scale = {best_params['beta_scale']:.2f}")
        print(f"μ_scale = {best_params['mu_scale']:.2f}")
        print(f"coupling_scale = {best_params['coupling_scale']:.2f}")
        print(f"\nMean χ²/N = {best_chi2:.3f}")
    
    # Test best parameters on a galaxy
    if best_params and len(galaxies) > 0:
        print("\nTesting on first galaxy...")
        solver = TunableGravitySolver(**best_params)
        test_galaxy = galaxies[0]
        result = solver.solve_galaxy(test_galaxy)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.errorbar(test_galaxy.R_kpc, test_galaxy.v_obs, yerr=test_galaxy.v_err,
                    fmt='ko', alpha=0.7, markersize=5, label='Observed')
        ax1.plot(test_galaxy.R_kpc, result['v_model'], 'r-', linewidth=2,
                label=f"Model (χ²/N={result['chi2_reduced']:.2f})")
        ax1.plot(test_galaxy.R_kpc, result['v_newton'], 'b--', alpha=0.7,
                label='Newtonian')
        ax1.set_xlabel('Radius (kpc)')
        ax1.set_ylabel('Velocity (km/s)')
        ax1.set_title(f'{test_galaxy.name} - Optimized Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(test_galaxy.R_kpc, result['residuals'], 'go-', alpha=0.7)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax2.fill_between(test_galaxy.R_kpc, -test_galaxy.v_err, test_galaxy.v_err,
                        alpha=0.2, color='gray')
        ax2.set_xlabel('Radius (kpc)')
        ax2.set_ylabel('Residuals (km/s)')
        ax2.set_title('Fit Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimized_fit_example.png', dpi=150)
        plt.show()

if __name__ == "__main__":
    main() 