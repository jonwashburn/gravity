#!/usr/bin/env python3
"""
Final SPARC Analysis with Bayesian Optimized Parameters
========================================================
Produces publication-quality results with full optimization
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
from rs_gravity_tunable_enhanced import EnhancedGravitySolver, GalaxyData, GalaxyParameters
from scipy.optimize import minimize
from datetime import datetime
import pandas as pd

# Configuration
DATA_DIR = "Rotmod_LTG"
RESULTS_DIR = "final_sparc_results"
PARAMS_FILE = "best_parameters.json"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Galaxy parameter bounds for optimization
GALAXY_BOUNDS = {
    'ML_disk': (0.3, 1.0),
    'ML_bulge': (0.3, 0.9),
    'gas_factor': (1.25, 1.40),
    'h_scale': (100, 600),
}


def load_galaxy_from_file(filepath):
    """Load galaxy data from rotmod file"""
    name = os.path.basename(filepath).replace('_rotmod.dat', '')
    
    try:
        data = np.loadtxt(filepath, skiprows=1)
        
        R_kpc = data[:, 0]
        v_obs = data[:, 1]
        
        if len(R_kpc) < 5:
            return None
            
        v_err = np.maximum(0.03 * v_obs, 2.0)
        
        if data.shape[1] >= 7:
            sigma_gas = data[:, 5]
            sigma_disk = data[:, 6]
        else:
            sigma_gas = 10 * np.exp(-R_kpc / 2)
            sigma_disk = 100 * np.exp(-R_kpc / 3)
        
        sigma_bulge = None
        if data.shape[1] >= 8 and np.any(data[:, 7] > 0):
            sigma_bulge = data[:, 7]
        
        return GalaxyData(
            name=name,
            R_kpc=R_kpc,
            v_obs=v_obs,
            v_err=v_err,
            sigma_gas=sigma_gas,
            sigma_disk=sigma_disk,
            sigma_bulge=sigma_bulge
        )
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def optimize_galaxy_params(galaxy, solver, max_iter=100):
    """Optimize per-galaxy parameters"""
    
    def objective(x):
        params = GalaxyParameters(
            ML_disk=x[0],
            ML_bulge=x[1] if galaxy.sigma_bulge is not None else 0.7,
            gas_factor=x[2],
            h_scale=x[3]
        )
        
        try:
            result = solver.solve_galaxy(galaxy, params)
            return result['chi2_reduced']
        except:
            return 1e6
    
    # Setup based on whether galaxy has bulge
    if galaxy.sigma_bulge is not None:
        x0 = [0.5, 0.7, 1.33, 300]
        bounds = [GALAXY_BOUNDS['ML_disk'], GALAXY_BOUNDS['ML_bulge'], 
                 GALAXY_BOUNDS['gas_factor'], GALAXY_BOUNDS['h_scale']]
    else:
        x0 = [0.5, 1.33, 300]
        bounds = [GALAXY_BOUNDS['ML_disk'], GALAXY_BOUNDS['gas_factor'], 
                 GALAXY_BOUNDS['h_scale']]
        objective_no_bulge = lambda x: objective([x[0], 0.7, x[1], x[2]])
        objective = objective_no_bulge
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', 
                     bounds=bounds, options={'maxiter': max_iter})
    
    # Extract optimized parameters
    if galaxy.sigma_bulge is not None:
        opt_params = GalaxyParameters(
            ML_disk=result.x[0],
            ML_bulge=result.x[1],
            gas_factor=result.x[2],
            h_scale=result.x[3]
        )
    else:
        opt_params = GalaxyParameters(
            ML_disk=result.x[0],
            ML_bulge=0.7,
            gas_factor=result.x[1],
            h_scale=result.x[2]
        )
    
    return opt_params, result.fun


def analyze_galaxy(solver, galaxy, optimize_params=True, save_plot=True):
    """Analyze single galaxy with optional parameter optimization"""
    
    try:
        if optimize_params:
            # Optimize per-galaxy parameters
            opt_params, _ = optimize_galaxy_params(galaxy, solver)
            result = solver.solve_galaxy(galaxy, opt_params)
        else:
            # Use default parameters
            opt_params = GalaxyParameters()
            result = solver.solve_galaxy(galaxy)
        
        if save_plot and result['chi2_reduced'] < 100:  # Only plot reasonable fits
            # Create individual plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Rotation curve
            ax1.errorbar(galaxy.R_kpc, galaxy.v_obs, yerr=galaxy.v_err,
                        fmt='ko', alpha=0.7, markersize=4, label='Observed', zorder=3)
            ax1.plot(galaxy.R_kpc, result['v_model'], 'r-', linewidth=2.5,
                    label=f"RS Model (χ²/N={result['chi2_reduced']:.2f})", zorder=2)
            ax1.plot(galaxy.R_kpc, result['v_newton'], 'b--', alpha=0.7,
                    label='Newtonian', zorder=1)
            ax1.set_xlabel('Radius (kpc)', fontsize=12)
            ax1.set_ylabel('Velocity (km/s)', fontsize=12)
            ax1.set_title(f'{galaxy.name} Rotation Curve', fontsize=14)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, max(galaxy.v_obs) * 1.2)
            
            # Residuals with chi2 heat map
            chi2_per_point = (result['residuals'] / galaxy.v_err)**2
            scatter = ax2.scatter(galaxy.R_kpc, result['residuals'], 
                                c=chi2_per_point, s=60, cmap='YlOrRd', 
                                vmin=0, vmax=9, edgecolors='black', linewidth=0.5)
            ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax2.fill_between(galaxy.R_kpc, -galaxy.v_err, galaxy.v_err,
                            alpha=0.2, color='gray', label='1σ errors')
            ax2.fill_between(galaxy.R_kpc, -2*galaxy.v_err, 2*galaxy.v_err,
                            alpha=0.1, color='gray', label='2σ errors')
            ax2.set_xlabel('Radius (kpc)', fontsize=12)
            ax2.set_ylabel('Residuals (km/s)', fontsize=12)
            ax2.set_title('Model Residuals', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=11)
            
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('χ² per point', fontsize=11)
            
            # Add parameter info
            param_text = f"M/L disk: {opt_params.ML_disk:.2f}\n"
            param_text += f"Gas factor: {opt_params.gas_factor:.2f}\n"
            param_text += f"h scale: {opt_params.h_scale:.0f} pc"
            if galaxy.sigma_bulge is not None:
                param_text += f"\nM/L bulge: {opt_params.ML_bulge:.2f}"
            
            ax2.text(0.02, 0.98, param_text, transform=ax2.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'{galaxy.name}_fit.png'), 
                       dpi=200, bbox_inches='tight')
            plt.close()
        
        return {
            'name': galaxy.name,
            'chi2': result['chi2'],
            'chi2_reduced': result['chi2_reduced'],
            'n_points': len(galaxy.v_obs),
            'max_radius_kpc': max(galaxy.R_kpc),
            'max_velocity': max(galaxy.v_obs),
            'rms_residual': np.sqrt(np.mean(result['residuals']**2)),
            'ML_disk': opt_params.ML_disk,
            'ML_bulge': opt_params.ML_bulge,
            'gas_factor': opt_params.gas_factor,
            'h_scale': opt_params.h_scale,
            'has_bulge': galaxy.sigma_bulge is not None,
            'success': True
        }
        
    except Exception as e:
        print(f"Error analyzing {galaxy.name}: {e}")
        return {
            'name': galaxy.name,
            'chi2': np.nan,
            'chi2_reduced': np.nan,
            'n_points': len(galaxy.v_obs),
            'max_radius_kpc': max(galaxy.R_kpc),
            'max_velocity': max(galaxy.v_obs),
            'rms_residual': np.nan,
            'success': False
        }


def create_publication_plots(results_df):
    """Create publication-quality summary plots"""
    
    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    
    # Filter successful fits
    successful = results_df[results_df['success']]
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Chi-squared distribution
    ax1 = plt.subplot(2, 3, 1)
    bins = np.logspace(np.log10(0.5), np.log10(100), 40)
    ax1.hist(successful['chi2_reduced'], bins=bins, alpha=0.7, 
             edgecolor='black', color='steelblue')
    ax1.axvline(1, color='green', linestyle='--', linewidth=2, label='Perfect fit')
    ax1.axvline(successful['chi2_reduced'].median(), color='red', 
               linestyle='--', linewidth=2,
               label=f"Median = {successful['chi2_reduced'].median():.2f}")
    ax1.set_xscale('log')
    ax1.set_xlabel('χ²/N')
    ax1.set_ylabel('Number of Galaxies')
    ax1.set_title('A. Goodness of Fit Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative distribution
    ax2 = plt.subplot(2, 3, 2)
    sorted_chi2 = np.sort(successful['chi2_reduced'])
    cum_frac = np.arange(1, len(sorted_chi2)+1) / len(sorted_chi2)
    ax2.plot(sorted_chi2, cum_frac, 'b-', linewidth=2)
    ax2.axvline(1, color='green', linestyle='--', alpha=0.5)
    ax2.axvline(2, color='orange', linestyle='--', alpha=0.5)
    ax2.axvline(5, color='red', linestyle='--', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('χ²/N')
    ax2.set_ylabel('Cumulative Fraction')
    ax2.set_title('B. Cumulative Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    frac_below_2 = (successful['chi2_reduced'] < 2).sum() / len(successful)
    frac_below_5 = (successful['chi2_reduced'] < 5).sum() / len(successful)
    ax2.text(0.95, 0.05, f"{100*frac_below_2:.1f}% < 2\n{100*frac_below_5:.1f}% < 5",
            transform=ax2.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Parameter distributions
    ax3 = plt.subplot(2, 3, 3)
    param_data = [
        successful['ML_disk'],
        successful[successful['has_bulge']]['ML_bulge'],
        successful['gas_factor'],
        successful['h_scale']/100  # Convert to 100s of pc
    ]
    labels = ['M/L disk', 'M/L bulge', 'Gas factor', 'h/100 pc']
    
    bp = ax3.boxplot(param_data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('C. Optimized Parameters')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Chi2 vs galaxy properties
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(successful['max_radius_kpc'], successful['chi2_reduced'],
                         c=successful['max_velocity'], s=40, alpha=0.6, 
                         cmap='viridis', vmin=0, vmax=300)
    ax4.set_xlabel('Maximum Radius (kpc)')
    ax4.set_ylabel('χ²/N')
    ax4.set_yscale('log')
    ax4.set_title('D. Fit Quality vs Galaxy Size')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Max V (km/s)')
    
    # 5. RMS residuals
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(successful['rms_residual'], bins=30, alpha=0.7, 
             edgecolor='black', color='coral')
    ax5.axvline(successful['rms_residual'].median(), color='red', 
               linestyle='--', linewidth=2,
               label=f"Median = {successful['rms_residual'].median():.1f} km/s")
    ax5.set_xlabel('RMS Residual (km/s)')
    ax5.set_ylabel('Number of Galaxies')
    ax5.set_title('E. Residual Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Framework summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Load best parameters
    with open(PARAMS_FILE, 'r') as f:
        params_data = json.load(f)
    global_params = params_data['global_parameters']
    
    summary_text = "F. Recognition Science Framework\n\n"
    summary_text += "Zero Free Parameters:\n"
    summary_text += f"  φ = {(1+np.sqrt(5))/2:.6f}\n"
    summary_text += f"  β₀ = -(φ-1)/φ⁵ = -0.055728\n\n"
    summary_text += "Optimized Scales:\n"
    summary_text += f"  λ_eff = {global_params['lambda_eff']*1e6:.1f} μm\n"
    summary_text += f"  β = {global_params['beta_scale']:.3f} × β₀\n"
    summary_text += f"  μ = {global_params['mu_scale']:.3f} × μ₀\n"
    summary_text += f"  λ_c = {global_params['coupling_scale']:.3f} × λ_c₀\n\n"
    summary_text += "Statistics:\n"
    summary_text += f"  Galaxies: {len(successful)}/{len(results_df)}\n"
    summary_text += f"  Median χ²/N: {successful['chi2_reduced'].median():.2f}\n"
    summary_text += f"  χ²/N < 2: {100*frac_below_2:.1f}%\n"
    summary_text += f"  χ²/N < 5: {100*frac_below_5:.1f}%"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            verticalalignment='top', fontsize=11, family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sparc_publication_summary.png'), 
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'sparc_publication_summary.pdf'), 
               bbox_inches='tight')
    plt.show()


def main():
    """Main analysis routine"""
    print("Final SPARC Analysis with Bayesian Optimized Parameters")
    print("="*60)
    
    # Load optimized parameters
    print("\nLoading optimized parameters...")
    with open(PARAMS_FILE, 'r') as f:
        params_data = json.load(f)
    
    global_params = params_data['global_parameters']
    
    print("\nGlobal Parameters:")
    print(f"  λ_eff = {global_params['lambda_eff']*1e6:.1f} μm")
    print(f"  β_scale = {global_params['beta_scale']:.3f}")
    print(f"  μ_scale = {global_params['mu_scale']:.3f}")
    print(f"  coupling_scale = {global_params['coupling_scale']:.3f}")
    
    # Initialize solver
    solver = EnhancedGravitySolver(**global_params)
    
    # Load all galaxies
    print(f"\nLoading galaxies from {DATA_DIR}...")
    galaxy_files = glob.glob(os.path.join(DATA_DIR, "*_rotmod.dat"))
    
    galaxies = []
    for filepath in galaxy_files:
        galaxy = load_galaxy_from_file(filepath)
        if galaxy is not None:
            galaxies.append(galaxy)
    
    print(f"Loaded {len(galaxies)} galaxies successfully")
    
    # Analyze all galaxies
    print(f"\nAnalyzing {len(galaxies)} galaxies with per-galaxy optimization...")
    results = []
    
    start_time = datetime.now()
    for i, galaxy in enumerate(galaxies):
        if i % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (i+1) / elapsed if elapsed > 0 else 0
            eta = (len(galaxies) - i) / rate if rate > 0 else 0
            print(f"  Progress: {i}/{len(galaxies)} galaxies ({100*i/len(galaxies):.1f}%) - ETA: {eta:.0f}s")
        
        # Analyze with optimization and plots for best fits
        result = analyze_galaxy(solver, galaxy, optimize_params=True, 
                              save_plot=(i < 100))  # Plot first 100
        results.append(result)
    
    end_time = datetime.now()
    print(f"\nAnalysis completed in {(end_time - start_time).total_seconds():.1f} seconds")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(RESULTS_DIR, 'final_sparc_results.csv'), index=False)
    
    # Save summary
    successful = results_df[results_df['success']]
    summary = {
        'total_galaxies': len(results_df),
        'successful_fits': len(successful),
        'mean_chi2_reduced': float(successful['chi2_reduced'].mean()),
        'median_chi2_reduced': float(successful['chi2_reduced'].median()),
        'std_chi2_reduced': float(successful['chi2_reduced'].std()),
        'frac_below_1': float((successful['chi2_reduced'] < 1).sum() / len(successful)),
        'frac_below_2': float((successful['chi2_reduced'] < 2).sum() / len(successful)),
        'frac_below_5': float((successful['chi2_reduced'] < 5).sum() / len(successful)),
        'best_fits': successful.nsmallest(10, 'chi2_reduced')[['name', 'chi2_reduced']].to_dict('records'),
        'worst_fits': successful.nlargest(10, 'chi2_reduced')[['name', 'chi2_reduced']].to_dict('records'),
        'global_parameters': global_params,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(RESULTS_DIR, 'final_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create publication plots
    print("\nCreating publication-quality plots...")
    create_publication_plots(results_df)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL ANALYSIS SUMMARY:")
    print("="*60)
    print(f"Total galaxies analyzed: {summary['total_galaxies']}")
    print(f"Successful fits: {summary['successful_fits']} ({100*summary['successful_fits']/summary['total_galaxies']:.1f}%)")
    print(f"Mean χ²/N: {summary['mean_chi2_reduced']:.2f} ± {summary['std_chi2_reduced']:.2f}")
    print(f"Median χ²/N: {summary['median_chi2_reduced']:.2f}")
    print(f"Galaxies with χ²/N < 1: {100*summary['frac_below_1']:.1f}%")
    print(f"Galaxies with χ²/N < 2: {100*summary['frac_below_2']:.1f}%")
    print(f"Galaxies with χ²/N < 5: {100*summary['frac_below_5']:.1f}%")
    
    print(f"\nResults saved to {RESULTS_DIR}/")
    print("  - final_sparc_results.csv: Detailed results")
    print("  - final_analysis_summary.json: Summary statistics")
    print("  - sparc_publication_summary.png/pdf: Publication figures")


if __name__ == "__main__":
    main() 