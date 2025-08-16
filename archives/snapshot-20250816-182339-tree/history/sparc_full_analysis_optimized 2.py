#!/usr/bin/env python3
"""
Full SPARC Analysis with Optimized Recognition Science Gravity
==============================================================
Analyzes all 175 SPARC galaxies using optimized parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
from rs_gravity_tunable import TunableGravitySolver, GalaxyData
from datetime import datetime
import pandas as pd

# Configuration
DATA_DIR = "Rotmod_LTG"
RESULTS_DIR = "sparc_analysis_results"
OPTIMIZED_PARAMS_FILE = "optimization_results.json"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_optimized_parameters():
    """Load optimized parameters from previous run"""
    with open(OPTIMIZED_PARAMS_FILE, 'r') as f:
        data = json.load(f)
    return data['best_parameters']

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
        
        # Check for bulge component
        sigma_bulge = None
        if data.shape[1] >= 8:
            sigma_bulge = data[:, 7] * 0.7  # Bulge M/L ratio
        
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

def analyze_galaxy(solver, galaxy, save_plot=True):
    """Analyze single galaxy and optionally save plot"""
    try:
        result = solver.solve_galaxy(galaxy)
        
        if save_plot:
            # Create individual plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Rotation curve
            ax1.errorbar(galaxy.R_kpc, galaxy.v_obs, yerr=galaxy.v_err,
                        fmt='ko', alpha=0.7, markersize=4, label='Observed')
            ax1.plot(galaxy.R_kpc, result['v_model'], 'r-', linewidth=2,
                    label=f"RS Model (χ²/N={result['chi2_reduced']:.2f})")
            ax1.plot(galaxy.R_kpc, result['v_newton'], 'b--', alpha=0.7,
                    label='Newtonian')
            ax1.set_xlabel('Radius (kpc)')
            ax1.set_ylabel('Velocity (km/s)')
            ax1.set_title(f'{galaxy.name} Rotation Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, max(galaxy.v_obs) * 1.2)
            
            # Residuals
            ax2.plot(galaxy.R_kpc, result['residuals'], 'go-', alpha=0.7)
            ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax2.fill_between(galaxy.R_kpc, -galaxy.v_err, galaxy.v_err,
                            alpha=0.2, color='gray', label='1σ errors')
            ax2.set_xlabel('Radius (kpc)')
            ax2.set_ylabel('Residuals (km/s)')
            ax2.set_title('Model Residuals')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'{galaxy.name}_fit.png'), dpi=150)
            plt.close()
        
        return {
            'name': galaxy.name,
            'chi2': result['chi2'],
            'chi2_reduced': result['chi2_reduced'],
            'n_points': len(galaxy.v_obs),
            'max_radius_kpc': max(galaxy.R_kpc),
            'max_velocity': max(galaxy.v_obs),
            'rms_residual': np.sqrt(np.mean(result['residuals']**2)),
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

def create_summary_plots(results_df):
    """Create summary plots of all results"""
    
    # Filter successful fits
    successful = results_df[results_df['success']]
    
    # 1. Chi-squared distribution
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram of chi2/N
    ax1.hist(successful['chi2_reduced'], bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(np.median(successful['chi2_reduced']), color='r', linestyle='--', 
               label=f"Median = {np.median(successful['chi2_reduced']):.2f}")
    ax1.set_xlabel('χ²/N')
    ax1.set_ylabel('Number of Galaxies')
    ax1.set_title('Distribution of Reduced Chi-Squared')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chi2 vs max radius
    ax2.scatter(successful['max_radius_kpc'], successful['chi2_reduced'], 
               alpha=0.6, c=successful['max_velocity'], cmap='viridis')
    ax2.set_xlabel('Maximum Radius (kpc)')
    ax2.set_ylabel('χ²/N')
    ax2.set_title('Fit Quality vs Galaxy Size')
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Max Velocity (km/s)')
    
    # RMS residuals distribution
    ax3.hist(successful['rms_residual'], bins=30, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('RMS Residual (km/s)')
    ax3.set_ylabel('Number of Galaxies')
    ax3.set_title('Distribution of RMS Residuals')
    ax3.grid(True, alpha=0.3)
    
    # Success rate by galaxy type
    ax4.text(0.1, 0.9, f"Total galaxies analyzed: {len(results_df)}", 
            transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.8, f"Successful fits: {len(successful)} ({100*len(successful)/len(results_df):.1f}%)", 
            transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.7, f"Mean χ²/N: {successful['chi2_reduced'].mean():.2f}", 
            transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.6, f"Median χ²/N: {successful['chi2_reduced'].median():.2f}", 
            transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.5, f"Best fit: {successful.loc[successful['chi2_reduced'].idxmin(), 'name']} (χ²/N = {successful['chi2_reduced'].min():.2f})", 
            transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.4, f"Worst fit: {successful.loc[successful['chi2_reduced'].idxmax(), 'name']} (χ²/N = {successful['chi2_reduced'].max():.2f})", 
            transform=ax4.transAxes, fontsize=12)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sparc_analysis_summary.png'), dpi=300)
    plt.show()
    
    # 2. Best and worst fits comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get best and worst galaxies
    best_idx = successful['chi2_reduced'].nsmallest(3).index
    worst_idx = successful['chi2_reduced'].nlargest(3).index
    
    for i, (idx, ax) in enumerate(zip(best_idx, axes[0])):
        name = results_df.loc[idx, 'name']
        img_path = os.path.join(RESULTS_DIR, f'{name}_fit.png')
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Best {i+1}: {name}')
    
    for i, (idx, ax) in enumerate(zip(worst_idx, axes[1])):
        name = results_df.loc[idx, 'name']
        img_path = os.path.join(RESULTS_DIR, f'{name}_fit.png')
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Worst {i+1}: {name}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'best_worst_fits.png'), dpi=300)
    plt.show()

def main():
    """Main analysis routine"""
    print("Recognition Science Full SPARC Analysis")
    print("="*60)
    
    # Load optimized parameters
    print("\nLoading optimized parameters...")
    params = load_optimized_parameters()
    
    print("\nOptimized Parameters:")
    print(f"  λ_eff = {params['lambda_eff']*1e6:.1f} μm")
    print(f"  h_scale = {params['h_scale']/3.086e16:.0f} pc")
    print(f"  β_scale = {params['beta_scale']:.2f}")
    print(f"  μ_scale = {params['mu_scale']:.2f}")
    print(f"  coupling_scale = {params['coupling_scale']:.2f}")
    
    # Initialize solver
    solver = TunableGravitySolver(**params)
    
    # Load all galaxies
    print(f"\nLoading galaxies from {DATA_DIR}...")
    galaxy_files = glob.glob(os.path.join(DATA_DIR, "*_rotmod.dat"))
    
    galaxies = []
    for filepath in galaxy_files:
        galaxy = load_galaxy_from_file(filepath)
        if galaxy is not None:
            galaxies.append(galaxy)
    
    print(f"Loaded {len(galaxies)} galaxies successfully")
    
    if len(galaxies) == 0:
        print("No galaxies loaded! Check data directory.")
        return
    
    # Analyze all galaxies
    print(f"\nAnalyzing {len(galaxies)} galaxies...")
    results = []
    
    start_time = datetime.now()
    for i, galaxy in enumerate(galaxies):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(galaxies)} galaxies analyzed...")
        
        # Analyze with plots for first 50 galaxies only
        result = analyze_galaxy(solver, galaxy, save_plot=(i < 50))
        results.append(result)
    
    end_time = datetime.now()
    print(f"\nAnalysis completed in {(end_time - start_time).total_seconds():.1f} seconds")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(RESULTS_DIR, 'sparc_analysis_results.csv'), index=False)
    
    # Save summary statistics
    summary = {
        'total_galaxies': len(results_df),
        'successful_fits': len(results_df[results_df['success']]),
        'mean_chi2_reduced': results_df[results_df['success']]['chi2_reduced'].mean(),
        'median_chi2_reduced': results_df[results_df['success']]['chi2_reduced'].median(),
        'std_chi2_reduced': results_df[results_df['success']]['chi2_reduced'].std(),
        'best_fit': {
            'name': results_df.loc[results_df['chi2_reduced'].idxmin(), 'name'],
            'chi2_reduced': results_df['chi2_reduced'].min()
        },
        'worst_fit': {
            'name': results_df.loc[results_df['chi2_reduced'].idxmax(), 'name'],
            'chi2_reduced': results_df['chi2_reduced'].max()
        },
        'parameters_used': params,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(RESULTS_DIR, 'sparc_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create summary plots
    print("\nCreating summary plots...")
    create_summary_plots(results_df)
    
    # Print final summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY:")
    print("="*60)
    print(f"Total galaxies analyzed: {summary['total_galaxies']}")
    print(f"Successful fits: {summary['successful_fits']} ({100*summary['successful_fits']/summary['total_galaxies']:.1f}%)")
    print(f"Mean χ²/N: {summary['mean_chi2_reduced']:.2f} ± {summary['std_chi2_reduced']:.2f}")
    print(f"Median χ²/N: {summary['median_chi2_reduced']:.2f}")
    print(f"Best fit: {summary['best_fit']['name']} (χ²/N = {summary['best_fit']['chi2_reduced']:.2f})")
    print(f"Worst fit: {summary['worst_fit']['name']} (χ²/N = {summary['worst_fit']['chi2_reduced']:.2f})")
    
    print(f"\nResults saved to {RESULTS_DIR}/")
    print("  - sparc_analysis_results.csv: Detailed results for each galaxy")
    print("  - sparc_analysis_summary.json: Summary statistics")
    print("  - sparc_analysis_summary.png: Statistical plots")
    print("  - Individual galaxy plots: First 50 galaxies")

if __name__ == "__main__":
    main() 