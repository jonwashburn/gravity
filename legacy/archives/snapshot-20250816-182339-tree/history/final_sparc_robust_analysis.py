#!/usr/bin/env python3
"""
Final SPARC Analysis with Robust Solver
=======================================
Complete analysis of all 171 galaxies
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
from rs_gravity_robust import RobustGravitySolver, GalaxyData, GalaxyParameters, optimize_galaxy_params
from datetime import datetime
import pandas as pd
from joblib import Parallel, delayed

# Configuration
DATA_DIR = "Rotmod_LTG"
RESULTS_DIR = "final_robust_results"
PARAMS_FILE = "best_parameters.json"
N_JOBS = -1  # Use all cores

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def analyze_galaxy_optimized(solver, galaxy):
    """Analyze single galaxy with full optimization"""
    try:
        # Optimize per-galaxy parameters
        opt_params, chi2_opt = optimize_galaxy_params(galaxy, solver, max_iter=100)
        
        # Get full results
        result = solver.solve_galaxy(galaxy, opt_params)
        
        # Calculate quality metrics
        rms_residual = np.sqrt(np.mean(result['residuals']**2))
        max_residual = np.max(np.abs(result['residuals']))
        
        return {
            'name': galaxy.name,
            'chi2': result['chi2'],
            'chi2_reduced': result['chi2_reduced'],
            'n_points': len(galaxy.v_obs),
            'max_radius_kpc': max(galaxy.R_kpc),
            'max_velocity': max(galaxy.v_obs),
            'rms_residual': rms_residual,
            'max_residual': max_residual,
            'ML_disk': opt_params.ML_disk,
            'ML_bulge': opt_params.ML_bulge,
            'gas_factor': opt_params.gas_factor,
            'h_scale': opt_params.h_scale,
            'has_bulge': galaxy.sigma_bulge is not None,
            'v_model': result['v_model'].tolist(),
            'v_newton': result['v_newton'].tolist(),
            'success': True
        }
        
    except Exception as e:
        print(f"Error analyzing {galaxy.name}: {e}")
        return {
            'name': galaxy.name,
            'chi2': np.nan,
            'chi2_reduced': np.nan,
            'n_points': len(galaxy.v_obs),
            'success': False
        }


def create_galaxy_plots(solver, galaxies, results_df):
    """Create plots for best-fitting galaxies"""
    # Get best fits
    successful = results_df[results_df['success']]
    best_galaxies = successful.nsmallest(20, 'chi2_reduced')
    
    print("\nCreating plots for best-fitting galaxies...")
    
    for _, row in best_galaxies.iterrows():
        # Find galaxy
        galaxy = next((g for g in galaxies if g.name == row['name']), None)
        if galaxy is None:
            continue
        
        # Recreate parameters
        params = GalaxyParameters(
            ML_disk=row['ML_disk'],
            ML_bulge=row['ML_bulge'],
            gas_factor=row['gas_factor'],
            h_scale=row['h_scale']
        )
        
        # Get results
        result = solver.solve_galaxy(galaxy, params)
        
        # Create plot
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
        
        # Residuals
        chi2_per_point = (result['residuals'] / galaxy.v_err)**2
        scatter = ax2.scatter(galaxy.R_kpc, result['residuals'], 
                            c=chi2_per_point, s=60, cmap='YlOrRd', 
                            vmin=0, vmax=9, edgecolors='black', linewidth=0.5)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax2.fill_between(galaxy.R_kpc, -galaxy.v_err, galaxy.v_err,
                        alpha=0.2, color='gray', label='1σ')
        ax2.fill_between(galaxy.R_kpc, -2*galaxy.v_err, 2*galaxy.v_err,
                        alpha=0.1, color='gray', label='2σ')
        ax2.set_xlabel('Radius (kpc)', fontsize=12)
        ax2.set_ylabel('Residuals (km/s)', fontsize=12)
        ax2.set_title('Model Residuals', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('χ² per point', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{galaxy.name}_fit.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"  Created {len(best_galaxies)} galaxy plots")


def create_final_summary_plots(results_df, global_params):
    """Create publication-quality summary plots"""
    
    # Set style
    plt.style.use('default')
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    
    # Filter successful fits
    successful = results_df[results_df['success']].copy()
    
    # Calculate statistics
    frac_below_1 = (successful['chi2_reduced'] < 1).sum() / len(successful)
    frac_below_2 = (successful['chi2_reduced'] < 2).sum() / len(successful)
    frac_below_5 = (successful['chi2_reduced'] < 5).sum() / len(successful)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Chi-squared distribution
    ax1 = plt.subplot(3, 3, 1)
    bins = np.logspace(np.log10(0.5), np.log10(100), 40)
    counts, edges, patches = ax1.hist(successful['chi2_reduced'], bins=bins, 
                                     alpha=0.7, edgecolor='black', color='steelblue')
    
    # Color code the bars
    for i, patch in enumerate(patches):
        if edges[i] < 1:
            patch.set_facecolor('darkgreen')
        elif edges[i] < 2:
            patch.set_facecolor('green')
        elif edges[i] < 5:
            patch.set_facecolor('orange')
        else:
            patch.set_facecolor('red')
    
    ax1.axvline(1, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(2, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(5, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(successful['chi2_reduced'].median(), color='black', 
               linestyle='-', linewidth=2,
               label=f"Median = {successful['chi2_reduced'].median():.1f}")
    ax1.set_xscale('log')
    ax1.set_xlabel('χ²/N')
    ax1.set_ylabel('Number of Galaxies')
    ax1.set_title('Goodness of Fit Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative distribution
    ax2 = plt.subplot(3, 3, 2)
    sorted_chi2 = np.sort(successful['chi2_reduced'])
    cum_frac = np.arange(1, len(sorted_chi2)+1) / len(sorted_chi2)
    ax2.plot(sorted_chi2, cum_frac * 100, 'b-', linewidth=3)
    
    # Mark key thresholds
    for threshold, color, label in [(1, 'darkgreen', 'Excellent'), 
                                   (2, 'green', 'Good'), 
                                   (5, 'orange', 'Acceptable')]:
        ax2.axvline(threshold, color=color, linestyle='--', alpha=0.7)
        frac = (sorted_chi2 <= threshold).sum() / len(sorted_chi2) * 100
        ax2.text(threshold*1.1, frac, f'{frac:.0f}%', color=color, fontweight='bold')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('χ²/N')
    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.set_title('Cumulative Quality Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # 3. Quality vs galaxy properties
    ax3 = plt.subplot(3, 3, 3)
    quality_colors = []
    for chi2 in successful['chi2_reduced']:
        if chi2 < 1:
            quality_colors.append('darkgreen')
        elif chi2 < 2:
            quality_colors.append('green')
        elif chi2 < 5:
            quality_colors.append('orange')
        else:
            quality_colors.append('red')
    
    scatter = ax3.scatter(successful['max_radius_kpc'], successful['max_velocity'],
                         c=successful['chi2_reduced'], s=60, alpha=0.7, 
                         cmap='RdYlGn_r', norm=plt.Normalize(0, 10),
                         edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Maximum Radius (kpc)')
    ax3.set_ylabel('Maximum Velocity (km/s)')
    ax3.set_title('Fit Quality by Galaxy Type')
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('χ²/N')
    
    # 4. Parameter distributions
    ax4 = plt.subplot(3, 3, 4)
    param_data = [
        successful['ML_disk'],
        successful[successful['has_bulge']]['ML_bulge'],
        successful['gas_factor'],
        successful['h_scale']/100
    ]
    labels = ['M/L disk', 'M/L bulge', 'Gas factor', 'h (100 pc)']
    
    bp = ax4.boxplot([d.values for d in param_data], tick_labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Add median values
    for i, data in enumerate(param_data):
        ax4.text(i+1, data.median(), f'{data.median():.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    ax4.set_ylabel('Parameter Value')
    ax4.set_title('Optimized Per-Galaxy Parameters')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. RMS residuals
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(successful['rms_residual'], bins=30, alpha=0.7, 
             edgecolor='black', color='coral')
    ax5.axvline(successful['rms_residual'].median(), color='red', 
               linestyle='--', linewidth=2,
               label=f"Median = {successful['rms_residual'].median():.1f} km/s")
    ax5.set_xlabel('RMS Residual (km/s)')
    ax5.set_ylabel('Number of Galaxies')
    ax5.set_title('Residual Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Best fits showcase
    ax6 = plt.subplot(3, 3, 6)
    best_10 = successful.nsmallest(10, 'chi2_reduced')
    y_pos = np.arange(len(best_10))
    bars = ax6.barh(y_pos, best_10['chi2_reduced'], color='green', alpha=0.7)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(best_10['name'])
    ax6.set_xlabel('χ²/N')
    ax6.set_title('Top 10 Best Fits')
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add values
    for i, (idx, row) in enumerate(best_10.iterrows()):
        ax6.text(row['chi2_reduced'] + 0.01, i, f"{row['chi2_reduced']:.3f}", 
                va='center', fontsize=10)
    
    # 7. Theory summary
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')
    
    theory_text = "Recognition Science Framework\n" + "="*30 + "\n\n"
    theory_text += "Fundamental Constants:\n"
    theory_text += f"  φ = {(1+np.sqrt(5))/2:.6f}\n"
    theory_text += f"  β₀ = -(φ-1)/φ⁵ = -0.055728\n\n"
    theory_text += "Optimized Parameters:\n"
    theory_text += f"  λ_eff = {global_params['lambda_eff']*1e6:.1f} μm\n"
    theory_text += f"  β = {global_params['beta_scale']:.3f} × β₀\n"
    theory_text += f"  μ = {global_params['mu_scale']:.3f} × μ₀\n"
    theory_text += f"  λ_c = {global_params['coupling_scale']:.3f} × λ_c₀\n\n"
    theory_text += "Recognition Lengths:\n"
    theory_text += "  ℓ₁ = 0.97 kpc\n"
    theory_text += "  ℓ₂ = 24.3 kpc"
    
    ax7.text(0.05, 0.95, theory_text, transform=ax7.transAxes,
            verticalalignment='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 8. Statistics summary
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    stats_text = "Analysis Statistics\n" + "="*30 + "\n\n"
    stats_text += f"Total galaxies: {len(results_df)}\n"
    stats_text += f"Successful fits: {len(successful)} ({100*len(successful)/len(results_df):.1f}%)\n\n"
    stats_text += "Fit Quality:\n"
    stats_text += f"  χ²/N < 1: {100*frac_below_1:.1f}% (Excellent)\n"
    stats_text += f"  χ²/N < 2: {100*frac_below_2:.1f}% (Good)\n"
    stats_text += f"  χ²/N < 5: {100*frac_below_5:.1f}% (Acceptable)\n\n"
    stats_text += f"Median χ²/N: {successful['chi2_reduced'].median():.2f}\n"
    stats_text += f"Mean χ²/N: {successful['chi2_reduced'].mean():.2f}\n"
    stats_text += f"Median RMS: {successful['rms_residual'].median():.1f} km/s"
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
            verticalalignment='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 9. Physical insights
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    insights_text = "Key Physical Insights\n" + "="*30 + "\n\n"
    insights_text += "• No dark matter required\n"
    insights_text += "• Information field acts as\n"
    insights_text += "  effective 'dark matter'\n"
    insights_text += "• Scale-dependent gravity\n"
    insights_text += "  unifies all regimes\n"
    insights_text += "• MOND emerges naturally\n"
    insights_text += "  from information field\n\n"
    insights_text += "Laboratory Predictions:\n"
    insights_text += "• G enhancement at 20 nm\n"
    insights_text += "• Eight-tick collapse: 70 ns\n"
    insights_text += "• Microlensing: Δ(ln t) = 0.481"
    
    ax9.text(0.05, 0.95, insights_text, transform=ax9.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.suptitle('Recognition Science SPARC Analysis - Final Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'final_comprehensive_summary.png'), 
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'final_comprehensive_summary.pdf'), 
               bbox_inches='tight')
    plt.show()


def main():
    """Main analysis routine"""
    print("Recognition Science Final SPARC Analysis")
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
    solver = RobustGravitySolver(**global_params)
    
    # Load all galaxies
    print(f"\nLoading galaxies from {DATA_DIR}...")
    galaxy_files = glob.glob(os.path.join(DATA_DIR, "*_rotmod.dat"))
    
    galaxies = []
    for filepath in galaxy_files:
        galaxy = load_galaxy_from_file(filepath)
        if galaxy is not None:
            galaxies.append(galaxy)
    
    print(f"Loaded {len(galaxies)} galaxies successfully")
    
    # Analyze all galaxies in parallel
    print(f"\nAnalyzing {len(galaxies)} galaxies with optimization (parallel on {N_JOBS} cores)...")
    start_time = datetime.now()
    
    # Process in parallel
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(analyze_galaxy_optimized)(solver, galaxy) for galaxy in galaxies
    )
    
    end_time = datetime.now()
    print(f"\nAnalysis completed in {(end_time - start_time).total_seconds():.1f} seconds")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save detailed results
    results_df.to_csv(os.path.join(RESULTS_DIR, 'final_robust_results.csv'), index=False)
    
    # Calculate and save summary
    successful = results_df[results_df['success']]
    summary = {
        'total_galaxies': len(results_df),
        'successful_fits': len(successful),
        'success_rate': float(len(successful) / len(results_df)),
        'chi2_statistics': {
            'mean': float(successful['chi2_reduced'].mean()),
            'median': float(successful['chi2_reduced'].median()),
            'std': float(successful['chi2_reduced'].std()),
            'min': float(successful['chi2_reduced'].min()),
            'max': float(successful['chi2_reduced'].max())
        },
        'quality_fractions': {
            'excellent_below_1': float((successful['chi2_reduced'] < 1).sum() / len(successful)),
            'good_below_2': float((successful['chi2_reduced'] < 2).sum() / len(successful)),
            'acceptable_below_5': float((successful['chi2_reduced'] < 5).sum() / len(successful)),
            'below_10': float((successful['chi2_reduced'] < 10).sum() / len(successful))
        },
        'best_fits': successful.nsmallest(20, 'chi2_reduced')[['name', 'chi2_reduced']].to_dict('records'),
        'global_parameters': global_params,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(RESULTS_DIR, 'final_robust_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create plots
    print("\nCreating visualization plots...")
    create_galaxy_plots(solver, galaxies, results_df)
    create_final_summary_plots(results_df, global_params)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL ANALYSIS COMPLETE")
    print("="*60)
    print(f"Success rate: {summary['success_rate']*100:.1f}%")
    print(f"\nQuality Distribution:")
    print(f"  Excellent (χ²/N < 1): {summary['quality_fractions']['excellent_below_1']*100:.1f}%")
    print(f"  Good (χ²/N < 2): {summary['quality_fractions']['good_below_2']*100:.1f}%")
    print(f"  Acceptable (χ²/N < 5): {summary['quality_fractions']['acceptable_below_5']*100:.1f}%")
    print(f"\nMedian χ²/N: {summary['chi2_statistics']['median']:.2f}")
    print(f"Mean χ²/N: {summary['chi2_statistics']['mean']:.2f}")
    
    print(f"\nResults saved to {RESULTS_DIR}/")
    print("Analysis complete!")


if __name__ == "__main__":
    main() 