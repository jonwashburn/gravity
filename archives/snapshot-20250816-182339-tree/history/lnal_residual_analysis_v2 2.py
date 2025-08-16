#!/usr/bin/env python3
"""
LNAL Residual Analysis V2
Analyzes correlations between model residuals and galaxy properties
Handles list format of results and gets properties from SPARC data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle
from typing import Dict, List, Tuple

# Import parser
from parse_sparc_mrt import parse_sparc_mrt

def load_and_process_results():
    """
    Load results and SPARC data, combine them
    """
    # Load model results (list format)
    with open('lnal_sparc_results.pkl', 'rb') as f:
        results_list = pickle.load(f)
    
    # Load SPARC galaxy data
    sparc_data = parse_sparc_mrt()
    
    # Convert sparc_data to dict if it's a list
    if isinstance(sparc_data, list):
        sparc_dict = {g['name']: g for g in sparc_data}
    else:
        sparc_dict = sparc_data
    
    # Combine results with galaxy properties
    galaxy_stats = {}
    
    for result in results_list:
        galaxy_name = result['name']
        
        # Skip if no model velocities
        if 'V_model' not in result or len(result['V_model']) == 0:
            continue
            
        # Get galaxy properties from SPARC data
        if galaxy_name in sparc_dict:
            sparc_galaxy = sparc_dict[galaxy_name]
            
            # Compute residuals
            v_model = np.array(result['V_model'])
            v_obs = np.array(result['V_obs'])
            
            # Handle any zeros or invalid values
            valid = (v_obs > 0) & np.isfinite(v_model) & np.isfinite(v_obs)
            if np.sum(valid) < 5:
                continue
                
            residuals = v_model[valid] / v_obs[valid]
            
            # Extract properties (convert from 10^9 M_sun to M_sun)
            M_star = sparc_galaxy.get('M_star', 0) * 1e9
            M_gas = sparc_galaxy.get('M_HI', 0) * 1e9  # HI mass
            R_disk = sparc_galaxy.get('R_disk', 0)
            distance = sparc_galaxy.get('distance', 10)  # Mpc
            inclination = sparc_galaxy.get('inclination', 45)
            galaxy_type = sparc_galaxy.get('type', 'Unknown')
            

            
            # Compute derived properties
            total_mass = M_star + M_gas
            f_gas = M_gas / total_mass if total_mass > 0 else 0
            
            # Store statistics
            galaxy_stats[galaxy_name] = {
                'mean_residual': np.mean(residuals),
                'median_residual': np.median(residuals),
                'std_residual': np.std(residuals),
                'n_points': len(residuals),
                'M_star': M_star,
                'M_gas': M_gas,
                'R_disk': R_disk,
                'quality': result.get('quality', 2),
                'f_gas': f_gas,
                'total_mass': total_mass,
                'log_M_star': np.log10(M_star) if M_star > 0 else 0,
                'log_total_mass': np.log10(total_mass) if total_mass > 0 else 0,
                'residuals': residuals,
                'galaxy_type': galaxy_type,
                'inclination': inclination
            }
    
    return galaxy_stats

def analyze_correlations(galaxy_stats: Dict) -> None:
    """
    Analyze correlations between residuals and galaxy properties
    """
    # Prepare arrays for analysis
    residuals = []
    f_gas = []
    log_M_star = []
    log_total_mass = []
    quality = []
    galaxy_names = []
    
    for galaxy, stats in galaxy_stats.items():
        if stats['n_points'] >= 5:
            residuals.append(stats['median_residual'])
            f_gas.append(stats['f_gas'])
            log_M_star.append(stats['log_M_star'])
            log_total_mass.append(stats['log_total_mass'])
            quality.append(stats['quality'])
            galaxy_names.append(galaxy)
    
    # Convert to arrays
    residuals = np.array(residuals)
    f_gas = np.array(f_gas)
    log_M_star = np.array(log_M_star)
    log_total_mass = np.array(log_total_mass)
    quality = np.array(quality)
    
    print(f"\n=== RESIDUAL ANALYSIS RESULTS ===")
    print(f"Analyzed {len(residuals)} galaxies")
    print(f"Mean V_model/V_obs: {np.mean(residuals):.3f} ± {np.std(residuals):.3f}")
    print(f"Median V_model/V_obs: {np.median(residuals):.3f}")
    
    # Correlation analysis
    print("\n=== CORRELATIONS ===")
    
    # Gas fraction
    mask = np.isfinite(f_gas) & np.isfinite(residuals)
    if np.sum(mask) > 10:
        r_gas, p_gas = stats.pearsonr(f_gas[mask], residuals[mask])
        print(f"\nGas fraction:")
        print(f"  Pearson r = {r_gas:.3f}, p = {p_gas:.3e}")
        
    # Stellar mass
    mask = (log_M_star > 0) & np.isfinite(log_M_star) & np.isfinite(residuals)
    if np.sum(mask) > 10:
        r_mass, p_mass = stats.pearsonr(log_M_star[mask], residuals[mask])
        print(f"\nStellar mass (log M_*):")
        print(f"  Pearson r = {r_mass:.3f}, p = {p_mass:.3e}")
    
    # Total mass
    mask = (log_total_mass > 0) & np.isfinite(log_total_mass) & np.isfinite(residuals)
    if np.sum(mask) > 10:
        r_total, p_total = stats.pearsonr(log_total_mass[mask], residuals[mask])
        print(f"\nTotal mass (log M_total):")
        print(f"  Pearson r = {r_total:.3f}, p = {p_total:.3e}")
    
    # Quality analysis
    print(f"\nBy quality flag:")
    for q in sorted(set(quality)):
        mask = quality == q
        if np.sum(mask) > 0:
            print(f"  Q={q}: mean = {np.mean(residuals[mask]):.3f}, n = {np.sum(mask)}")
    
    # Find outliers
    print(f"\nOutliers (|residual - 1| > 0.5):")
    outlier_mask = np.abs(residuals - 1.0) > 0.5
    for i, name in enumerate(galaxy_names):
        if outlier_mask[i]:
            print(f"  {name}: {residuals[i]:.3f}")
    
    # Create diagnostic plots
    create_diagnostic_plots(residuals, f_gas, log_M_star, log_total_mass, quality)

def create_diagnostic_plots(residuals, f_gas, log_M_star, log_total_mass, quality):
    """
    Create diagnostic plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gas fraction
    ax = axes[0, 0]
    mask = np.isfinite(f_gas) & np.isfinite(residuals)
    sc = ax.scatter(f_gas[mask], residuals[mask], alpha=0.6, c=quality[mask], 
                    cmap='viridis', s=50)
    ax.set_xlabel('Gas fraction')
    ax.set_ylabel('V_model / V_obs')
    ax.set_title('Residual vs Gas Fraction')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Perfect')
    ax.axhline(0.939, color='black', linestyle='--', alpha=0.5, label='Mean')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.2, 2.0)
    
    # Stellar mass
    ax = axes[0, 1]
    mask = (log_M_star > 0) & np.isfinite(log_M_star) & np.isfinite(residuals)
    ax.scatter(log_M_star[mask], residuals[mask], alpha=0.6, c=quality[mask], 
               cmap='viridis', s=50)
    ax.set_xlabel('log(M_* / M_sun)')
    ax.set_ylabel('V_model / V_obs')
    ax.set_title('Residual vs Stellar Mass')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0.939, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.2, 2.0)
    
    # Total mass
    ax = axes[1, 0]
    mask = (log_total_mass > 0) & np.isfinite(log_total_mass) & np.isfinite(residuals)
    ax.scatter(log_total_mass[mask], residuals[mask], alpha=0.6, c=quality[mask], 
               cmap='viridis', s=50)
    ax.set_xlabel('log(M_total / M_sun)')
    ax.set_ylabel('V_model / V_obs')
    ax.set_title('Residual vs Total Mass')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0.939, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.2, 2.0)
    
    # Histogram
    ax = axes[1, 1]
    ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.8, label='Perfect')
    ax.axvline(np.mean(residuals), color='blue', linestyle='-', 
               alpha=0.8, label=f'Mean={np.mean(residuals):.3f}')
    ax.axvline(np.median(residuals), color='green', linestyle='-', 
               alpha=0.8, label=f'Median={np.median(residuals):.3f}')
    ax.set_xlabel('V_model / V_obs')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(sc, ax=axes, label='Quality flag', pad=0.02)
    
    plt.tight_layout()
    plt.savefig('lnal_residual_correlations_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional plot: residual vs residual scatter
    plt.figure(figsize=(8, 6))
    std_by_galaxy = []
    mean_by_galaxy = []
    
    # Get standard deviation for each galaxy from the raw data
    # This is simplified - would need full galaxy_stats for proper analysis
    
    plt.scatter(residuals, np.ones_like(residuals) * 0.32, alpha=0.5)
    plt.xlabel('Median V_model / V_obs per galaxy')
    plt.ylabel('Scatter (placeholder)')
    plt.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='Perfect')
    plt.axvline(0.939, color='black', linestyle='--', alpha=0.5, label='Mean')
    plt.title('Residual vs Scatter')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lnal_residual_scatter_v2.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """
    Main analysis pipeline
    """
    print("=== LNAL Residual Analysis V2 ===")
    
    try:
        # Load and process data
        print("Loading results and SPARC data...")
        galaxy_stats = load_and_process_results()
        
        print(f"Successfully processed {len(galaxy_stats)} galaxies")
        
        # Analyze correlations
        analyze_correlations(galaxy_stats)
        
        # Save processed statistics
        with open('lnal_residual_stats_v2.pkl', 'wb') as f:
            pickle.dump(galaxy_stats, f)
        
        print("\nAnalysis complete! Outputs:")
        print("  - lnal_residual_correlations_v2.png")
        print("  - lnal_residual_scatter_v2.png")
        print("  - lnal_residual_stats_v2.pkl")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 