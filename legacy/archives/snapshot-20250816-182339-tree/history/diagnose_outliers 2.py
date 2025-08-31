#!/usr/bin/env python3
"""
Diagnostic Tool for High Chi-Squared Galaxies
=============================================
Identifies problematic data points and suggests fixes
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from rs_gravity_tunable_enhanced import EnhancedGravitySolver, GalaxyData, GalaxyParameters
import json

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
            sigma_gas = data[:, 5] * 1.33
            sigma_disk = data[:, 6] * 0.5
        else:
            sigma_gas = 10 * np.exp(-R_kpc / 2)
            sigma_disk = 100 * np.exp(-R_kpc / 3)
        
        sigma_bulge = None
        if data.shape[1] >= 8:
            sigma_bulge = data[:, 7] * 0.7
        
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

def diagnose_galaxy(galaxy, solver, params=None):
    """Diagnose issues with a galaxy fit"""
    result = solver.solve_galaxy(galaxy, params)
    
    # Identify outliers
    z_scores = np.abs(result['residuals']) / galaxy.v_err
    outlier_mask = z_scores > 3
    
    # Check for data issues
    issues = []
    
    # 1. Check for radius centering
    if galaxy.R_kpc[0] > 0.5:
        issues.append({
            'type': 'centering',
            'severity': 'high',
            'description': f'First radius point at {galaxy.R_kpc[0]:.2f} kpc - possible mis-centering'
        })
    
    # 2. Check for gaps in data
    dr = np.diff(galaxy.R_kpc)
    gap_threshold = 2 * np.median(dr)
    gaps = np.where(dr > gap_threshold)[0]
    if len(gaps) > 0:
        for gap_idx in gaps:
            issues.append({
                'type': 'data_gap',
                'severity': 'medium',
                'description': f'Large gap between {galaxy.R_kpc[gap_idx]:.2f} and {galaxy.R_kpc[gap_idx+1]:.2f} kpc'
            })
    
    # 3. Check for velocity spikes
    dv = np.diff(galaxy.v_obs)
    spike_threshold = 3 * np.std(dv)
    spikes = np.where(np.abs(dv) > spike_threshold)[0]
    if len(spikes) > 0:
        for spike_idx in spikes:
            issues.append({
                'type': 'velocity_spike',
                'severity': 'high',
                'description': f'Velocity spike at {galaxy.R_kpc[spike_idx]:.2f} kpc'
            })
    
    # 4. Check surface density issues
    if np.all(galaxy.sigma_gas < 0.1):
        issues.append({
            'type': 'missing_gas',
            'severity': 'medium',
            'description': 'Very low gas surface density - possible missing HI data'
        })
    
    if galaxy.sigma_bulge is None and galaxy.R_kpc[0] < 1.0:
        issues.append({
            'type': 'missing_bulge',
            'severity': 'low',
            'description': 'No bulge component but has inner data points'
        })
    
    # 5. Check for asymmetry (if velocity doesn't rise monotonically)
    v_smooth = np.convolve(galaxy.v_obs, np.ones(3)/3, mode='same')
    if np.any(np.diff(v_smooth[2:-2]) < -10):
        issues.append({
            'type': 'asymmetry',
            'severity': 'medium',
            'description': 'Non-monotonic rotation curve - possible warp or asymmetry'
        })
    
    # Calculate per-point contributions to chi2
    chi2_per_point = (result['residuals'] / galaxy.v_err)**2
    
    return {
        'chi2_reduced': result['chi2_reduced'],
        'outliers': {
            'indices': np.where(outlier_mask)[0].tolist(),
            'radii': galaxy.R_kpc[outlier_mask].tolist(),
            'z_scores': z_scores[outlier_mask].tolist()
        },
        'chi2_per_point': chi2_per_point,
        'issues': issues,
        'max_chi2_point': {
            'index': np.argmax(chi2_per_point),
            'radius': galaxy.R_kpc[np.argmax(chi2_per_point)],
            'contribution': np.max(chi2_per_point)
        }
    }

def create_diagnostic_plot(galaxy, solver, params=None, save_path=None):
    """Create detailed diagnostic plot"""
    result = solver.solve_galaxy(galaxy, params)
    diagnosis = diagnose_galaxy(galaxy, solver, params)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Rotation curve with outliers highlighted
    ax1.errorbar(galaxy.R_kpc, galaxy.v_obs, yerr=galaxy.v_err,
                fmt='ko', alpha=0.7, markersize=4, label='Data')
    ax1.plot(galaxy.R_kpc, result['v_model'], 'r-', linewidth=2, label='Model')
    
    # Highlight outliers
    outlier_indices = diagnosis['outliers']['indices']
    if len(outlier_indices) > 0:
        ax1.scatter(galaxy.R_kpc[outlier_indices], galaxy.v_obs[outlier_indices],
                   c='red', s=100, marker='x', linewidth=3, label='Outliers (>3σ)')
    
    ax1.set_xlabel('Radius (kpc)')
    ax1.set_ylabel('Velocity (km/s)')
    ax1.set_title(f"{galaxy.name}: χ²/N = {result['chi2_reduced']:.2f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual heat map
    ax2.scatter(galaxy.R_kpc, result['residuals'], 
               c=diagnosis['chi2_per_point'], s=50, cmap='hot')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.fill_between(galaxy.R_kpc, -galaxy.v_err, galaxy.v_err,
                    alpha=0.2, color='gray')
    ax2.set_xlabel('Radius (kpc)')
    ax2.set_ylabel('Residuals (km/s)')
    ax2.set_title('Residuals colored by χ² contribution')
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('χ² per point')
    ax2.grid(True, alpha=0.3)
    
    # 3. Surface density profiles
    ax3.semilogy(galaxy.R_kpc, galaxy.sigma_gas, 'b-', label='Gas')
    ax3.semilogy(galaxy.R_kpc, galaxy.sigma_disk, 'g-', label='Disk')
    if galaxy.sigma_bulge is not None:
        ax3.semilogy(galaxy.R_kpc, galaxy.sigma_bulge, 'r-', label='Bulge')
    ax3.set_xlabel('Radius (kpc)')
    ax3.set_ylabel('Σ (M☉/pc²)')
    ax3.set_title('Surface Density Components')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Issues summary
    ax4.axis('off')
    ax4.text(0.05, 0.95, f"Diagnostic Summary for {galaxy.name}", 
            transform=ax4.transAxes, fontsize=14, weight='bold')
    
    y_pos = 0.85
    if len(diagnosis['issues']) > 0:
        for issue in diagnosis['issues']:
            color = {'high': 'red', 'medium': 'orange', 'low': 'yellow'}[issue['severity']]
            ax4.text(0.05, y_pos, f"• [{issue['severity'].upper()}] {issue['description']}", 
                    transform=ax4.transAxes, fontsize=11, color=color)
            y_pos -= 0.08
    else:
        ax4.text(0.05, y_pos, "No major issues detected", 
                transform=ax4.transAxes, fontsize=11, color='green')
    
    ax4.text(0.05, 0.3, f"Max χ² contribution: {diagnosis['max_chi2_point']['contribution']:.1f} at r={diagnosis['max_chi2_point']['radius']:.2f} kpc", 
            transform=ax4.transAxes, fontsize=11)
    
    if len(outlier_indices) > 0:
        ax4.text(0.05, 0.2, f"Outliers: {len(outlier_indices)} points exceed 3σ", 
                transform=ax4.transAxes, fontsize=11, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
    
    return diagnosis

def analyze_worst_galaxies(n_worst=20):
    """Analyze the worst-fitting galaxies"""
    # Load previous results
    results_df = pd.read_csv('sparc_analysis_results/sparc_analysis_results.csv')
    
    # Get worst galaxies
    worst_galaxies = results_df.nlargest(n_worst, 'chi2_reduced')
    
    # Load optimized parameters
    with open('optimization_results.json', 'r') as f:
        opt_data = json.load(f)
    params = opt_data['best_parameters']
    
    # Extract only global parameters for solver
    global_params = {k: v for k, v in params.items() if k != 'h_scale'}
    
    # Initialize solver
    solver = EnhancedGravitySolver(**global_params)
    
    # Create diagnostics directory
    os.makedirs('diagnostics', exist_ok=True)
    
    # Analyze each galaxy
    diagnoses = []
    for _, row in worst_galaxies.iterrows():
        print(f"\nAnalyzing {row['name']} (χ²/N = {row['chi2_reduced']:.2f})...")
        
        # Load galaxy
        filepath = os.path.join('Rotmod_LTG', f"{row['name']}_rotmod.dat")
        galaxy = load_galaxy_from_file(filepath)
        
        if galaxy is not None:
            # Create diagnostic plot
            save_path = os.path.join('diagnostics', f"{galaxy.name}_diagnostic.png")
            diagnosis = create_diagnostic_plot(galaxy, solver, save_path=save_path)
            
            # Add galaxy name
            diagnosis['name'] = galaxy.name
            diagnoses.append(diagnosis)
            
            # Print summary
            print(f"  Issues found: {len(diagnosis['issues'])}")
            for issue in diagnosis['issues']:
                print(f"    - [{issue['severity']}] {issue['description']}")
    
    # Save diagnoses
    with open('diagnostics/outlier_analysis.json', 'w') as f:
        json.dump(diagnoses, f, indent=2)
    
    # Create summary report
    with open('diagnostics/outlier_summary.txt', 'w') as f:
        f.write("OUTLIER GALAXY DIAGNOSTIC SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        # Count issue types
        issue_counts = {}
        for diag in diagnoses:
            for issue in diag['issues']:
                issue_type = issue['type']
                if issue_type not in issue_counts:
                    issue_counts[issue_type] = 0
                issue_counts[issue_type] += 1
        
        f.write("Most common issues:\n")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - {issue_type}: {count} galaxies\n")
        
        f.write(f"\nTotal galaxies analyzed: {len(diagnoses)}\n")
        f.write(f"Galaxies with issues: {sum(1 for d in diagnoses if len(d['issues']) > 0)}\n")
    
    print(f"\nDiagnostics saved to diagnostics/")
    return diagnoses

if __name__ == "__main__":
    analyze_worst_galaxies(20) 