#!/usr/bin/env python3
"""
Visualize the best-fit galaxy rotation curves from the 0.48 reproduction
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, 'Gravity Reproduction')
from ledger_final_combined import recognition_weight_combined

def plot_best_fits():
    """Plot the best-fit examples from our reproduction"""
    
    # Load results
    with open('reproduction_048_results.pkl', 'rb') as f:
        output = pickle.load(f)
    
    with open('sparc_master.pkl', 'rb') as f:
        master_table = pickle.load(f)
    
    results = output['results']
    params_paper = output['params_paper']
    lambda_norm = output['lambda_norm']
    
    # Sort by chi2
    results_sorted = sorted(results, key=lambda x: x['chi2_reduced'])
    
    # Select diverse examples
    examples = []
    # Best overall
    examples.append(results_sorted[0])
    # Best dwarf  
    dwarfs = [r for r in results_sorted if r['galaxy_type'] == 'dwarf']
    if dwarfs and dwarfs[0]['name'] != examples[0]['name']:
        examples.append(dwarfs[0])
    # Best spiral (not already selected)
    spirals = [r for r in results_sorted if r['galaxy_type'] == 'spiral']
    for s in spirals:
        if s['name'] not in [e['name'] for e in examples]:
            examples.append(s)
            break
    # Median case
    median_idx = len(results_sorted)//2
    if results_sorted[median_idx]['name'] not in [e['name'] for e in examples]:
        examples.append(results_sorted[median_idx])
    
    # Ensure we have 4 examples
    while len(examples) < 4:
        for r in results_sorted[len(examples):]:
            if r['name'] not in [e['name'] for e in examples]:
                examples.append(r)
                break
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for ax, res in zip(axes, examples[:4]):
        galaxy = master_table[res['name']]
        
        r = galaxy['r']
        v_obs = galaxy['v_obs']
        v_model = res['v_model']
        v_err = galaxy['data']['verr'].values
        n_r = res['n_r']
        
        # Newtonian prediction
        v_gas = galaxy['data']['vgas'].values
        v_disk = galaxy['data']['vdisk'].values * np.sqrt(res['ml'])
        v_bul = galaxy['data']['vbul'].values
        v_newton = np.sqrt(v_gas**2 + v_disk**2 + v_bul**2)
        
        # Main plot
        ax.errorbar(r, v_obs, yerr=v_err, fmt='ko', markersize=4,
                   alpha=0.8, label='Observed', capsize=2)
        ax.plot(r, v_model, 'r-', linewidth=2.5,
               label=f'LNAL fit (χ²/N={res["chi2_reduced"]:.3f})')
        ax.plot(r, v_newton, 'b--', linewidth=1.5, alpha=0.7,
               label='Newtonian')
        
        # Recognition weight as shaded region
        ax2 = ax.twinx()
        w = lambda_norm * res['n_r'] * res['zeta']  # Simplified w(r) for visualization
        ax2.fill_between(r, 0, w/lambda_norm, alpha=0.15, color='green')
        ax2.plot(r, w/lambda_norm, 'g-', alpha=0.5, linewidth=1)
        ax2.set_ylabel('w(r)/λ', color='green', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(0, max(w/lambda_norm)*1.2)
        
        # Labels and formatting
        ax.set_xlabel('Radius [kpc]', fontsize=11)
        ax.set_ylabel('Velocity [km/s]', fontsize=11)
        ax.set_title(f'{res["name"]} ({res["galaxy_type"]}), M/L={res["ml"]:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(r)*1.05)
        ax.set_ylim(0, max(v_obs)*1.1)
        
    plt.suptitle('LNAL Gravity: Best-Fit Galaxy Rotation Curves\n' + 
                 f'Global Parameters: α={params_paper[0]:.3f}, C₀={params_paper[1]:.3f}, ' +
                 f'γ={params_paper[2]:.3f}, δ={params_paper[3]:.3f}, λ={lambda_norm:.3f}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('best_fits_reproduction.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization: best_fits_reproduction.png")
    
    # Create histogram of chi2 values
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    chi2_values = [r['chi2_reduced'] for r in results]
    chi2_dwarfs = [r['chi2_reduced'] for r in results if r['galaxy_type'] == 'dwarf']
    chi2_spirals = [r['chi2_reduced'] for r in results if r['galaxy_type'] == 'spiral']
    
    bins = np.logspace(-2, 1.5, 30)
    
    ax.hist(chi2_values, bins=bins, alpha=0.6, label=f'All ({len(chi2_values)})', 
            color='blue', edgecolor='black')
    ax.hist(chi2_dwarfs, bins=bins, alpha=0.6, label=f'Dwarfs ({len(chi2_dwarfs)})',
            color='green', edgecolor='black')
    ax.hist(chi2_spirals, bins=bins, alpha=0.6, label=f'Spirals ({len(chi2_spirals)})',
            color='red', edgecolor='black')
    
    ax.axvline(np.median(chi2_values), color='blue', linestyle='--', linewidth=2,
              label=f'Median = {np.median(chi2_values):.3f}')
    ax.axvline(0.48, color='black', linestyle=':', linewidth=2,
              label='Paper claim = 0.48')
    
    ax.set_xscale('log')
    ax.set_xlabel('χ²/N', fontsize=12)
    ax.set_ylabel('Number of Galaxies', fontsize=12)
    ax.set_title('Distribution of Fit Quality\nLNAL Gravity Reproduction', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chi2_distribution_reproduction.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization: chi2_distribution_reproduction.png")

if __name__ == "__main__":
    plot_best_fits() 