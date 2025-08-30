#!/usr/bin/env python3
"""
Reproduce the 0.48 fit from the paper using exact optimized parameters.
Uses the final combined solver with the 5 global parameters from the paper.
"""

import numpy as np
import pickle
import sys
import os

# Add parent directory to path to import from Gravity Reproduction
sys.path.insert(0, 'Gravity Reproduction')

from ledger_final_combined import (
    recognition_weight_combined, 
    fit_galaxy_combined,
    global_objective_combined,
    analyze_final_results,
    compute_full_error,
    G_kpc
)

def main():
    """Run the exact 0.48 fit with paper parameters"""
    print("\n" + "="*60)
    print("REPRODUCING THE 0.48 FIT FROM PAPER")
    print("="*60)
    
    # Load master table
    print("\nLoading SPARC master table...")
    with open('sparc_master.pkl', 'rb') as f:
        master_table = pickle.load(f)
    
    print(f"Loaded {len(master_table)} galaxies")
    
    # EXACT parameters from the paper (Gravity_First_Principles.txt)
    # These are the values that achieved χ²/N = 0.48
    params_paper = [
        # Global parameters
        0.194,    # α (time scaling exponent)
        5.064,    # C₀ (gas complexity coefficient)
        2.953,    # γ (gas fraction exponent, nearly cubic)
        0.216,    # δ (surface brightness exponent)
        0.3,      # h_z/R_d (vertical scale ratio)
        
        # Hyperparameters (from ledger_refined_model.py context)
        0.003,    # smoothness (spline regularization)
        0.032,    # prior_strength (profile regularization)
        
        # Error model parameters 
        0.25,     # α_beam (beam smearing coefficient)
        0.5       # β_asym (asymmetric drift coefficient)
    ]
    
    print("\nUsing EXACT parameters from paper:")
    print(f"  α = {params_paper[0]:.3f}")
    print(f"  C₀ = {params_paper[1]:.3f}")
    print(f"  γ = {params_paper[2]:.3f}")
    print(f"  δ = {params_paper[3]:.3f}")
    print(f"  h_z/R_d = {params_paper[4]:.3f}")
    print(f"\nExpected global λ ≈ 0.119")
    
    # Calculate global χ²/N and get detailed results
    print("\nCalculating fits for all galaxies...")
    chi2_global, lambda_norm, results = global_objective_combined(
        params_paper, master_table, return_details=True
    )
    
    print(f"\nActual global λ = {lambda_norm:.4f}")
    print(f"Average boost factor = {1/lambda_norm:.1f}×")
    
    # Analyze results
    chi2_values = [r['chi2_reduced'] for r in results]
    dwarf_chi2 = [r['chi2_reduced'] for r in results if r['galaxy_type'] == 'dwarf']
    spiral_chi2 = [r['chi2_reduced'] for r in results if r['galaxy_type'] == 'spiral']
    
    print("\n" + "="*60)
    print("REPRODUCTION RESULTS")
    print("="*60)
    print(f"\nOverall performance ({len(results)} galaxies):")
    print(f"  Median χ²/N = {np.median(chi2_values):.3f}")
    print(f"  Mean χ²/N = {np.mean(chi2_values):.3f}")
    print(f"  Best χ²/N = {np.min(chi2_values):.3f}")
    print(f"  Worst χ²/N = {np.max(chi2_values):.3f}")
    
    print("\nBy galaxy type:")
    if dwarf_chi2:
        print(f"  Dwarfs: median = {np.median(dwarf_chi2):.3f} (N={len(dwarf_chi2)})")
    if spiral_chi2:
        print(f"  Spirals: median = {np.median(spiral_chi2):.3f} (N={len(spiral_chi2)})")
    
    print("\nPerformance distribution:")
    print(f"  χ²/N < 0.5: {100*np.mean(np.array(chi2_values) < 0.5):.1f}%")
    print(f"  χ²/N < 1.0: {100*np.mean(np.array(chi2_values) < 1.0):.1f}%")
    print(f"  χ²/N < 1.5: {100*np.mean(np.array(chi2_values) < 1.5):.1f}%")
    print(f"  χ²/N < 2.0: {100*np.mean(np.array(chi2_values) < 2.0):.1f}%")
    
    # Compare to paper claims
    print("\n" + "="*60)
    print("COMPARISON TO PAPER")
    print("="*60)
    print(f"Paper claimed median χ²/N = 0.48")
    print(f"We reproduced median χ²/N = {np.median(chi2_values):.3f}")
    
    if abs(np.median(chi2_values) - 0.48) < 0.1:
        print("\n✓ SUCCESS: Reproduction matches paper within tolerance!")
    else:
        print("\n⚠ WARNING: Reproduction differs from paper.")
        print("This may be due to:")
        print("  - Different spline control points per galaxy")
        print("  - Minor implementation differences")
        print("  - Additional fine-tuning in the paper")
    
    # Save results
    output = {
        'params_paper': params_paper,
        'lambda_norm': lambda_norm,
        'results': results,
        'chi2_values': chi2_values,
        'median_chi2': np.median(chi2_values),
        'mean_chi2': np.mean(chi2_values)
    }
    
    with open('reproduction_048_results.pkl', 'wb') as f:
        pickle.dump(output, f)
    
    print(f"\nResults saved to: reproduction_048_results.pkl")
    
    # Print best-fit galaxies
    print("\n" + "="*60)
    print("TOP 10 BEST-FIT GALAXIES")
    print("="*60)
    
    results_sorted = sorted(results, key=lambda x: x['chi2_reduced'])
    for i, res in enumerate(results_sorted[:10]):
        print(f"{i+1:2d}. {res['name']:15s} χ²/N = {res['chi2_reduced']:6.3f} ({res['galaxy_type']})")
    
    print("\nReproduction complete!")

if __name__ == "__main__":
    main() 