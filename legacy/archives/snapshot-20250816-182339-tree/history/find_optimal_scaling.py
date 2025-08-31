#!/usr/bin/env python3
"""
Find optimal mass scaling factor for LNAL
=========================================
The LNAL predictions are systematically too high.
Find the global scaling factor that minimizes χ².
"""

import numpy as np
import pickle
from scipy.optimize import minimize_scalar
from lnal_gravity_fixed import LNALGravityFixed

def compute_total_chi2(scale_factor):
    """Compute total χ² for all galaxies with given mass scaling"""
    
    # Load pre-processed SPARC data
    with open('sparc_real_data.pkl', 'rb') as f:
        sparc_data = pickle.load(f)
    
    # Load previous results to get masses
    with open('lnal_sparc_results_corrected.pkl', 'rb') as f:
        previous_results = pickle.load(f)
    
    # Create name->result mapping
    result_map = {r['name']: r for r in previous_results}
    
    # Initialize LNAL
    lnal = LNALGravityFixed()
    
    total_chi2 = 0
    total_points = 0
    
    for name, galaxy in sparc_data.items():
        if name not in result_map:
            continue
            
        # Get previous mass estimates
        prev = result_map[name]
        M_disk = prev['M_disk'] * scale_factor  # Apply scaling
        M_gas = prev['M_gas'] * scale_factor
        R_d = prev['R_d']
        
        # Get data
        curve = galaxy['curve']
        r_kpc = np.array(curve['r'])
        v_obs = np.array(curve['V_obs'])
        v_err = np.array(curve.get('e_V', np.maximum(0.03 * v_obs, 2.0)))
        
        # Calculate LNAL prediction
        try:
            v_newton, v_lnal, mu_values = lnal.galaxy_rotation_curve(
                r_kpc, M_disk, R_d, M_gas, R_d*2
            )
            
            # Chi-squared
            residuals = v_obs - v_lnal
            chi2 = np.sum((residuals / v_err)**2)
            
            total_chi2 += chi2
            total_points += len(v_obs)
            
        except:
            continue
    
    chi2_per_point = total_chi2 / total_points if total_points > 0 else 1e10
    print(f"Scale factor {scale_factor:.3f}: χ²/N = {chi2_per_point:.1f}")
    
    return chi2_per_point

def main():
    print("Finding optimal mass scaling factor for LNAL...")
    print("="*50)
    
    # Search for optimal scaling
    result = minimize_scalar(compute_total_chi2, bounds=(0.1, 2.0), method='bounded')
    
    optimal_scale = result.x
    optimal_chi2 = result.fun
    
    print("\n" + "="*50)
    print(f"OPTIMAL SCALING FACTOR: {optimal_scale:.3f}")
    print(f"Resulting χ²/N: {optimal_chi2:.2f}")
    print("="*50)
    
    # What does this scaling mean?
    print(f"\nInterpretation:")
    print(f"- If scale < 1: Baryon masses are overestimated")
    print(f"- If scale > 1: Baryon masses are underestimated")
    print(f"- Scale = {optimal_scale:.3f} means multiply all masses by this factor")
    
    # Save result
    with open('lnal_optimal_scaling.txt', 'w') as f:
        f.write(f"Optimal mass scaling factor: {optimal_scale:.6f}\n")
        f.write(f"Resulting chi2/N: {optimal_chi2:.6f}\n")

if __name__ == "__main__":
    main() 