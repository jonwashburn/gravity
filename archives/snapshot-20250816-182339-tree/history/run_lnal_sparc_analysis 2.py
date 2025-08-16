#!/usr/bin/env python3
"""
Run LNAL SPARC Analysis with Corrected Framework
===============================================
Uses the corrected a₀ = 1.195×10⁻¹⁰ m/s² from 4D voxel counting.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from lnal_gravity_fixed import LNALGravityFixed

def analyze_sparc_galaxies():
    """Analyze SPARC galaxies with corrected LNAL framework"""
    
    # Load pre-processed SPARC data
    print("Loading SPARC data...")
    with open('sparc_real_data.pkl', 'rb') as f:
        sparc_data = pickle.load(f)
    
    print(f"Found {len(sparc_data)} galaxies\n")
    
    # Initialize LNAL gravity solver
    lnal = LNALGravityFixed()
    print(f"Using a₀ = {lnal.a_0:.3e} m/s²\n")
    
    # Results storage
    results = []
    chi2_values = []
    
    # Process each galaxy
    for i, (name, galaxy) in enumerate(sparc_data.items()):
        if i % 10 == 0:
            print(f"Processing galaxy {i+1}/{len(sparc_data)}...")
        
        try:
            # Extract data from the curve subdictionary
            if 'curve' not in galaxy or galaxy['curve'] is None:
                continue
                
            curve = galaxy['curve']
            catalog = galaxy.get('catalog', {})
            
            r_kpc = np.array(curve['r'])
            v_obs = np.array(curve['V_obs'])
            v_err = np.array(curve.get('e_V', np.maximum(0.03 * v_obs, 2.0)))
            
            # Get velocity components
            v_gas = np.array(curve.get('V_gas', np.zeros_like(r_kpc)))
            v_disk = np.array(curve.get('V_disk', np.zeros_like(r_kpc)))
            v_bul = np.array(curve.get('V_bul', np.zeros_like(r_kpc)))
            
            # Get surface densities if available
            if 'SB_disk' in curve and 'SB_gas' in curve:
                # Use surface brightness/density directly
                SB_disk = np.array(curve['SB_disk'])  # L_sun/pc^2
                SB_gas = np.array(curve['SB_gas'])    # L_sun/pc^2
                
                # Convert to mass surface density assuming M/L ratios
                # Typical disk M/L ~ 0.5 in 3.6μm band
                # Gas includes 1.33 for helium
                sigma_disk = 0.5 * SB_disk  # M_sun/pc^2
                sigma_gas = 1.33 * SB_gas   # M_sun/pc^2
                
                # Integrate to get total masses
                if len(r_kpc) > 1:
                    dr = np.gradient(r_kpc)  # kpc
                    area_ring = 2 * np.pi * r_kpc * dr  # kpc^2
                    # Convert: 1 kpc^2 = 1e6 pc^2
                    M_disk = np.sum(sigma_disk * area_ring * 1e6)  # M_sun
                    M_gas = np.sum(sigma_gas * area_ring * 1e6)    # M_sun
                    
                    # Estimate scale lengths from profiles
                    # Find where surface density drops to 1/e
                    if np.any(sigma_disk > 0):
                        sigma_max = np.max(sigma_disk)
                        idx_e = np.argmin(np.abs(sigma_disk - sigma_max/np.e))
                        R_d = r_kpc[idx_e]
                    else:
                        R_d = 2.0
                    R_gas = R_d * 2.0
                else:
                    # Fallback for single point
                    M_disk = 1e10
                    M_gas = 1e9
                    R_d = 2.0
                    R_gas = 4.0
            else:
                # Fallback: estimate from velocities
                G_kpc = 4.302e-6  # G in (km/s)²⋅kpc/M_sun
                
                if np.any(v_disk > 0):
                    # Estimate from peak disk velocity
                    v_peak = np.max(v_disk)
                    r_peak = r_kpc[np.argmax(v_disk)]
                    M_disk = v_peak**2 * r_peak / G_kpc
                    R_d = r_peak / 2.2  # Typical for exponential disk
                else:
                    M_disk = 1e10
                    R_d = 2.0
                    
                if np.any(v_gas > 0):
                    v_gas_peak = np.max(v_gas)
                    r_gas_peak = r_kpc[np.argmax(v_gas)]
                    M_gas = v_gas_peak**2 * r_gas_peak / G_kpc
                    R_gas = R_d * 2.0
                else:
                    M_gas = 0
                    R_gas = R_d
            
            # Get LNAL prediction
            v_newton, v_lnal, mu_values = lnal.galaxy_rotation_curve(
                r_kpc, M_disk, R_d, M_gas, R_gas
            )
            
            # Calculate chi-squared
            residuals = v_obs - v_lnal
            chi2 = np.sum((residuals / v_err)**2)
            chi2_reduced = chi2 / (len(v_obs) - 2)
            
            # Store results
            results.append({
                'name': name,
                'chi2': chi2,
                'chi2_reduced': chi2_reduced,
                'N_points': len(v_obs),
                'M_disk': M_disk,
                'R_d': R_d,
                'M_gas': M_gas,
                'r_kpc': r_kpc,
                'v_obs': v_obs,
                'v_err': v_err,
                'v_lnal': v_lnal,
                'v_newton': v_newton
            })
            chi2_values.append(chi2_reduced)
            
        except Exception as e:
            print(f"  Error processing {name}: {e}")
            continue
    
    # Summary statistics
    chi2_values = np.array(chi2_values)
    print("\n" + "="*60)
    print("LNAL SPARC ANALYSIS RESULTS (Corrected a₀)")
    print("="*60)
    print(f"Galaxies analyzed: {len(chi2_values)}")
    print(f"Mean χ²/N: {np.mean(chi2_values):.3f}")
    print(f"Median χ²/N: {np.median(chi2_values):.3f}")
    print(f"Best χ²/N: {np.min(chi2_values):.3f}")
    print(f"Worst χ²/N: {np.max(chi2_values):.3f}")
    print(f"Fraction with χ²/N < 2: {np.sum(chi2_values < 2)/len(chi2_values)*100:.1f}%")
    print(f"Fraction with χ²/N < 5: {np.sum(chi2_values < 5)/len(chi2_values)*100:.1f}%")
    
    # Save results
    with open('lnal_sparc_results_corrected.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to lnal_sparc_results_corrected.pkl")
    
    # Plot best fits
    sorted_results = sorted(results, key=lambda x: x['chi2_reduced'])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(6):
        if i < len(sorted_results):
            res = sorted_results[i]
            ax = axes[i]
            
            ax.errorbar(res['r_kpc'], res['v_obs'], yerr=res['v_err'],
                       fmt='ko', alpha=0.6, markersize=4, label='Data')
            ax.plot(res['r_kpc'], res['v_lnal'], 'b-', linewidth=2,
                   label=f'LNAL (χ²/N={res["chi2_reduced"]:.2f})')
            ax.plot(res['r_kpc'], res['v_newton'], 'r--', alpha=0.7,
                   label='Newton')
            
            ax.set_xlabel('R (kpc)')
            ax.set_ylabel('V (km/s)')
            ax.set_title(res['name'])
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(res['r_kpc'])*1.1)
            ax.set_ylim(0, max(res['v_obs'])*1.2)
    
    plt.tight_layout()
    plt.savefig('lnal_sparc_best_fits_corrected.png', dpi=150, bbox_inches='tight')
    print(f"Best fits plot saved to lnal_sparc_best_fits_corrected.png")
    
    # Chi-squared distribution
    plt.figure(figsize=(8, 6))
    plt.hist(chi2_values, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(1.0, color='red', linestyle='--', label='χ²/N = 1')
    plt.axvline(np.median(chi2_values), color='green', linestyle='--', 
                label=f'Median = {np.median(chi2_values):.2f}')
    plt.xlabel('χ²/N')
    plt.ylabel('Number of galaxies')
    plt.title('LNAL SPARC Analysis: χ² Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lnal_sparc_chi2_distribution.png', dpi=150, bbox_inches='tight')
    print(f"Chi-squared distribution saved to lnal_sparc_chi2_distribution.png")
    
    return results

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LNAL GRAVITY SPARC ANALYSIS - CORRECTED FRAMEWORK")
    print("a₀ = 1.195×10⁻¹⁰ m/s² (from 4D voxel counting)")
    print("="*70 + "\n")
    
    results = analyze_sparc_galaxies() 