#!/usr/bin/env python3
"""
RS Gravity v5 - Parallel SPARC Analysis
Full pipeline with optimized processing
"""

import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our optimized solver
from rs_gravity_v5_optimized import RSGravityOptimized as RSGravityV5
from parse_sparc_mrt import parse_sparc_mrt

def load_sparc_data():
    """Load SPARC galaxy catalog"""
    sparc_file = "SPARC_Lelli2016c.mrt.txt"
    
    # Use the existing parser - it returns a list of dicts
    galaxies_list = parse_sparc_mrt(sparc_file)
    
    # Convert to DataFrame with the fields we need
    galaxies = []
    for data in galaxies_list:
        galaxy = {
            'name': data['name'],
            'type': data.get('type', 10),
            'distance': data.get('distance', 10.0),
            'distance_err': 0.1 * data.get('distance', 10.0),  # Approximate
            'inclination': data.get('inclination', 60.0),
            'inclination_err': 5.0,  # Approximate
            'L36': data.get('L_36', 1.0),
            'L36_err': 0.1 * data.get('L_36', 1.0),  # Approximate
            'scale_length': data.get('R_disk', 1.0),
            'vflat': data.get('V_flat', 100.0),
            'quality': data.get('quality', 3)
        }
        # Only add if we have valid data
        if galaxy['L36'] > 0 and galaxy['distance'] > 0:
            galaxies.append(galaxy)
    
    return pd.DataFrame(galaxies)

def load_rotation_curve(galaxy_name):
    """Load rotation curve for a galaxy"""
    # Check both possible locations
    rotmod_file = Path(f"Rotmod_LTG/{galaxy_name}_rotmod.dat")
    
    if not rotmod_file.exists():
        return None
    
    try:
        # Skip comment lines and load data
        data = []
        with open(rotmod_file, 'r') as f:
            for line in f:
                if not line.startswith('#') and line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        data.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        if len(data) < 5:  # Need at least 5 points
            return None
            
        data = np.array(data)
        return {
            'r': data[:, 0],
            'v_obs': data[:, 1],
            'v_err': data[:, 2]
        }
    except:
        return None

def process_galaxy(galaxy_data, solver_params):
    """Process a single galaxy"""
    name = galaxy_data['name']
    
    # Load rotation curve
    rot_data = load_rotation_curve(name)
    if rot_data is None:
        return {
            'name': name,
            'status': 'no_data',
            'chi2': np.nan,
            'chi2_per_n': np.nan
        }
    
    # Check for sufficient points
    if len(rot_data['r']) < 5:
        return {
            'name': name,
            'status': 'too_few_points',
            'chi2': np.nan,
            'chi2_per_n': np.nan
        }
    
    try:
        # Create solver instance with galaxy name
        solver = RSGravityV5(name, **solver_params)
        
        # Prepare baryon data
        M_star = galaxy_data['L36'] * 1e9 * 0.6  # M/L = 0.6
        scale_length = galaxy_data['scale_length']
        
        # Convert to SI units
        kpc = 3.086e19  # m
        M_sun = 1.989e30  # kg
        
        # Exponential disk profile
        def baryon_density(r_kpc):
            r_m = r_kpc * kpc
            Sigma_0 = M_star * M_sun / (2 * np.pi * (scale_length * kpc)**2)
            Sigma = Sigma_0 * np.exp(-r_kpc / scale_length)
            h_z = 0.3 * kpc  # 300 pc scale height
            return Sigma / (2 * h_z)  # thin disk approximation
        
        # Solve
        r_kpc = rot_data['r']
        v_obs = rot_data['v_obs']
        v_err = rot_data['v_err']
        
        # Convert to SI
        r_m = r_kpc * kpc
        v_obs_m = v_obs * 1000  # km/s to m/s
        v_err_m = v_err * 1000
        
        # Create density array
        rho_b = np.array([baryon_density(r) for r in r_kpc])
        
        # Prepare velocity components (empty for disk galaxies)
        v_components = {
            'gas': np.zeros_like(r_m),
            'disk': np.zeros_like(r_m),
            'bulge': np.zeros_like(r_m)
        }
        
        # Solve for velocity curve
        v_pred, v_baryon, t_elapsed = solver.predict_rotation_curve(r_m, rho_b, v_components)
        
        # Convert back to km/s
        v_model = v_pred / 1000
        
        # Calculate chi-squared
        chi2 = np.sum(((v_obs - v_model) / v_err)**2)
        chi2_per_n = chi2 / len(v_obs)
        
        return {
            'name': name,
            'status': 'success',
            'chi2': chi2,
            'chi2_per_n': chi2_per_n,
            'r_obs': r_kpc,
            'v_obs': v_obs,
            'v_err': v_err,
            'v_model': v_model,
            'v_baryon': v_baryon / 1000  # Convert to km/s
        }
        
    except Exception as e:
        return {
            'name': name,
            'status': f'error: {str(e)}',
            'chi2': np.nan,
            'chi2_per_n': np.nan
        }

def parallel_process_galaxies(df, solver_params, n_processes=None):
    """Process galaxies in parallel"""
    if n_processes is None:
        n_processes = mp.cpu_count() - 1
    
    # Create process pool
    with mp.Pool(n_processes) as pool:
        # Partial function with fixed solver params
        process_func = partial(process_galaxy, solver_params=solver_params)
        
        # Convert dataframe to list of dicts
        galaxy_list = df.to_dict('records')
        
        # Process with progress bar
        results = list(tqdm(
            pool.imap(process_func, galaxy_list),
            total=len(galaxy_list),
            desc="Processing galaxies"
        ))
    
    return results

def plot_results(results, output_dir="sparc_results_v5"):
    """Plot analysis results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Filter successful results
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print("No successful fits to plot!")
        return
    
    # Sort by chi2
    successful.sort(key=lambda x: x['chi2_per_n'])
    
    # Plot best and worst fits
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot 3 best and 3 worst
    to_plot = successful[:3] + successful[-3:]
    
    for ax, result in zip(axes, to_plot):
        r = result['r_obs']
        v_obs = result['v_obs']
        v_err = result['v_err']
        v_model = result['v_model']
        
        ax.errorbar(r, v_obs, yerr=v_err, fmt='o', alpha=0.6, label='Observed')
        ax.plot(r, v_model, 'r-', lw=2, label='RS Model')
        
        # Also plot full resolution
        if 'v_baryon' in result:
            ax.plot(r, result['v_baryon'], 'b--', alpha=0.5, lw=1, label='Baryons')
        
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Velocity (km/s)')
        ax.set_title(f"{result['name']}: χ²/N = {result['chi2_per_n']:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "best_worst_fits.png", dpi=150)
    plt.close()
    
    # Chi-squared distribution
    chi2_values = [r['chi2_per_n'] for r in successful]
    
    plt.figure(figsize=(10, 6))
    plt.hist(chi2_values, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(5, color='red', linestyle='--', label='χ²/N = 5')
    plt.xlabel('χ²/N')
    plt.ylabel('Number of Galaxies')
    plt.title(f'Chi-squared Distribution ({len(successful)} galaxies)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "chi2_distribution.png", dpi=150)
    plt.close()

def save_results(results, df):
    """Save results to files"""
    output_dir = Path("sparc_results_v5")
    output_dir.mkdir(exist_ok=True)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "sparc_results.csv", index=False)
    
    # Summary statistics
    successful = results_df[results_df['status'] == 'success']
    
    summary = {
        'total_galaxies': len(results),
        'successful_fits': len(successful),
        'success_rate': len(successful) / len(results),
        'median_chi2': float(successful['chi2_per_n'].median()) if len(successful) > 0 else np.nan,
        'mean_chi2': float(successful['chi2_per_n'].mean()) if len(successful) > 0 else np.nan,
        'best_fit': {
            'name': successful.iloc[0]['name'] if len(successful) > 0 else None,
            'chi2_per_n': float(successful.iloc[0]['chi2_per_n']) if len(successful) > 0 else np.nan
        } if len(successful) > 0 else None,
        'good_fits': len(successful[successful['chi2_per_n'] < 5]) if len(successful) > 0 else 0,
        'status_counts': results_df['status'].value_counts().to_dict()
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n=== Analysis Summary ===")
    print(f"Total galaxies: {summary['total_galaxies']}")
    print(f"Successful fits: {summary['successful_fits']} ({summary['success_rate']:.1%})")
    if summary['successful_fits'] > 0:
        print(f"Median χ²/N: {summary['median_chi2']:.2f}")
        print(f"Good fits (χ²/N < 5): {summary['good_fits']}")
        print(f"Best fit: {summary['best_fit']['name']} (χ²/N = {summary['best_fit']['chi2_per_n']:.2f})")
    
    print("\nFailed fits:")
    for status, count in summary['status_counts'].items():
        if status != 'success':
            print(f"  {status}: {count}")

def main():
    """Main analysis pipeline"""
    print("=== RS Gravity v5 - Parallel SPARC Analysis ===")
    print(f"CPUs available: {mp.cpu_count()}")
    
    # Load SPARC catalog
    print("\nLoading SPARC catalog...")
    df = load_sparc_data()
    print(f"Loaded {len(df)} galaxies from SPARC")
    
    # Set up solver parameters (optimized values)
    solver_params = {
        'use_gpu': False,           # Set True if GPU available
    }
    
    # Process galaxies in parallel
    n_processes = min(mp.cpu_count() - 1, 9)  # Leave one CPU free
    print(f"\nProcessing {len(df)} galaxies with {n_processes} processes...")
    
    start_time = time.time()
    results = parallel_process_galaxies(df, solver_params, n_processes)
    elapsed = time.time() - start_time
    
    print(f"Completed in {elapsed:.1f} seconds")
    print(f"Average time per galaxy: {elapsed/len(df)*1000:.1f} ms")
    
    # Save and plot results
    save_results(results, df)
    plot_results(results)
    
    print("\nResults saved to sparc_results_v5/")

if __name__ == "__main__":
    main() 