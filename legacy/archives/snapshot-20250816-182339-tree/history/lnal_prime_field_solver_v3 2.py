#!/usr/bin/env python3
"""
LNAL Prime Recognition Gravity: Mathematically Corrected Field Solver v3.0
Fixes dimensional analysis and numerical stability issues from v2.0
Uses proper scaling and units throughout.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import pickle
import os

# Physical constants (SI units)
c = 2.998e8  # m/s
G = 6.674e-11  # m³/kg/s²
hbar = 1.055e-34  # J⋅s
kpc_to_m = 3.086e19  # m/kpc
km_to_m = 1000  # m/km

# Recognition Science parameters
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
ell_1 = 0.97  # Primary recognition length (kpc)
ell_2 = 24.3  # Secondary recognition length (kpc)
g_dagger = 1.2e-10  # Universal acceleration scale (m/s²)

class PrimeFieldSolverV3:
    """Mathematically corrected prime field solver"""
    
    def __init__(self, baryon_data_file='sparc_exact_baryons.pkl'):
        self.baryon_data = self.load_baryon_data(baryon_data_file)
        self.results = {}
        
        # Derived parameters (dimensionally consistent)
        self.mu_inv = ell_1  # Inverse mass scale (kpc)
        self.alpha_0 = 1 / (phi - 1)  # Base coupling
        self.epsilon = phi**(-2)  # Enhancement factor
        
    def load_baryon_data(self, filename):
        """Load exact baryonic source data"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded exact baryon data for {len(data)} galaxies")
            return data
        else:
            print(f"Baryon data file {filename} not found")
            return {}
    
    def mond_interpolation(self, x):
        """MOND interpolation function μ(x) = x/√(1+x²)"""
        x_safe = np.clip(x, 0, 1e3)  # Prevent overflow
        return x_safe / np.sqrt(1 + x_safe**2)
    
    def scale_dependent_coupling(self, R_kpc):
        """Scale-dependent prime coupling α_p(R)"""
        return self.alpha_0 * (1 + self.epsilon * np.exp(-R_kpc / ell_2))
    
    def solve_galaxy_simplified(self, galaxy_name):
        """Solve using simplified but stable approach"""
        if galaxy_name not in self.baryon_data:
            return None
            
        data = self.baryon_data[galaxy_name]
        R = data['radius']  # kpc
        v_obs = data['v_obs']  # km/s
        v_err = data['v_err']  # km/s
        
        # Extract baryonic velocity components
        v_gas = np.sqrt(np.maximum(data['sigma_gas'] * 2 * np.pi * G * R * kpc_to_m, 0)) / km_to_m
        v_disk = np.sqrt(np.maximum(data['sigma_disk'] * 2 * np.pi * G * R * kpc_to_m, 0)) / km_to_m
        v_bulge = np.sqrt(np.maximum(data['sigma_bulge'] * 2 * np.pi * G * R * kpc_to_m, 0)) / km_to_m
        
        # Total baryonic velocity
        v_baryon = np.sqrt(v_gas**2 + v_disk**2 + v_bulge**2)
        
        # Newtonian acceleration from baryons
        a_N = (v_baryon * km_to_m)**2 / (R * kpc_to_m)  # m/s²
        
        # MOND-like enhancement with prime recognition
        x = a_N / g_dagger
        mu_x = self.mond_interpolation(x)
        
        # Scale-dependent enhancement
        alpha_R = self.scale_dependent_coupling(R)
        
        # Prime-enhanced acceleration
        a_prime = a_N * mu_x * alpha_R
        
        # Convert back to velocity
        v_model = np.sqrt(a_prime * R * kpc_to_m) / km_to_m
        
        # Ensure physical velocities
        v_model = np.maximum(v_model, v_baryon)
        
        # Compute χ²
        chi2 = np.sum(((v_obs - v_model) / v_err)**2)
        chi2_reduced = chi2 / len(v_obs)
        
        return {
            'galaxy': galaxy_name,
            'R': R,
            'v_obs': v_obs,
            'v_err': v_err,
            'v_model': v_model,
            'v_baryon': v_baryon,
            'chi2': chi2,
            'chi2_reduced': chi2_reduced,
            'N_points': len(v_obs)
        }
    
    def solve_all_galaxies(self, max_galaxies=None):
        """Solve for all galaxies"""
        galaxy_names = list(self.baryon_data.keys())
        if max_galaxies:
            galaxy_names = galaxy_names[:max_galaxies]
        
        print(f"Solving prime field equation for {len(galaxy_names)} galaxies...")
        
        results = []
        chi2_values = []
        
        for i, galaxy_name in enumerate(galaxy_names):
            print(f"[{i+1}/{len(galaxy_names)}] Solving {galaxy_name}...")
            
            try:
                result = self.solve_galaxy_simplified(galaxy_name)
                if result and np.isfinite(result['chi2_reduced']):
                    results.append(result)
                    chi2_values.append(result['chi2_reduced'])
                    print(f"  χ²/N = {result['chi2_reduced']:.3f}")
                else:
                    print(f"  Failed: invalid result")
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if not chi2_values:
            print("No valid results obtained")
            return None
        
        # Compute statistics
        chi2_mean = np.mean(chi2_values)
        chi2_std = np.std(chi2_values)
        chi2_median = np.median(chi2_values)
        
        print(f"\nOverall Results:")
        print(f"Galaxies processed: {len(results)}")
        print(f"Mean χ²/N: {chi2_mean:.3f} ± {chi2_std:.3f}")
        print(f"Median χ²/N: {chi2_median:.3f}")
        print(f"Range: {np.min(chi2_values):.3f} - {np.max(chi2_values):.3f}")
        
        self.results = {
            'individual': results,
            'chi2_mean': chi2_mean,
            'chi2_std': chi2_std,
            'chi2_median': chi2_median,
            'chi2_values': chi2_values,
            'n_galaxies': len(results)
        }
        
        # Save results
        with open('lnal_prime_v3_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        print("Results saved to lnal_prime_v3_results.pkl")
        
        return self.results
    
    def plot_example_fits(self, n_examples=6):
        """Plot example galaxy fits"""
        if not self.results or not self.results['individual']:
            print("No results to plot")
            return
        
        # Select examples with different χ² values
        results = self.results['individual']
        chi2_sorted = sorted(results, key=lambda x: x['chi2_reduced'])
        
        # Pick examples: best, worst, and some in between
        indices = [0, len(chi2_sorted)//4, len(chi2_sorted)//2, 
                  3*len(chi2_sorted)//4, len(chi2_sorted)-1]
        examples = [chi2_sorted[i] for i in indices[:n_examples-1]]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, result in enumerate(examples):
            ax = axes[i]
            
            # Plot rotation curve
            ax.errorbar(result['R'], result['v_obs'], yerr=result['v_err'], 
                       fmt='ko', alpha=0.7, markersize=4, label='Observed')
            ax.plot(result['R'], result['v_model'], 'r-', linewidth=2, 
                   label=f'Prime Model')
            ax.plot(result['R'], result['v_baryon'], 'b--', linewidth=1, 
                   alpha=0.7, label='Baryonic')
            
            ax.set_xlabel('Radius (kpc)')
            ax.set_ylabel('Velocity (km/s)')
            ax.set_title(f'{result["galaxy"]}\nχ²/N = {result["chi2_reduced"]:.2f}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(examples), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('lnal_prime_v3_examples.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_chi2_distribution(self):
        """Plot χ² distribution"""
        if not self.results or not self.results['chi2_values']:
            print("No results to plot")
            return
        
        chi2_values = self.results['chi2_values']
        
        plt.figure(figsize=(10, 6))
        plt.hist(chi2_values, bins=30, alpha=0.7, edgecolor='black', density=True)
        plt.axvline(1.0, color='red', linestyle='--', linewidth=2, 
                   label='Perfect fit (χ²/N = 1)')
        plt.axvline(self.results['chi2_mean'], color='blue', linestyle='-', 
                   linewidth=2, label=f'Mean = {self.results["chi2_mean"]:.3f}')
        plt.axvline(self.results['chi2_median'], color='green', linestyle='-', 
                   linewidth=2, label=f'Median = {self.results["chi2_median"]:.3f}')
        
        plt.xlabel('χ²/N')
        plt.ylabel('Probability Density')
        plt.title(f'LNAL Prime Recognition: χ² Distribution\n({self.results["n_galaxies"]} galaxies)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, min(10, np.percentile(chi2_values, 95)))
        
        plt.savefig('lnal_prime_v3_chi2_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main execution"""
    solver = PrimeFieldSolverV3()
    
    # Test on 10 galaxies first
    print("Testing on 10 galaxies...")
    test_results = solver.solve_all_galaxies(max_galaxies=10)
    
    if test_results and test_results['chi2_mean'] < 10.0:
        print("Test successful. Running full analysis...")
        full_results = solver.solve_all_galaxies()
        solver.plot_chi2_distribution()
        solver.plot_example_fits()
    else:
        if test_results:
            print(f"Test results: χ²/N = {test_results['chi2_mean']:.3f}")
        print("Test completed. Plotting test results...")
        solver.plot_chi2_distribution()
        solver.plot_example_fits()

if __name__ == "__main__":
    main() 