#!/usr/bin/env python3
"""
LNAL Prime Recognition: MOND Transition Implementation
Simplified approach focusing on robust MOND phenomenology
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Constants
c = 2.998e8  # m/s
G = 6.674e-11  # m³/kg/s²
kpc_to_m = 3.086e19  # m/kpc
km_to_m = 1000  # m/km
g_dagger = 1.2e-10  # m/s² (MOND scale)

# Recognition Science parameters
phi = (1 + np.sqrt(5)) / 2
ell_1_kpc = 0.97  # kpc
ell_2_kpc = 24.3  # kpc

class MONDTransitionSolver:
    """Implements MOND transition from Recognition Science"""
    
    def __init__(self, baryon_data_file='sparc_exact_baryons.pkl'):
        self.baryon_data = self.load_baryon_data(baryon_data_file)
        
    def load_baryon_data(self, filename):
        """Load baryon data"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def interpolation_function(self, x):
        """
        Interpolation function that transitions from Newtonian to MOND
        ν(x) where x = a/a₀
        """
        # Simple interpolation function
        # ν(x) → 1 for x >> 1 (Newtonian)
        # ν(x) → x for x << 1 (MOND)
        return x / np.sqrt(1 + x**2)
    
    def solve_galaxy(self, galaxy_name):
        """Solve using algebraic MOND formula"""
        if galaxy_name not in self.baryon_data:
            return None
            
        data = self.baryon_data[galaxy_name]
        R_kpc = data['radius']
        v_obs = data['v_obs']
        v_err = data['v_err']
        
        # Get baryonic acceleration
        sigma_total = data['sigma_gas'] + data['sigma_disk'] + data['sigma_bulge']
        a_baryon = 2 * np.pi * G * sigma_total  # m/s²
        
        # Apply MOND formula: a = ν(a/a₀) × a_baryon
        x = a_baryon / g_dagger
        nu_x = self.interpolation_function(x)
        
        # Total acceleration with Recognition Science enhancement
        # Add scale-dependent factor based on radius
        r_factor = 1 + 0.2 * np.exp(-R_kpc / ell_1_kpc)  # Enhancement near center
        a_total = a_baryon * nu_x * r_factor
        
        # In deep MOND regime, ensure correct asymptotic behavior
        deep_mond = x < 0.1
        if np.any(deep_mond):
            a_total[deep_mond] = np.sqrt(a_baryon[deep_mond] * g_dagger) * r_factor[deep_mond]
        
        # Convert to velocity
        R = R_kpc * kpc_to_m
        v_model = np.sqrt(a_total * R) / km_to_m  # km/s
        
        # Ensure physical bounds
        v_baryon = np.sqrt(a_baryon * R) / km_to_m
        v_model = np.maximum(v_model, v_baryon)
        
        # Compute χ²
        chi2 = np.sum(((v_obs - v_model) / v_err)**2)
        chi2_reduced = chi2 / len(v_obs)
        
        return {
            'galaxy': galaxy_name,
            'R_kpc': R_kpc,
            'v_obs': v_obs,
            'v_err': v_err,
            'v_model': v_model,
            'v_baryon': v_baryon,
            'a_baryon': a_baryon,
            'a_total': a_total,
            'x': x,
            'chi2_reduced': chi2_reduced
        }
    
    def test_all_galaxies(self, max_galaxies=None):
        """Test on all galaxies"""
        galaxy_names = list(self.baryon_data.keys())
        if max_galaxies:
            galaxy_names = galaxy_names[:max_galaxies]
            
        results = []
        chi2_values = []
        
        print(f"Testing MOND transition on {len(galaxy_names)} galaxies...")
        
        for i, galaxy in enumerate(galaxy_names):
            result = self.solve_galaxy(galaxy)
            if result:
                results.append(result)
                chi2_values.append(result['chi2_reduced'])
                if i < 10:  # Print first 10
                    print(f"{galaxy:12s}: χ²/N = {result['chi2_reduced']:6.2f}")
        
        chi2_values = np.array(chi2_values)
        print(f"\nMean χ²/N: {np.mean(chi2_values):.2f}")
        print(f"Median χ²/N: {np.median(chi2_values):.2f}")
        print(f"Best: {np.min(chi2_values):.2f}, Worst: {np.max(chi2_values):.2f}")
        
        return results, chi2_values
    
    def plot_example(self, result):
        """Plot example galaxy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Rotation curve
        ax1.errorbar(result['R_kpc'], result['v_obs'], yerr=result['v_err'],
                    fmt='ko', alpha=0.7, label='Observed')
        ax1.plot(result['R_kpc'], result['v_model'], 'r-', linewidth=2,
                label=f'Model (χ²/N={result["chi2_reduced"]:.1f})')
        ax1.plot(result['R_kpc'], result['v_baryon'], 'b--', alpha=0.7,
                label='Baryonic')
        ax1.set_xlabel('Radius (kpc)')
        ax1.set_ylabel('Velocity (km/s)')
        ax1.set_title(result['galaxy'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Acceleration relation
        ax2.loglog(result['a_baryon'], result['a_total'], 'o', alpha=0.7)
        
        # Theory curves
        a_range = np.logspace(-13, -8, 100)
        a_newton = a_range
        a_mond = np.sqrt(a_range * g_dagger)
        
        ax2.loglog(a_range, a_newton, 'k:', label='Newtonian')
        ax2.loglog(a_range, a_mond, 'r--', label='MOND')
        ax2.set_xlabel('a_baryon (m/s²)')
        ax2.set_ylabel('a_total (m/s²)')
        ax2.set_title('Acceleration Relation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    solver = MONDTransitionSolver()
    
    # Test implementation
    results, chi2_values = solver.test_all_galaxies(max_galaxies=20)
    
    # Plot best fit
    best_idx = np.argmin(chi2_values)
    solver.plot_example(results[best_idx])
    
    # Show what needs improvement
    print("\nAnalysis:")
    print("Current implementation uses simplified MOND transition.")
    print("To achieve χ²/N ≈ 1.0, we need:")
    print("1. Full information field dynamics (not just algebraic)")
    print("2. Prime number oscillations at small scales")
    print("3. Multi-scale recognition hierarchy")
    print("4. Proper boundary conditions at ℓ₁ and ℓ₂")

if __name__ == "__main__":
    main() 