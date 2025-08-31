#!/usr/bin/env python3
"""
LNAL Recognition Science Gravity - Simplified Implementation
Focus on the key insight: Running Newton's constant G(r) = G_∞ (λ_rec/r)^β
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Physical constants
c = 2.998e8  # m/s
G_inf = 6.674e-11  # m³/kg/s² (cosmic scale)
kpc_to_m = 3.086e19  # m/kpc
km_to_m = 1000  # m/km

# Recognition Science constants
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
beta = -(phi - 1) / phi**5  # ≈ -0.0557
lambda_rec = 42.9e-9  # m (42.9 nm)
g_dagger = 1.2e-10  # m/s² (MOND scale)
CLOCK_LAG = 45 / 960  # 4.69% cosmological clock lag

print(f"Recognition Science Parameters:")
print(f"  β = {beta:.6f}")
print(f"  λ_rec = {lambda_rec*1e9:.1f} nm")
print(f"  Clock lag = {CLOCK_LAG*100:.2f}%")

class SimpleRecognitionGravity:
    def __init__(self, baryon_data_file='sparc_exact_baryons.pkl'):
        self.baryon_data = self.load_baryon_data(baryon_data_file)
        
    def load_baryon_data(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded {len(data)} galaxies")
            return data
        return {}
    
    def G_running(self, r):
        """Running Newton coupling: G(r) = G_∞ (λ_rec/r)^β"""
        return G_inf * (lambda_rec / r)**beta
    
    def compute_acceleration(self, r, sigma_total):
        """
        Compute acceleration with running G(r) and MOND interpolation
        """
        # Running Newton's constant at this radius
        G_r = self.G_running(r)
        
        # Newtonian acceleration with running G
        a_N = 2 * np.pi * G_r * sigma_total
        
        # MOND parameter
        x = a_N / g_dagger
        
        # Simple MOND interpolation
        mu = x / np.sqrt(1 + x**2)
        
        # Total acceleration
        # In deep MOND: a ≈ √(a_N * g_dagger)
        # In Newtonian: a ≈ a_N
        a_total = a_N * mu + np.sqrt(a_N * g_dagger) * (1 - mu)
        
        # Apply clock lag correction
        a_total *= (1 + CLOCK_LAG)
        
        return a_total, a_N
    
    def solve_galaxy(self, galaxy_name):
        if galaxy_name not in self.baryon_data:
            return None
            
        data = self.baryon_data[galaxy_name]
        R_kpc = data['radius']
        v_obs = data['v_obs']
        v_err = data['v_err']
        
        # Total surface density
        sigma_total = data['sigma_gas'] + data['sigma_disk'] + data['sigma_bulge']
        
        # Convert radius to meters
        R = R_kpc * kpc_to_m
        
        # Compute accelerations
        a_total, a_N = self.compute_acceleration(R, sigma_total)
        
        # Convert to velocity
        v_model = np.sqrt(a_total * R) / km_to_m  # km/s
        v_baryon = np.sqrt(a_N * R) / km_to_m
        
        # Ensure v_model >= v_baryon
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
            'chi2_reduced': chi2_reduced
        }
    
    def solve_all(self, max_galaxies=None):
        galaxies = list(self.baryon_data.keys())[:max_galaxies]
        
        results = []
        chi2_values = []
        
        print(f"\nSolving {len(galaxies)} galaxies...")
        for i, name in enumerate(galaxies):
            result = self.solve_galaxy(name)
            if result:
                results.append(result)
                chi2_values.append(result['chi2_reduced'])
                print(f"[{i+1:3d}] {name:15s} χ²/N = {result['chi2_reduced']:7.2f}")
        
        chi2_values = np.array(chi2_values)
        print(f"\nMean χ²/N: {np.mean(chi2_values):.2f}")
        print(f"Median χ²/N: {np.median(chi2_values):.2f}")
        print(f"Best: {np.min(chi2_values):.2f}")
        print(f"< 10: {np.mean(chi2_values < 10):.1%}")
        
        return results
    
    def plot_example(self, galaxy_name):
        result = self.solve_galaxy(galaxy_name)
        if not result:
            return
            
        plt.figure(figsize=(8, 6))
        plt.errorbar(result['R_kpc'], result['v_obs'], yerr=result['v_err'],
                    fmt='ko', alpha=0.6, label='Observed')
        plt.plot(result['R_kpc'], result['v_model'], 'r-', linewidth=2,
                label='Recognition Model')
        plt.plot(result['R_kpc'], result['v_baryon'], 'b--', 
                label='Baryonic')
        
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Velocity (km/s)')
        plt.title(f'{galaxy_name} - χ²/N = {result["chi2_reduced"]:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Test
solver = SimpleRecognitionGravity()
if solver.baryon_data:
    # Test on one galaxy
    test_galaxy = list(solver.baryon_data.keys())[0]
    print(f"\nTesting on {test_galaxy}...")
    solver.plot_example(test_galaxy)
    
    # Run on subset
    results = solver.solve_all(max_galaxies=20) 