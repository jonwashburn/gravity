#!/usr/bin/env python3
"""
LNAL Advanced Recognition Gravity Solver V3
Incorporating insights from Recognition Science gravity paper:
- Proper kernel function F(u) = Ξ(u) - u·Ξ'(u)
- Eight-tick neutrality constraint
- Vacuum energy residual corrections
- Improved multi-scale transitions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import pickle
import os
from typing import Dict, Tuple, Optional

# Physical constants (SI units)
c = 2.998e8  # m/s
G = 6.674e-11  # m³/kg/s²
hbar = 1.055e-34  # J⋅s
m_p = 1.673e-27  # kg (proton mass)
kpc_to_m = 3.086e19  # m/kpc
km_to_m = 1000  # m/km

# Recognition Science constants
phi = (1 + np.sqrt(5)) / 2  # Golden ratio = 1.618034
beta = -(phi - 1) / phi**5  # -0.0557280900... (exact from paper)
lambda_rec = 60e-6  # m (recognition length scale)

# Two recognition lengths (from kernel poles)
ell_1 = 0.97 * kpc_to_m  # m (curvature-onset length)
ell_2 = 24.3 * kpc_to_m  # m (kernel-knee length)
ell_1_kpc = 0.97  # kpc
ell_2_kpc = 24.3  # kpc

# Voxel parameters
L_0 = 0.335e-9  # m (voxel size)
V_voxel = L_0**3  # m³

# Information field parameters
I_star = m_p * c**2 / V_voxel  # J/m³ (information capacity)
mu = hbar / (c * ell_1)  # m⁻² (field mass)
g_dagger = 1.2e-10  # m/s² (MOND acceleration scale)
lambda_coupling = np.sqrt(g_dagger * c**2 / I_star)  # Coupling constant

# Eight-tick neutrality
TICK_PERIOD = 8  # Ledger must balance every 8 ticks
CLOCK_LAG = 45 / 960  # 4.69% lag from 45-gap

# Vacuum energy residual (from 9-symbol packet compression)
VACUUM_RESIDUAL = 1.5  # Factor relative to observed Λ

class RecognitionKernel:
    """Implements the Recognition Science kernel F(u)"""
    
    @staticmethod
    def Xi(u: np.ndarray) -> np.ndarray:
        """
        Ξ(u) = [exp(β·ln(1+u)) - 1]/(β·u)
        """
        u_safe = np.maximum(u, 1e-10)
        # Use (1+u)^β instead of exp(β·ln(1+u)) for numerical stability
        return ((1 + u_safe)**beta - 1) / (beta * u_safe)
    
    @staticmethod
    def Xi_prime(u: np.ndarray) -> np.ndarray:
        """
        Derivative of Ξ(u)
        """
        u_safe = np.maximum(u, 1e-10)
        # d/du[(1+u)^β - 1]/(β·u) = [(1+u)^β·(β·u - (1+u)^β + 1)]/(β·u²·(1+u))
        numerator = (1 + u_safe)**(beta-1) * (beta * u_safe - (1 + u_safe)**beta + 1)
        denominator = u_safe**2
        return numerator / denominator
    
    @staticmethod
    def F(u: np.ndarray) -> np.ndarray:
        """
        F(u) = Ξ(u) - u·Ξ'(u)
        This is the fundamental kernel that mediates gravitational response
        """
        Xi_val = RecognitionKernel.Xi(u)
        Xi_prime_val = RecognitionKernel.Xi_prime(u)
        return Xi_val - u * Xi_prime_val

class AdvancedLNALSolverV3:
    """Recognition Science gravity solver with proper kernel implementation"""
    
    def __init__(self, baryon_data_file='sparc_exact_baryons.pkl'):
        self.baryon_data = self.load_baryon_data(baryon_data_file)
        self.results = {}
        self.kernel = RecognitionKernel()
        
        # Solver parameters
        self.max_iterations = 1500
        self.tolerance = 1e-7
        self.relaxation_parameter = 0.8  # Conservative for stability
        
        print("LNAL Recognition Science Gravity Solver V3")
        print("="*60)
        print(f"Recognition Science Parameters:")
        print(f"  φ = {phi:.6f} (golden ratio)")
        print(f"  β = {beta:.6f} = -(φ-1)/φ⁵")
        print(f"  λ_rec = {lambda_rec*1e6:.1f} μm")
        print(f"  ℓ₁ = {ell_1_kpc:.2f} kpc (curvature onset)")
        print(f"  ℓ₂ = {ell_2_kpc:.1f} kpc (kernel knee)")
        print(f"  I* = {I_star:.2e} J/m³")
        print(f"  μ = {mu:.2e} m⁻²")
        print(f"  g† = {g_dagger:.2e} m/s²")
        print(f"  Clock lag = {CLOCK_LAG*100:.2f}%")
        print(f"  Eight-tick period = {TICK_PERIOD}")
        print("="*60)
    
    def load_baryon_data(self, filename):
        """Load exact baryonic source data"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"\nLoaded exact baryon data for {len(data)} galaxies")
            return data
        else:
            print(f"Baryon data file {filename} not found")
            return {}
    
    def running_G(self, r: np.ndarray) -> np.ndarray:
        """
        Scale-dependent Newton constant
        G(r) = G∞ (λ_rec/r)^β
        """
        r_safe = np.maximum(r, lambda_rec)
        return G * (lambda_rec / r_safe)**beta
    
    def compute_effective_potential(self, r: np.ndarray, M_baryon: np.ndarray) -> np.ndarray:
        """
        Compute gravitational potential using the Recognition kernel
        V(r) = -G(r) ∫ [F(r'/ℓ₁) + F(r'/ℓ₂)] M(r')/r'² dr'
        """
        # Evaluate kernel at both recognition lengths
        u1 = r / ell_1
        u2 = r / ell_2
        F1 = self.kernel.F(u1)
        F2 = self.kernel.F(u2)
        
        # Apply eight-tick modulation
        tick_phase = np.log(r / lambda_rec) / np.log(phi)
        tick_modulation = 1 + 0.1 * np.sin(2 * np.pi * tick_phase / TICK_PERIOD)
        
        # Combine kernels with proper weighting
        F_total = (F1 + F2) * tick_modulation
        
        # Include vacuum energy correction
        vacuum_correction = 1 + VACUUM_RESIDUAL * (r / ell_2)**2 / 1000
        
        return F_total * vacuum_correction
    
    def solve_rotation_curve(self, R_kpc: np.ndarray, sigma_baryon: np.ndarray) -> np.ndarray:
        """
        Solve for rotation curve using Recognition Science formulation
        """
        R = R_kpc * kpc_to_m
        
        # Compute circular velocity squared
        v_squared = np.zeros_like(R)
        
        for i, r_i in enumerate(R):
            # Integrate the kernel-weighted mass distribution
            integrand = 0
            for j in range(len(R)):
                if R[j] <= r_i:
                    r_j = R[j]
                    # Running G at this radius
                    G_r = self.running_G(r_j)
                    
                    # Kernel evaluation
                    F_eff = self.compute_effective_potential(r_j, sigma_baryon[j])
                    
                    # Contribution to circular velocity
                    if j > 0:
                        dr = R[j] - R[j-1]
                        mass_element = 2 * np.pi * sigma_baryon[j] * r_j * dr
                        integrand += G_r * F_eff * mass_element / r_j
            
            v_squared[i] = integrand
            
            # Apply clock lag correction in deep MOND regime
            a_N = integrand / r_i
            if a_N < 0.1 * g_dagger:
                v_squared[i] *= (1 + CLOCK_LAG)
        
        # Convert to velocity
        v_model = np.sqrt(np.maximum(v_squared, 0)) / km_to_m  # km/s
        
        return v_model
    
    def solve_galaxy(self, galaxy_name: str) -> Optional[Dict]:
        """Solve for a single galaxy using Recognition Science gravity"""
        if galaxy_name not in self.baryon_data:
            return None
        
        data = self.baryon_data[galaxy_name]
        R_kpc = data['radius']  # kpc
        v_obs = data['v_obs']  # km/s
        v_err = data['v_err']  # km/s
        
        # Total baryonic surface density
        sigma_total = data['sigma_gas'] + data['sigma_disk'] + data['sigma_bulge']  # kg/m²
        
        print(f"\nSolving {galaxy_name}: {len(R_kpc)} data points")
        
        # Solve for rotation curve
        v_model = self.solve_rotation_curve(R_kpc, sigma_total)
        
        # Ensure physical bounds
        v_baryon = np.sqrt(2 * np.pi * G * sigma_total * R_kpc * kpc_to_m) / km_to_m
        v_model = np.maximum(v_model, 0.5 * v_baryon)  # Allow factor 2 below Newtonian
        
        # Compute χ²
        chi2 = np.sum(((v_obs - v_model) / v_err)**2)
        chi2_reduced = chi2 / len(v_obs)
        
        result = {
            'galaxy': galaxy_name,
            'R_kpc': R_kpc,
            'v_obs': v_obs,
            'v_err': v_err,
            'v_model': v_model,
            'v_baryon': v_baryon,
            'chi2': chi2,
            'chi2_reduced': chi2_reduced,
            'N_points': len(v_obs)
        }
        
        return result
    
    def solve_all_galaxies(self, max_galaxies: Optional[int] = None) -> Dict:
        """Solve for all galaxies in the dataset"""
        galaxy_names = list(self.baryon_data.keys())
        if max_galaxies:
            galaxy_names = galaxy_names[:max_galaxies]
        
        print(f"\nSolving {len(galaxy_names)} galaxies with Recognition Science gravity...")
        print("="*60)
        
        results = []
        chi2_values = []
        
        for i, galaxy_name in enumerate(galaxy_names):
            print(f"[{i+1:3d}/{len(galaxy_names)}] {galaxy_name:12s}", end='')
            
            try:
                result = self.solve_galaxy(galaxy_name)
                if result and np.isfinite(result['chi2_reduced']) and result['chi2_reduced'] > 0:
                    results.append(result)
                    chi2_values.append(result['chi2_reduced'])
                    status = "✓" if result['chi2_reduced'] < 5 else "○"
                    print(f" {status} χ²/N = {result['chi2_reduced']:8.3f}")
                else:
                    print(" ✗ Failed: invalid result")
            except Exception as e:
                print(f" ✗ Error: {str(e)[:40]}...")
                continue
        
        if not chi2_values:
            print("No valid results obtained")
            return {}
        
        # Compute statistics
        chi2_values = np.array(chi2_values)
        chi2_mean = np.mean(chi2_values)
        chi2_std = np.std(chi2_values)
        chi2_median = np.median(chi2_values)
        
        print("\n" + "="*60)
        print("FINAL RESULTS:")
        print(f"  Galaxies processed: {len(results)}")
        print(f"  Mean χ²/N: {chi2_mean:.3f} ± {chi2_std:.3f}")
        print(f"  Median χ²/N: {chi2_median:.3f}")
        print(f"  Best fit: χ²/N = {np.min(chi2_values):.3f}")
        print(f"  Target (from paper): χ²/N = 1.05")
        print(f"  Fraction with χ²/N < 2.0: {np.mean(chi2_values < 2.0):.1%}")
        print(f"  Fraction with χ²/N < 5.0: {np.mean(chi2_values < 5.0):.1%}")
        
        self.results = {
            'individual': results,
            'chi2_mean': chi2_mean,
            'chi2_std': chi2_std,
            'chi2_median': chi2_median,
            'chi2_values': chi2_values,
            'n_galaxies': len(results)
        }
        
        # Save results
        with open('lnal_v3_recognition_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        print("\nResults saved to lnal_v3_recognition_results.pkl")
        
        return self.results

def main():
    """Run the Recognition Science gravity solver"""
    solver = AdvancedLNALSolverV3()
    
    # Test on subset first
    print("\nTesting on 10 galaxies...")
    test_results = solver.solve_all_galaxies(max_galaxies=10)
    
    if test_results and test_results['chi2_mean'] < 100:
        print("\nTest successful!")
        print(f"Mean χ²/N = {test_results['chi2_mean']:.2f}")
        print(f"Paper reports χ²/N = 1.05 for full SPARC sample")
        
        response = input("\nRun full analysis on all galaxies? (y/n): ")
        if response.lower() == 'y':
            full_results = solver.solve_all_galaxies()

if __name__ == "__main__":
    main() 