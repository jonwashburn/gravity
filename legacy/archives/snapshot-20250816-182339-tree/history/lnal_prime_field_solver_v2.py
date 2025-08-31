#!/usr/bin/env python3
"""
LNAL Prime Recognition Gravity: Refined Field Solver v2.0
Solves the prime-balanced information field equation with:
1. Exact baryonic sources B(R) from SPARC data
2. Scale-dependent prime coupling α_p(R) = α₀[1 + φ⁻²e^(-R/ℓ₂)]
3. Renormalized kernel K(R,R') for prime interactions

Target: χ²/N ≈ 1.02 ± 0.05 across 135 SPARC galaxies
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import pickle
import os
from lnal_prime_exact_baryon_parser import SPARCBaryonParser

# Physical constants
c = 2.998e8  # m/s
G = 6.674e-11  # m³/kg/s²
hbar = 1.055e-34  # J⋅s
kpc_to_m = 3.086e19  # m/kpc

# Recognition Science parameters (fixed by theory)
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
ell_1 = 0.97 * kpc_to_m  # Primary recognition length (m)
ell_2 = 24.3 * kpc_to_m  # Secondary recognition length (m)
tau_0 = 7.33e-15  # Fundamental time scale (s)
g_dagger = 1.2e-10  # Universal acceleration scale (m/s²)

# Derived parameters
I_star = 4.0e18  # Information capacity scale (J/m³)
mu = hbar / (c * ell_1)  # Field mass (1/m²)
lambda_coupling = np.sqrt(g_dagger * c**2 / I_star)  # Field-baryon coupling
alpha_0 = 1 / (phi - 1)  # Base prime coupling strength
epsilon = phi**(-2)  # Inner enhancement factor

class PrimeFieldSolver:
    """Solve prime-balanced information field equation"""
    
    def __init__(self, baryon_data_file='sparc_exact_baryons.pkl'):
        self.baryon_data = self.load_baryon_data(baryon_data_file)
        self.results = {}
    
    def load_baryon_data(self, filename):
        """Load exact baryonic source data"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded exact baryon data for {len(data)} galaxies")
            return data
        else:
            print(f"Baryon data file {filename} not found. Generating...")
            parser = SPARCBaryonParser()
            data = parser.process_all_galaxies()
            parser.save_results(data, filename)
            return data
    
    def scale_dependent_coupling(self, R):
        """Scale-dependent prime coupling α_p(R)"""
        R_m = R * kpc_to_m  # Convert to meters
        return alpha_0 * (1 + epsilon * np.exp(-R_m / ell_2))
    
    def prime_kernel(self, R, R_prime):
        """Prime interaction kernel K(R,R')"""
        # Simplified form: exponentially decaying with prime-number oscillations
        delta_R = abs(R - R_prime) * kpc_to_m
        kernel = np.exp(-delta_R / ell_1) * np.cos(np.pi * np.sqrt(R * R_prime))
        return kernel / (1 + R * R_prime)  # Dimensional regularization
    
    def mond_interpolation(self, x):
        """MOND interpolation function μ(x) = x/√(1+x²)"""
        return x / np.sqrt(1 + x**2)
    
    def field_equation_residual(self, R, I, dI_dR, B_R):
        """Residual of the prime-balanced field equation"""
        # Convert to SI units
        R_m = R * kpc_to_m
        
        # Gradient magnitude for MOND interpolation
        grad_I_mag = abs(dI_dR) / kpc_to_m  # Convert to SI
        x = grad_I_mag / I_star
        mu_interp = self.mond_interpolation(x)
        
        # Laplacian in spherical coordinates: ∇²I = d²I/dR² + (2/R)dI/dR
        d2I_dR2 = np.gradient(dI_dR, R)  # Numerical second derivative
        laplacian_I = d2I_dR2 + (2 / R) * dI_dR
        
        # Scale-dependent coupling
        alpha_R = self.scale_dependent_coupling(R)
        
        # Prime interaction integral (simplified to local approximation)
        prime_interaction = alpha_R * I * np.exp(-R_m / ell_1)
        
        # Field equation: ∇·[μ(|∇I|/I*)∇I] - μ²I + αI∫K = -λB
        residual = (mu_interp * laplacian_I - mu**2 * I + 
                   prime_interaction + lambda_coupling * B_R)
        
        return residual
    
    def solve_galaxy(self, galaxy_name, plot=False):
        """Solve field equation for a single galaxy"""
        if galaxy_name not in self.baryon_data:
            print(f"No baryon data for {galaxy_name}")
            return None
        
        data = self.baryon_data[galaxy_name]
        R = data['radius']  # kpc
        B_R = data['B_R']   # J/m³
        v_obs = data['v_obs']  # km/s
        v_err = data['v_err']  # km/s
        
        # Boundary conditions: I(0) finite, I(∞) → 0
        R_max = R[-1] * 1.5  # Extend slightly beyond data
        R_solve = np.linspace(0.1, R_max, 200)  # Avoid R=0 singularity
        
        # Interpolate baryonic source onto solver grid
        B_interp = interp1d(R, B_R, bounds_error=False, fill_value=0)
        B_solve = B_interp(R_solve)
        
        # Initial guess: Yukawa-like solution
        I_guess = np.exp(-R_solve / (ell_1 / kpc_to_m)) * B_solve.max() / mu**2
        
        # Solve using iterative approach (simplified for speed)
        I_solution = self.iterative_solve(R_solve, B_solve, I_guess)
        
        # Compute information acceleration
        dI_dR = np.gradient(I_solution, R_solve)
        a_info = lambda_coupling * dI_dR / c**2  # m/s²
        
        # Convert to velocity contribution
        v_info_squared = a_info * R_solve * kpc_to_m  # (m/s)²
        v_info = np.sqrt(np.maximum(v_info_squared, 0)) / 1000  # km/s
        
        # Interpolate back to observation points
        v_info_interp = interp1d(R_solve, v_info, bounds_error=False, fill_value=0)
        v_model = v_info_interp(R)
        
        # Compute χ²
        chi2 = np.sum(((v_obs - v_model) / v_err)**2)
        chi2_reduced = chi2 / len(v_obs)
        
        result = {
            'galaxy': galaxy_name,
            'R': R,
            'v_obs': v_obs,
            'v_err': v_err,
            'v_model': v_model,
            'I_field': I_solution,
            'R_solve': R_solve,
            'chi2': chi2,
            'chi2_reduced': chi2_reduced,
            'N_points': len(v_obs)
        }
        
        if plot:
            self.plot_galaxy_fit(result)
        
        return result
    
    def iterative_solve(self, R, B_R, I_initial, max_iter=20, tolerance=1e-6):
        """Iterative solver for nonlinear field equation"""
        I = I_initial.copy()
        
        for iteration in range(max_iter):
            I_old = I.copy()
            
            # Update field using linearized equation
            for i in range(1, len(R)-1):
                # Central differences for derivatives
                dI_dR = (I[i+1] - I[i-1]) / (2 * (R[i+1] - R[i-1]))
                d2I_dR2 = (I[i+1] - 2*I[i] + I[i-1]) / ((R[i+1] - R[i-1])/2)**2
                
                # MOND interpolation
                grad_I_mag = abs(dI_dR) / kpc_to_m
                x = grad_I_mag / I_star
                mu_interp = self.mond_interpolation(x)
                
                # Scale-dependent coupling
                alpha_R = self.scale_dependent_coupling(R[i])
                
                # Update equation (simplified)
                laplacian = d2I_dR2 + (2 / R[i]) * dI_dR
                source = -lambda_coupling * B_R[i]
                damping = mu**2 + alpha_R * np.exp(-(R[i] * kpc_to_m) / ell_1)
                
                # Semi-implicit update
                I[i] = (I[i] + 0.1 * (mu_interp * laplacian + source)) / (1 + 0.1 * damping)
            
            # Apply boundary conditions
            I[0] = I[1]  # Finite at origin
            I[-1] = I[-2] * np.exp(-(R[-1] - R[-2]) * kpc_to_m / ell_1)  # Decay at infinity
            
            # Check convergence
            relative_change = np.max(np.abs(I - I_old) / (np.abs(I_old) + 1e-10))
            if relative_change < tolerance:
                print(f"  Converged after {iteration+1} iterations")
                break
        else:
            print(f"  Warning: Did not converge after {max_iter} iterations")
        
        return I
    
    def plot_galaxy_fit(self, result):
        """Plot galaxy rotation curve fit"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Rotation curve
        ax1.errorbar(result['R'], result['v_obs'], yerr=result['v_err'], 
                    fmt='ko', alpha=0.7, label='Observed')
        ax1.plot(result['R'], result['v_model'], 'r-', linewidth=2, 
                label=f'LNAL Prime (χ²/N = {result["chi2_reduced"]:.2f})')
        
        ax1.set_xlabel('Radius (kpc)')
        ax1.set_ylabel('Velocity (km/s)')
        ax1.set_title(f'{result["galaxy"]} Rotation Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Information field
        ax2.semilogy(result['R_solve'], result['I_field'], 'b-', linewidth=2)
        ax2.set_xlabel('Radius (kpc)')
        ax2.set_ylabel('Information Field I(R) (J/m³)')
        ax2.set_title('Prime Recognition Field')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'lnal_prime_v2_{result["galaxy"]}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def solve_all_galaxies(self, max_galaxies=None, save_results=True):
        """Solve field equation for all galaxies"""
        galaxy_names = list(self.baryon_data.keys())
        if max_galaxies:
            galaxy_names = galaxy_names[:max_galaxies]
        
        print(f"Solving prime field equation for {len(galaxy_names)} galaxies...")
        
        results = []
        chi2_values = []
        
        for i, galaxy_name in enumerate(galaxy_names):
            print(f"[{i+1}/{len(galaxy_names)}] Solving {galaxy_name}...")
            
            try:
                result = self.solve_galaxy(galaxy_name)
                if result:
                    results.append(result)
                    chi2_values.append(result['chi2_reduced'])
                    print(f"  χ²/N = {result['chi2_reduced']:.3f}")
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        # Compute overall statistics
        chi2_mean = np.mean(chi2_values)
        chi2_std = np.std(chi2_values)
        
        print(f"\nOverall Results:")
        print(f"Galaxies processed: {len(results)}")
        print(f"Mean χ²/N: {chi2_mean:.3f} ± {chi2_std:.3f}")
        print(f"Median χ²/N: {np.median(chi2_values):.3f}")
        print(f"Range: {np.min(chi2_values):.3f} - {np.max(chi2_values):.3f}")
        
        self.results = {
            'individual': results,
            'chi2_mean': chi2_mean,
            'chi2_std': chi2_std,
            'chi2_values': chi2_values,
            'n_galaxies': len(results)
        }
        
        if save_results:
            with open('lnal_prime_v2_results.pkl', 'wb') as f:
                pickle.dump(self.results, f)
            print("Results saved to lnal_prime_v2_results.pkl")
        
        return self.results
    
    def plot_chi2_distribution(self):
        """Plot χ² distribution"""
        if not self.results:
            print("No results to plot")
            return
        
        chi2_values = self.results['chi2_values']
        
        plt.figure(figsize=(10, 6))
        plt.hist(chi2_values, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect fit (χ²/N = 1)')
        plt.axvline(np.mean(chi2_values), color='blue', linestyle='-', linewidth=2, 
                   label=f'Mean = {np.mean(chi2_values):.3f}')
        
        plt.xlabel('χ²/N')
        plt.ylabel('Number of Galaxies')
        plt.title('LNAL Prime Recognition: χ² Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('lnal_prime_v2_chi2_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main execution"""
    solver = PrimeFieldSolver()
    
    # Test on a few galaxies first
    print("Testing on 5 galaxies...")
    test_results = solver.solve_all_galaxies(max_galaxies=5)
    
    if test_results['chi2_mean'] < 2.0:  # Reasonable threshold
        print("Test successful. Running full analysis...")
        full_results = solver.solve_all_galaxies()
        solver.plot_chi2_distribution()
    else:
        print(f"Test failed (χ²/N = {test_results['chi2_mean']:.3f}). Check implementation.")

if __name__ == "__main__":
    main() 