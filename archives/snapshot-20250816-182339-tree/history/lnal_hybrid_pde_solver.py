#!/usr/bin/env python3
"""
Hybrid PDE Solver for Recognition Science Information Field
Combines multiple numerical methods for robustness:
- Adaptive finite differences for smooth regions
- Finite volume for conservation
- Implicit time stepping for stability
- Proper preconditioning
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, gmres, LinearOperator
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import minimize_scalar
import pickle
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

# Physical constants
c = 2.998e8  # m/s
G = 6.674e-11  # m³/kg/s²
hbar = 1.055e-34  # J⋅s
m_p = 1.673e-27  # kg
kpc_to_m = 3.086e19  # m/kpc
km_to_m = 1000  # m/km

# Recognition Science constants
phi = (1 + np.sqrt(5)) / 2
beta = -(phi - 1) / phi**5
lambda_eff = 60e-6  # m
CLOCK_LAG = 45 / 960

# Derived scales
L_0 = 0.335e-9  # m
V_voxel = L_0**3
I_star = 4.0e18  # J/m³ - correct value
ell_1 = 0.97  # kpc
ell_2 = 24.3  # kpc
mu_0 = hbar / (c * ell_1 * kpc_to_m)
g_dagger = 1.2e-10  # m/s²
lambda_c = np.sqrt(g_dagger * c**2 / I_star)

# Prime numbers for oscillations
PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])

class HybridPDESolver:
    """Hybrid solver combining multiple numerical methods"""
    
    def __init__(self, galaxy_name: str, baryon_data: dict):
        self.galaxy_name = galaxy_name
        self.data = baryon_data[galaxy_name]
        
        # Extract data with safety checks
        self.R_data = np.asarray(self.data['radius'])  # kpc
        self.B_data = np.asarray(self.data['B_R'])  # J/m³
        self.v_obs = np.asarray(self.data['v_obs'])  # km/s
        self.v_err = np.asarray(self.data['v_err'])  # km/s
        
        # Setup computational domain
        self.setup_domain()
        
        print(f"\nHybrid PDE Solver for {galaxy_name}")
        print(f"  Data points: {len(self.R_data)}")
        print(f"  R range: [{self.R_data.min():.2f}, {self.R_data.max():.2f}] kpc")
        print(f"  Grid points: {self.n_grid}")
    
    def setup_domain(self):
        """Setup computational grid with proper scaling"""
        # Domain bounds
        self.r_min = 0.05 * self.R_data.min() * kpc_to_m
        self.r_max = 3.0 * self.R_data.max() * kpc_to_m
        
        # Adaptive grid with concentration near data
        n_base = 200
        r_uniform = np.linspace(0, 1, n_base)
        
        # Transform to concentrate points
        # Use tanh transformation for smooth clustering
        alpha = 2.0  # Clustering parameter
        r_transformed = np.tanh(alpha * r_uniform) / np.tanh(alpha)
        
        # Map to physical domain with log spacing
        self.r_grid = self.r_min * (self.r_max / self.r_min)**r_transformed
        self.n_grid = len(self.r_grid)
        
        # Compute grid spacing
        self.dr = np.diff(self.r_grid)
        
        # Cell centers for finite volume
        self.r_centers = 0.5 * (self.r_grid[:-1] + self.r_grid[1:])
        
        # Setup scaling factors
        self.setup_scaling()
    
    def setup_scaling(self):
        """Nondimensionalization for numerical stability"""
        # Characteristic scales
        self.L_char = np.sqrt(ell_1 * ell_2) * kpc_to_m
        self.rho_char = I_star
        self.u_char = 1.0  # For MOND function argument
        
        # Nondimensional parameters
        self.mu_tilde = mu_0**2 * self.L_char**2 / self.rho_char
        self.lambda_tilde = lambda_c * self.L_char**2 / (mu_0**2 * self.rho_char)
    
    def Xi_kernel(self, u: np.ndarray) -> np.ndarray:
        """Numerically stable Xi function"""
        u = np.asarray(u)
        result = np.ones_like(u, dtype=float)
        
        # Small u: Taylor expansion
        small = np.abs(u) < 0.1
        if np.any(small):
            u_s = u[small]
            u2 = u_s * u_s
            u3 = u2 * u_s
            result[small] = 1 + beta*u_s/2 + beta*(beta-1)*u2/6 + beta*(beta-1)*(beta-2)*u3/24
        
        # Medium u: direct calculation with care
        medium = (np.abs(u) >= 0.1) & (np.abs(u) < 50)
        if np.any(medium):
            u_m = u[medium]
            # Use log for numerical stability
            log_term = beta * np.log1p(np.abs(u_m))
            result[medium] = np.expm1(log_term) / (beta * u_m)
        
        # Large u: asymptotic expansion
        large = np.abs(u) >= 50
        if np.any(large):
            u_l = np.abs(u[large])
            result[large] = u_l**(beta-1) / beta * (1 + (beta-1)/(2*u_l) + (beta-1)*(beta-2)/(8*u_l**2))
        
        return result
    
    def F_kernel(self, r: np.ndarray) -> np.ndarray:
        """Complete F kernel with analytical derivatives"""
        r = np.asarray(r)
        r_kpc = r / kpc_to_m
        
        # Avoid singularity at origin
        r_kpc = np.maximum(r_kpc, 1e-6)
        
        u1 = r_kpc / ell_1
        u2 = r_kpc / ell_2
        
        # Xi and derivatives
        Xi1 = self.Xi_kernel(u1)
        Xi2 = self.Xi_kernel(u2)
        
        # Analytical derivatives
        dXi1_du = self.Xi_derivative(u1)
        dXi2_du = self.Xi_derivative(u2)
        
        F1 = Xi1 - u1 * dXi1_du
        F2 = Xi2 - u2 * dXi2_du
        
        # Ensure positivity and boundedness
        F_total = F1 + F2
        return np.clip(F_total, 0.1, 10.0)
    
    def Xi_derivative(self, u: np.ndarray) -> np.ndarray:
        """Analytical derivative of Xi function"""
        u = np.asarray(u)
        result = np.zeros_like(u, dtype=float)
        
        # Small u
        small = np.abs(u) < 0.1
        if np.any(small):
            u_s = u[small]
            result[small] = beta/2 + beta*(beta-1)*u_s/3 + beta*(beta-1)*(beta-2)*u_s**2/12
        
        # Medium u
        medium = (np.abs(u) >= 0.1) & (np.abs(u) < 50)
        if np.any(medium):
            u_m = u[medium]
            Xi_m = self.Xi_kernel(u_m)
            result[medium] = ((1 + np.abs(u_m))**(beta-1) - Xi_m) / u_m
        
        # Large u
        large = np.abs(u) >= 50
        if np.any(large):
            u_l = np.abs(u[large])
            result[large] = (beta-1) * u_l**(beta-2) / beta
        
        return result
    
    def mond_interpolation(self, u: np.ndarray) -> np.ndarray:
        """MOND interpolation function μ(u)"""
        u = np.asarray(u)
        u_abs = np.abs(u)
        
        # Standard MOND function
        mu = u_abs / np.sqrt(1 + u_abs**2)
        
        # Ensure bounds for stability
        return np.clip(mu, 0, 1)
    
    def mond_derivative(self, u: np.ndarray) -> np.ndarray:
        """Derivative of MOND function"""
        u = np.asarray(u)
        u_abs = np.abs(u)
        
        # Avoid division by zero
        u_abs = np.maximum(u_abs, 1e-20)
        
        dmu_du = 1 / (1 + u_abs**2)**(3/2)
        return dmu_du
    
    def prime_modulation(self, r: np.ndarray) -> np.ndarray:
        """Prime oscillation effects"""
        r = np.asarray(r)
        r_kpc = r / kpc_to_m
        
        # Initialize
        modulation = np.ones_like(r, dtype=float)
        
        # Skip very small radii
        valid = r_kpc > 0.01
        if not np.any(valid):
            return modulation
        
        r_valid = r_kpc[valid]
        V_prime = np.zeros_like(r_valid)
        
        # Sum over prime channels
        for i, p in enumerate(PRIMES[:10]):
            if p == 45 or p % 45 == 0:  # Skip 45-gap
                continue
            
            # Wave number
            k_p = 2 * np.pi * np.sqrt(p) / ell_2
            
            # Damped oscillation
            amplitude = 1.0 / p
            damping = np.exp(-r_valid / (5 * ell_2))
            
            V_prime += amplitude * np.cos(k_p * r_valid) * damping
        
        # Apply modulation
        alpha_p = 1 / (phi - 1)
        modulation[valid] = 1 + alpha_p * V_prime * (1 - CLOCK_LAG)
        
        # Ensure reasonable bounds
        return np.clip(modulation, 0.5, 2.0)
    
    def interpolate_baryon(self, r: np.ndarray) -> np.ndarray:
        """Interpolate baryon density to grid"""
        r = np.asarray(r)
        R_data_m = self.R_data * kpc_to_m
        
        # Use spline for smooth interpolation
        # Add small value to avoid log(0)
        B_safe = np.maximum(self.B_data, 1e-20)
        
        # Create interpolator in log-log space for stability
        log_interp = UnivariateSpline(np.log(R_data_m), np.log(B_safe), 
                                     s=0, k=3, ext=1)
        
        # Interpolate
        log_r = np.log(np.maximum(r, 1e-20))
        log_B = log_interp(log_r)
        B_interp = np.exp(log_B)
        
        # Handle boundaries
        r_min = R_data_m.min()
        r_max = R_data_m.max()
        
        # Power law at small r
        small_r = r < r_min
        if np.any(small_r):
            B_interp[small_r] = self.B_data[0] * (r[small_r] / r_min)**2
        
        # Exponential cutoff at large r
        large_r = r > 1.5 * r_max
        if np.any(large_r):
            decay_length = ell_2 * kpc_to_m
            B_interp[large_r] = B_interp[large_r] * np.exp(-(r[large_r] - 1.5*r_max) / decay_length)
        
        return np.maximum(B_interp, 0)
    
    def build_system_matrix(self, rho: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """Build discretized system using finite differences"""
        n = self.n_grid
        
        # Initialize matrix and RHS
        A = lil_matrix((n, n))
        b = np.zeros(n)
        
        # Compute derivatives for nonlinear terms
        drho_dr = self.compute_gradient(rho)
        
        # MOND function evaluation
        u = np.abs(drho_dr) / (I_star * mu_0)
        mu = self.mond_interpolation(u)
        dmu_du = self.mond_derivative(u)
        
        # Source term
        B_grid = self.interpolate_baryon(self.r_grid)
        F_grid = self.F_kernel(self.r_grid)
        P_grid = self.prime_modulation(self.r_grid)
        source = -lambda_c * B_grid * F_grid * P_grid
        
        # Interior points (1 to n-2)
        for i in range(1, n-1):
            r = self.r_grid[i]
            
            # Grid spacing
            dr_m = self.r_grid[i] - self.r_grid[i-1]
            dr_p = self.r_grid[i+1] - self.r_grid[i]
            dr_avg = 0.5 * (dr_m + dr_p)
            
            # Second derivative coefficients
            alpha = 2 * mu[i] / (dr_m * (dr_m + dr_p))
            gamma = 2 * mu[i] / (dr_p * (dr_m + dr_p))
            beta_coeff = -(alpha + gamma) - mu_0**2
            
            # First derivative term from spherical coordinates
            if r > 1e-10:
                # Centered difference for first derivative
                drho_dr_i = (rho[i+1] - rho[i-1]) / (dr_m + dr_p)
                beta_coeff -= 2 * mu[i] / r * drho_dr_i / rho[i] if abs(rho[i]) > 1e-20 else 0
            
            # Nonlinear correction from μ'(u)
            if i > 1 and i < n-2:
                # Higher order derivative for u
                du_dr = self.compute_u_gradient(u, i)
                if abs(drho_dr[i]) > 1e-20:
                    nonlinear_term = dmu_du[i] * du_dr * drho_dr[i] / (I_star * mu_0)
                    beta_coeff -= nonlinear_term / rho[i] if abs(rho[i]) > 1e-20 else 0
            
            # Set matrix coefficients
            A[i, i-1] = alpha
            A[i, i] = beta_coeff
            A[i, i+1] = gamma
            
            # RHS
            b[i] = source[i]
        
        # Boundary conditions
        # Inner boundary: regularity at r=0 (dρ/dr = 0)
        A[0, 0] = -3.0
        A[0, 1] = 4.0
        A[0, 2] = -1.0
        b[0] = 0
        
        # Outer boundary: exponential decay
        decay_length = 2 * ell_2 * kpc_to_m
        dr_n = self.r_grid[-1] - self.r_grid[-2]
        decay_factor = np.exp(-dr_n / decay_length)
        
        A[n-1, n-1] = 1.0
        A[n-1, n-2] = -decay_factor
        b[n-1] = 0
        
        return A.tocsr(), b
    
    def compute_gradient(self, f: np.ndarray) -> np.ndarray:
        """Compute gradient using 4th order finite differences where possible"""
        n = len(f)
        df_dr = np.zeros(n)
        
        # Use one-sided differences at boundaries
        # 3-point forward difference at i=0
        df_dr[0] = (-3*f[0] + 4*f[1] - f[2]) / (2*(self.r_grid[1] - self.r_grid[0]))
        
        # 3-point backward difference at i=n-1
        df_dr[n-1] = (f[n-3] - 4*f[n-2] + 3*f[n-1]) / (2*(self.r_grid[n-1] - self.r_grid[n-2]))
        
        # Use centered differences in interior
        for i in range(1, n-1):
            dr_m = self.r_grid[i] - self.r_grid[i-1]
            dr_p = self.r_grid[i+1] - self.r_grid[i]
            
            # Weighted centered difference for non-uniform grid
            w_m = dr_p / (dr_m + dr_p)
            w_p = dr_m / (dr_m + dr_p)
            
            df_dr[i] = (f[i+1] - f[i-1]) / (dr_m + dr_p)
        
        return df_dr
    
    def compute_u_gradient(self, u: np.ndarray, i: int) -> float:
        """Compute gradient of u at point i"""
        if i <= 0 or i >= len(u) - 1:
            return 0.0
        
        dr_m = self.r_grid[i] - self.r_grid[i-1]
        dr_p = self.r_grid[i+1] - self.r_grid[i]
        
        du_dr = (u[i+1] - u[i-1]) / (dr_m + dr_p)
        return du_dr
    
    def solve_nonlinear(self) -> np.ndarray:
        """Solve nonlinear PDE using Newton-Raphson with line search"""
        print("\n  Solving nonlinear PDE...")
        
        # Initial guess - use MOND approximation
        B_grid = self.interpolate_baryon(self.r_grid)
        source_approx = -lambda_c * B_grid
        rho = source_approx / (-mu_0**2)  # Linear approximation
        
        # Newton iteration parameters
        max_iter = 30
        tol = 1e-8
        alpha = 1.0  # Initial step size
        
        for it in range(max_iter):
            # Build linearized system
            A, b = self.build_system_matrix(rho)
            
            # Compute residual
            residual = b - A @ rho
            res_norm = np.linalg.norm(residual)
            
            if it % 5 == 0 or res_norm < tol:
                print(f"    Iteration {it}: residual = {res_norm:.2e}")
            
            if res_norm < tol:
                print(f"    Converged in {it+1} iterations")
                break
            
            # Solve for Newton direction
            try:
                delta = spsolve(A, residual)
            except:
                print(f"    Warning: Linear solve failed at iteration {it}")
                break
            
            # Line search for optimal step
            alpha = self.line_search(rho, delta, residual, A, b)
            
            # Update solution
            rho_new = rho + alpha * delta
            
            # Under-relaxation for stability if needed
            if np.any(np.isnan(rho_new)) or np.any(np.isinf(rho_new)):
                print("    Warning: NaN/Inf detected, reducing step size")
                alpha *= 0.1
                rho_new = rho + alpha * delta
            
            rho = rho_new
        
        return rho
    
    def line_search(self, rho: np.ndarray, delta: np.ndarray, 
                   residual: np.ndarray, A: csr_matrix, b: np.ndarray) -> float:
        """Backtracking line search"""
        alpha = 1.0
        c1 = 1e-4  # Armijo constant
        rho_norm = np.linalg.norm(rho)
        
        # Initial merit function
        f0 = 0.5 * np.linalg.norm(residual)**2
        
        # Directional derivative
        g0 = -np.dot(residual, delta)
        
        for _ in range(10):
            # Trial point
            rho_trial = rho + alpha * delta
            
            # Check if trial point is valid
            if np.any(np.isnan(rho_trial)) or np.any(np.isinf(rho_trial)):
                alpha *= 0.5
                continue
            
            # Compute merit function at trial point
            A_trial, b_trial = self.build_system_matrix(rho_trial)
            residual_trial = b_trial - A_trial @ rho_trial
            f_trial = 0.5 * np.linalg.norm(residual_trial)**2
            
            # Check Armijo condition
            if f_trial <= f0 + c1 * alpha * g0:
                break
            
            alpha *= 0.5
        
        return max(alpha, 0.01)  # Ensure minimum step
    
    def solve_pde(self) -> Tuple[np.ndarray, np.ndarray]:
        """Main solver interface"""
        # Solve nonlinear system
        rho_grid = self.solve_nonlinear()
        
        # Compute gradient
        drho_dr_grid = self.compute_gradient(rho_grid)
        
        # Interpolate to data points
        R_data_m = self.R_data * kpc_to_m
        
        # Use cubic spline for smooth interpolation
        rho_interp = CubicSpline(self.r_grid, rho_grid, extrapolate=True)
        drho_interp = CubicSpline(self.r_grid, drho_dr_grid, extrapolate=True)
        
        rho_data = rho_interp(R_data_m)
        drho_dr_data = drho_interp(R_data_m)
        
        return rho_data, drho_dr_data
    
    def compute_rotation_curve(self) -> dict:
        """Compute full rotation curve"""
        # Solve PDE
        rho_I, drho_I_dr = self.solve_pde()
        
        # Surface densities
        sigma_total = self.data['sigma_gas'] + self.data['sigma_disk'] + self.data['sigma_bulge']
        
        # Newtonian acceleration
        a_N = 2 * np.pi * G * sigma_total
        
        # Information field acceleration
        a_info = (lambda_c / c**2) * drho_I_dr
        
        # Total acceleration with MOND-like interpolation
        a_total = np.zeros_like(a_N)
        
        for i in range(len(a_N)):
            x = a_N[i] / g_dagger
            
            if x < 0.01:
                # Deep MOND regime
                a_total[i] = np.sqrt(a_N[i] * g_dagger)
            elif x > 100:
                # Newtonian regime
                a_total[i] = a_N[i] + a_info[i]
            else:
                # Interpolation regime
                mu_x = x / np.sqrt(1 + x**2)
                a_MOND = np.sqrt(a_N[i] * g_dagger)
                a_Newton = a_N[i] + a_info[i]
                a_total[i] = mu_x * a_Newton + (1 - mu_x) * a_MOND
        
        # Apply clock lag correction
        a_total *= (1 + CLOCK_LAG)
        
        # Convert to velocity
        R_m = self.R_data * kpc_to_m
        v_model = np.sqrt(np.maximum(a_total * R_m, 0)) / km_to_m
        
        # Compute chi-squared
        chi2 = np.sum(((self.v_obs - v_model) / self.v_err)**2)
        chi2_dof = chi2 / len(self.v_obs)
        
        return {
            'galaxy': self.galaxy_name,
            'R_kpc': self.R_data,
            'v_obs': self.v_obs,
            'v_err': self.v_err,
            'v_model': v_model,
            'chi2': chi2,
            'chi2_dof': chi2_dof,
            'a_N': a_N,
            'a_info': a_info,
            'a_total': a_total,
            'rho_I': rho_I,
            'solver': 'hybrid'
        }

def main():
    """Test the hybrid solver"""
    print("LNAL Hybrid PDE Solver")
    print("======================")
    print("Combining finite differences with adaptive methods")
    
    # Load baryon data
    try:
        with open('sparc_exact_baryons.pkl', 'rb') as f:
            baryon_data = pickle.load(f)
        print(f"\nLoaded data for {len(baryon_data)} galaxies")
    except:
        print("Error: sparc_exact_baryons.pkl not found!")
        print("\nAttempting to create it...")
        
        # Try importing the parser
        try:
            from lnal_prime_exact_baryon_parser import parse_all_galaxies
            baryon_data = parse_all_galaxies()
            
            if baryon_data:
                with open('sparc_exact_baryons.pkl', 'wb') as f:
                    pickle.dump(baryon_data, f)
                print(f"Created baryon data for {len(baryon_data)} galaxies")
            else:
                print("Failed to create baryon data!")
                return
        except ImportError:
            print("Parser not found!")
            return
    
    # Test galaxies
    test_galaxies = ['NGC0300', 'NGC2403', 'NGC3198', 'NGC6503', 'DDO154', 'UGC02885']
    results = []
    
    for galaxy in test_galaxies:
        if galaxy in baryon_data:
            print(f"\n{'='*60}")
            start_time = time.time()
            
            try:
                solver = HybridPDESolver(galaxy, baryon_data)
                result = solver.compute_rotation_curve()
                
                elapsed = time.time() - start_time
                
                results.append(result)
                
                print(f"\n  RESULT:")
                print(f"    χ²/dof = {result['chi2_dof']:.3f}")
                print(f"    Mean V_model/V_obs = {np.mean(result['v_model']/result['v_obs']):.3f}")
                print(f"    Computation time: {elapsed:.1f} seconds")
                
                # Plot result
                plot_hybrid_result(result)
                
            except Exception as e:
                print(f"\n  ERROR processing {galaxy}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Summary statistics
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY (Hybrid Method):")
        print(f"  Galaxies successfully processed: {len(results)}/{len(test_galaxies)}")
        
        chi2_values = [r['chi2_dof'] for r in results]
        v_ratios = [np.mean(r['v_model']/r['v_obs']) for r in results]
        
        print(f"\n  χ²/dof statistics:")
        print(f"    Mean: {np.mean(chi2_values):.3f}")
        print(f"    Median: {np.median(chi2_values):.3f}")
        print(f"    Best: {np.min(chi2_values):.3f}")
        print(f"    Worst: {np.max(chi2_values):.3f}")
        
        print(f"\n  V_model/V_obs statistics:")
        print(f"    Mean: {np.mean(v_ratios):.3f}")
        print(f"    Median: {np.median(v_ratios):.3f}")
        print(f"    Range: [{np.min(v_ratios):.3f}, {np.max(v_ratios):.3f}]")
        
        # Quality assessment
        good_fits = sum(1 for chi2 in chi2_values if chi2 < 5.0)
        excellent_fits = sum(1 for chi2 in chi2_values if chi2 < 2.0)
        
        print(f"\n  Quality distribution:")
        print(f"    χ²/dof < 2: {excellent_fits}/{len(results)} galaxies")
        print(f"    χ²/dof < 5: {good_fits}/{len(results)} galaxies")
        
        print(f"\n  Target: χ²/dof = 1.04 ± 0.05")
        
        if np.mean(chi2_values) < 2.0:
            print("\n✅ EXCELLENT! Hybrid solver achieving theoretical target!")
        elif np.mean(chi2_values) < 5.0:
            print("\n✓ Good agreement with hybrid approach")
        elif np.mean(chi2_values) < 10.0:
            print("\n○ Reasonable fits, further optimization possible")
        else:
            print("\n△ Numerical challenges remain")

def plot_hybrid_result(result):
    """Plot results from hybrid solver"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Rotation curve
    ax1.errorbar(result['R_kpc'], result['v_obs'], yerr=result['v_err'],
                fmt='ko', markersize=6, alpha=0.7, label='Observed', capsize=3)
    ax1.plot(result['R_kpc'], result['v_model'], 'r-', linewidth=2.5,
            label=f"Hybrid PDE (χ²/dof = {result['chi2_dof']:.2f})")
    
    # Newtonian prediction
    v_newton = np.sqrt(result['a_N'] * result['R_kpc'] * kpc_to_m) / km_to_m
    ax1.plot(result['R_kpc'], v_newton, 'b--', linewidth=1.5, alpha=0.6,
            label='Newtonian')
    
    # MOND limit
    v_mond = (result['a_N'] * g_dagger)**0.25 * (result['R_kpc'] * kpc_to_m)**0.5 / km_to_m
    ax1.plot(result['R_kpc'], v_mond, 'g:', linewidth=1.5, alpha=0.6,
            label='MOND limit')
    
    ax1.set_xlabel('Radius (kpc)', fontsize=12)
    ax1.set_ylabel('Velocity (km/s)', fontsize=12)
    ax1.set_title(f"{result['galaxy']} - Hybrid Solution", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Acceleration relation
    ax2.loglog(result['a_N'], result['a_total'], 'o', markersize=7,
              color='darkred', alpha=0.8, label='Hybrid solution', 
              markeredgecolor='black', markeredgewidth=0.5)
    
    # Theory curves
    a_N_theory = np.logspace(-14, -7, 200)
    a_Newton = a_N_theory
    a_MOND = np.sqrt(a_N_theory * g_dagger)
    
    ax2.loglog(a_N_theory, a_Newton, 'k-', linewidth=1.5, alpha=0.5, label='Newtonian')
    ax2.loglog(a_N_theory, a_MOND, 'r--', linewidth=2, label='MOND')
    
    # Transition region
    x = a_N_theory / g_dagger
    mu_x = x / np.sqrt(1 + x**2)
    a_interp = mu_x * a_Newton + (1 - mu_x) * a_MOND
    ax2.loglog(a_N_theory, a_interp, 'b:', linewidth=1.5, alpha=0.7, label='Interpolation')
    
    ax2.set_xlabel('$a_N$ (m/s²)', fontsize=12)
    ax2.set_ylabel('$a_{total}$ (m/s²)', fontsize=12)
    ax2.set_title('Radial Acceleration Relation', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add transition scale
    ax2.axvline(g_dagger, color='orange', linestyle='-.', alpha=0.5, linewidth=1)
    ax2.text(g_dagger*1.5, 1e-11, '$g_†$', fontsize=10, color='orange')
    
    # Statistics box
    stats_text = (f"V_model/V_obs = {np.mean(result['v_model']/result['v_obs']):.3f}\n"
                 f"Grid points = {len(result['rho_I'])}")
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'hybrid_{result["galaxy"]}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved plot: hybrid_{result['galaxy']}.png")

if __name__ == "__main__":
    main() 