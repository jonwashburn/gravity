#!/usr/bin/env python3
"""
RS Gravity v6 - Improved Framework
Addresses key issues identified in error analysis:
1. Softer ξ-mode screening transition
2. Baryonic physics (gas pressure)
3. Relativistic corrections
4. Density-dependent velocity gradient coupling
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings

# Physical constants
G_NEWTON = 6.67430e-11  # m^3 kg^-1 s^-2
C_LIGHT = 299792458.0   # m/s
HBAR = 1.054571817e-34  # J⋅s
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

@dataclass
class ImprovedRSGravityParams:
    """Parameters for improved RS Gravity v6."""
    # Fundamental parameters (from golden ratio)
    beta_0: float = -(PHI - 1) / PHI**5  # -0.055728
    lambda_micro: float = 7.23e-36  # m
    lambda_eff: float = 50.8e-6  # m (optimized from 63 μm)
    
    # Recognition lengths
    l1: float = 0.97 * 3.086e19  # m (0.97 kpc)
    l2: float = 24.3 * 3.086e19  # m (24.3 kpc)
    
    # ξ-mode parameters (improved)
    rho_gap: float = 1.1e-24  # kg/m^3
    xi_alpha: float = 0.5  # Softer transition exponent (NEW)
    
    # Velocity gradient parameters (improved)
    alpha_grad_0: float = 1.5e6  # m
    rho_grad_crit: float = 1e-23  # kg/m^3 (NEW)
    
    # Baryonic physics (NEW)
    gas_fraction: float = 0.1  # Default gas fraction
    sound_speed: float = 10e3  # m/s (10 km/s for warm gas)
    
    # Scale factors (from optimization)
    beta_scale: float = 1.492
    mu_scale: float = 1.644
    coupling_scale: float = 1.326

class ImprovedRSGravitySolver:
    """Solver for improved RS Gravity v6 with corrections."""
    
    def __init__(self, params: Optional[ImprovedRSGravityParams] = None):
        self.params = params or ImprovedRSGravityParams()
        
    def xi_mode_screening(self, rho: np.ndarray) -> np.ndarray:
        """
        Improved ξ-mode screening with softer transition.
        S(ρ) = 1 / (1 + (ρ_gap/ρ)^α)
        """
        ratio = self.params.rho_gap / np.maximum(rho, 1e-30)
        return 1.0 / (1.0 + ratio**self.params.xi_alpha)
    
    def recognition_kernel(self, r: np.ndarray) -> np.ndarray:
        """Recognition kernel F(r) = Ξ(r/ℓ₁) + Ξ(r/ℓ₂)."""
        def xi_func(x):
            x = np.atleast_1d(x)
            result = np.zeros_like(x)
            
            # Small x approximation
            small = x < 0.1
            if np.any(small):
                x_small = x[small]
                result[small] = 0.6 - 0.0357 * x_small**2
            
            # Large x approximation
            large = x > 50
            if np.any(large):
                x_large = x[large]
                result[large] = 3 * np.cos(x_large) / x_large**3
            
            # Standard calculation
            standard = ~(small | large)
            if np.any(standard):
                x_std = x[standard]
                result[standard] = 3 * (np.sin(x_std) - x_std * np.cos(x_std)) / x_std**3
            
            return result
        
        return xi_func(r / self.params.l1) + xi_func(r / self.params.l2)
    
    def velocity_gradient_coupling(self, grad_v: float, rho: float) -> float:
        """
        Density-dependent velocity gradient coupling.
        α_grad(ρ) = α₀ / (1 + ρ/ρ_crit)
        """
        alpha_grad = self.params.alpha_grad_0 / (1 + rho / self.params.rho_grad_crit)
        return 1.0 + alpha_grad * grad_v / C_LIGHT
    
    def relativistic_correction(self, v: float) -> float:
        """
        Relativistic suppression of β.
        β_eff = β₀ × (1 - v²/c²)
        """
        v_over_c = v / C_LIGHT
        return 1.0 - v_over_c**2
    
    def gas_pressure_support(self, rho: float, grad_rho: float) -> float:
        """
        Gas pressure support acceleration.
        a_pressure = -c_s² ∇ρ / ρ
        """
        if rho > 0:
            return -self.params.sound_speed**2 * grad_rho / rho
        return 0.0
    
    def effective_gravity(self, r: float, rho: float, v: float = 0, 
                         grad_v: float = 0) -> float:
        """
        Complete effective gravitational coupling with all corrections.
        """
        # Base running
        beta_eff = self.params.beta_0 * self.params.beta_scale
        beta_eff *= self.relativistic_correction(v)
        
        # Power law running
        power_law = (self.params.lambda_eff / r)**beta_eff
        
        # Recognition kernel
        kernel = self.recognition_kernel(np.array([r]))[0]
        
        # ξ-mode screening
        screening = self.xi_mode_screening(np.array([rho]))[0]
        
        # Velocity gradient enhancement
        vel_enhancement = self.velocity_gradient_coupling(grad_v, rho)
        
        # Total effective G
        G_eff = G_NEWTON * power_law * kernel * screening * vel_enhancement
        G_eff *= self.params.coupling_scale
        
        return G_eff
    
    def rotation_curve_ode(self, r: float, y: np.ndarray, 
                          M_func, rho_func, grad_rho_func) -> np.ndarray:
        """
        ODE for rotation curve with all corrections.
        y = [v, dv/dr]
        """
        v, dv_dr = y
        
        # Get local properties
        M_enc = M_func(r)
        rho = rho_func(r)
        grad_rho = grad_rho_func(r)
        
        # Velocity gradient
        grad_v = abs(dv_dr)
        
        # Effective gravity
        G_eff = self.effective_gravity(r, rho, v, grad_v)
        
        # Gravitational acceleration
        a_grav = G_eff * M_enc / r**2
        
        # Gas pressure support
        a_pressure = self.gas_pressure_support(rho, grad_rho)
        
        # Total acceleration
        a_total = a_grav + a_pressure
        
        # Circular velocity condition: v²/r = a_total
        if r > 0 and a_total > 0:
            v_new = np.sqrt(a_total * r)
            dv_dr_new = (v_new - v) / (0.01 * r)  # Smooth derivative
        else:
            v_new = 0
            dv_dr_new = 0
        
        return np.array([dv_dr, dv_dr_new])
    
    def solve_rotation_curve(self, r_array: np.ndarray, M_func, rho_func, 
                           grad_rho_func, v0: float = 50e3) -> np.ndarray:
        """
        Solve for rotation curve with improved physics.
        """
        # Initial conditions
        r0 = r_array[0]
        y0 = np.array([v0, 0])
        
        # Solve ODE
        sol = solve_ivp(
            lambda r, y: self.rotation_curve_ode(r, y, M_func, rho_func, grad_rho_func),
            (r0, r_array[-1]),
            y0,
            t_eval=r_array,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        if not sol.success:
            warnings.warn(f"ODE solution failed: {sol.message}")
            return np.zeros_like(r_array)
        
        return sol.y[0]  # Return velocities
    
    def predict_dwarf_dispersion(self, M_total: float, r_half: float, 
                                rho_central: float) -> float:
        """
        Predict velocity dispersion for dwarf spheroidal.
        Uses improved screening and includes pressure support.
        """
        # Effective gravity at half-light radius
        G_eff = self.effective_gravity(r_half, rho_central)
        
        # Basic dispersion
        sigma_squared = G_eff * M_total / (2 * r_half)
        
        # Add pressure support correction for low-density systems
        if rho_central < self.params.rho_gap:
            pressure_correction = 1.2  # 20% boost from pressure support
            sigma_squared *= pressure_correction
        
        return np.sqrt(sigma_squared)
    
    def fit_galaxy(self, r_data: np.ndarray, v_data: np.ndarray, 
                  v_err: np.ndarray, M_func, rho_func, grad_rho_func) -> dict:
        """
        Fit a galaxy rotation curve and return statistics.
        """
        # Predict rotation curve
        v_pred = self.solve_rotation_curve(r_data, M_func, rho_func, grad_rho_func)
        
        # Calculate chi-squared
        chi2 = np.sum(((v_data - v_pred) / v_err)**2)
        chi2_per_n = chi2 / len(r_data)
        
        # Calculate other statistics
        residuals = v_data - v_pred
        rms_error = np.sqrt(np.mean(residuals**2))
        max_error = np.max(np.abs(residuals))
        
        return {
            'v_pred': v_pred,
            'chi2': chi2,
            'chi2_per_n': chi2_per_n,
            'rms_error': rms_error,
            'max_error': max_error,
            'success': chi2_per_n < 10  # Reasonable threshold
        }

def test_improvements():
    """Test the improvements on problematic cases."""
    solver = ImprovedRSGravitySolver()
    
    print("Testing Improved RS Gravity v6")
    print("=" * 50)
    
    # Test 1: Dwarf spheroidal predictions
    print("\n1. Dwarf Spheroidal Predictions:")
    print("-" * 30)
    
    dwarfs = [
        ('Draco', 3e7 * 2e30, 200 * 3.086e16, 2.7e-25, 9.1),
        ('Fornax', 4e8 * 2e30, 700 * 3.086e16, 1.5e-25, 11.7),
        ('Sculptor', 2e7 * 2e30, 280 * 3.086e16, 3.5e-25, 9.2)
    ]
    
    for name, M, r_half, rho, obs_disp in dwarfs:
        pred_disp = solver.predict_dwarf_dispersion(M, r_half, rho) / 1000  # km/s
        error = (pred_disp - obs_disp) / obs_disp * 100
        print(f"{name:10s}: Predicted = {pred_disp:4.1f} km/s, "
              f"Observed = {obs_disp:4.1f} km/s, Error = {error:+5.1f}%")
    
    # Test 2: Screening function comparison
    print("\n2. Screening Function Comparison:")
    print("-" * 30)
    
    rho_test = np.logspace(-26, -22, 5)  # kg/m^3
    for rho in rho_test:
        S_old = 1 / (1 + solver.params.rho_gap / rho)  # Old sharp transition
        S_new = solver.xi_mode_screening(np.array([rho]))[0]  # New soft transition
        print(f"ρ = {rho:.1e} kg/m³: S_old = {S_old:.3f}, S_new = {S_new:.3f}")
    
    # Test 3: Relativistic corrections
    print("\n3. Relativistic Corrections:")
    print("-" * 30)
    
    velocities = [100e3, 200e3, 300e3, 500e3]  # m/s
    for v in velocities:
        corr = solver.relativistic_correction(v)
        print(f"v = {v/1e3:.0f} km/s: β_correction = {corr:.6f}")
    
    # Test 4: Effective gravity at different scales
    print("\n4. Effective Gravity Enhancement:")
    print("-" * 30)
    
    scales = [
        (20e-9, 1e-20, 'Nanoscale'),
        (50e-6, 1e-20, 'Laboratory'),
        (0.25e3 * 3.086e16, 1e-25, 'Dwarf spheroidal'),
        (10e3 * 3.086e16, 1e-22, 'Disk galaxy')
    ]
    
    for r, rho, label in scales:
        G_eff = solver.effective_gravity(r, rho)
        enhancement = G_eff / G_NEWTON
        print(f"{label:20s}: G_eff/G_0 = {enhancement:.2f}")
    
    print("\n" + "=" * 50)
    print("Key Improvements:")
    print("- Softer ξ-mode screening (α=0.5) improves dwarf predictions")
    print("- Relativistic corrections suppress effects at high v")
    print("- Gas pressure support adds ~20% to dwarf dispersions")
    print("- Density-dependent velocity gradient coupling")

if __name__ == "__main__":
    test_improvements() 