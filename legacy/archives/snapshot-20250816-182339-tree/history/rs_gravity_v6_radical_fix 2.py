#!/usr/bin/env python3
"""
RS Gravity v6 - Radical Fix for Dwarf Spheroidals
Key insight: The problem isn't just the screening function,
but the fundamental scale at which RS gravity operates.
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Optional

# Physical constants
G_NEWTON = 6.67430e-11  # m^3 kg^-1 s^-2
C_LIGHT = 299792458.0   # m/s
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

@dataclass
class RadicalRSGravityParams:
    """Parameters with radical changes for dwarf spheroidals."""
    # Fundamental parameters
    beta_0: float = -(PHI - 1) / PHI**5  # -0.055728
    
    # NEW: Scale-dependent beta that weakens at small scales
    beta_dwarf_suppression: float = 0.1  # Reduces beta by 90% for dwarfs
    
    # Recognition lengths - NEW: Add intermediate scale for dwarfs
    l0: float = 0.1 * 3.086e19   # m (0.1 kpc - dwarf scale)
    l1: float = 0.97 * 3.086e19  # m (0.97 kpc)
    l2: float = 24.3 * 3.086e19  # m (24.3 kpc)
    
    # NEW: Dual screening mechanism
    rho_gap_low: float = 1e-26    # kg/m^3 (ultra-low density cutoff)
    rho_gap_high: float = 1e-23   # kg/m^3 (high density transition)
    
    # NEW: Environmental suppression factor
    isolation_factor: float = 0.3  # Isolated systems have 70% weaker effects

class RadicalRSGravitySolver:
    """Solver with radical fixes for dwarf spheroidals."""
    
    def __init__(self, params: Optional[RadicalRSGravityParams] = None):
        self.params = params or RadicalRSGravityParams()
    
    def dual_screening(self, rho: float, r: float) -> float:
        """
        NEW: Dual screening mechanism.
        - Ultra-low densities: complete screening
        - Intermediate: partial screening based on scale
        - High densities: no screening
        """
        if rho < self.params.rho_gap_low:
            return 0.01  # 99% screening
        elif rho < self.params.rho_gap_high:
            # Scale-dependent screening
            scale_factor = np.exp(-r / self.params.l0)
            density_factor = (rho / self.params.rho_gap_high)**0.3
            return scale_factor * density_factor
        else:
            return 1.0  # No screening
    
    def scale_dependent_beta(self, r: float, rho: float) -> float:
        """
        NEW: Beta depends on both scale and density.
        Dramatically reduced for dwarf-scale systems.
        """
        # Base beta
        beta = self.params.beta_0
        
        # Dwarf scale suppression
        if r < self.params.l1:
            suppression = self.params.beta_dwarf_suppression
            beta *= suppression
        
        # Additional density suppression
        if rho < 1e-24:
            beta *= (rho / 1e-24)**0.2
        
        return beta
    
    def environmental_factor(self, is_isolated: bool, has_substructure: bool) -> float:
        """
        NEW: Environmental effects on gravity enhancement.
        """
        factor = 1.0
        
        if is_isolated:
            factor *= self.params.isolation_factor
        
        if not has_substructure:
            factor *= 0.7  # Smooth systems have weaker effects
        
        return factor
    
    def recognition_kernel_modified(self, r: float) -> float:
        """
        Modified kernel with three scales including dwarf scale.
        """
        def xi_func(x):
            if x < 0.1:
                return 0.6 - 0.0357 * x**2
            elif x > 50:
                return 3 * np.cos(x) / x**3
            else:
                return 3 * (np.sin(x) - x * np.cos(x)) / x**3
        
        # Three-scale kernel
        kernel = (0.5 * xi_func(r / self.params.l0) +  # Dwarf scale
                 xi_func(r / self.params.l1) + 
                 xi_func(r / self.params.l2))
        
        return kernel
    
    def effective_gravity_dwarf(self, r: float, rho: float, 
                               is_isolated: bool = True,
                               has_substructure: bool = False) -> float:
        """
        Effective gravity with all dwarf-specific modifications.
        """
        # Scale-dependent beta
        beta = self.scale_dependent_beta(r, rho)
        
        # Power law (much weaker for dwarfs)
        power_law = (50.8e-6 / r)**beta
        
        # Modified kernel
        kernel = self.recognition_kernel_modified(r)
        
        # Dual screening
        screening = self.dual_screening(rho, r)
        
        # Environmental factor
        env_factor = self.environmental_factor(is_isolated, has_substructure)
        
        # Total effective G
        G_eff = G_NEWTON * power_law * kernel * screening * env_factor
        
        return G_eff
    
    def predict_dwarf_dispersion_new(self, M_total: float, r_half: float, 
                                    rho_central: float, 
                                    is_isolated: bool = True) -> float:
        """
        Predict dwarf spheroidal velocity dispersion with all fixes.
        """
        # Use modified gravity
        G_eff = self.effective_gravity_dwarf(r_half, rho_central, is_isolated)
        
        # King model correction for dwarf spheroidals
        king_factor = 0.4  # Accounts for non-isothermal distribution
        
        # Anisotropy correction
        anisotropy_factor = 1.3  # Radial orbits increase dispersion
        
        # Calculate dispersion
        sigma_squared = king_factor * anisotropy_factor * G_eff * M_total / r_half
        
        return np.sqrt(sigma_squared)
    
    def effective_gravity_disk(self, r: float, rho: float) -> float:
        """
        Standard RS gravity for disk galaxies (works well).
        """
        beta = self.params.beta_0 * 1.492  # Standard scaling
        power_law = (50.8e-6 / r)**beta
        
        # Standard kernel for disks
        def xi_func(x):
            if x < 0.1:
                return 0.6 - 0.0357 * x**2
            elif x > 50:
                return 3 * np.cos(x) / x**3
            else:
                return 3 * (np.sin(x) - x * np.cos(x)) / x**3
        
        kernel = xi_func(r / self.params.l1) + xi_func(r / self.params.l2)
        
        # Minimal screening for disks
        screening = 1.0 if rho > 1e-23 else 0.9
        
        # Velocity gradient enhancement for disks
        vel_enhancement = 2.0  # Typical for rotating disks
        
        G_eff = G_NEWTON * power_law * kernel * screening * vel_enhancement * 1.326
        
        return G_eff

def test_radical_fixes():
    """Test the radical fixes on dwarf spheroidals."""
    solver = RadicalRSGravitySolver()
    
    print("Testing Radical Fixes for RS Gravity v6")
    print("=" * 50)
    
    # Test 1: Dwarf spheroidal predictions
    print("\n1. Dwarf Spheroidal Predictions (with radical fixes):")
    print("-" * 50)
    
    dwarfs = [
        ('Draco', 3e7 * 2e30, 200 * 3.086e16, 2.7e-25, 9.1),
        ('Fornax', 4e8 * 2e30, 700 * 3.086e16, 1.5e-25, 11.7),
        ('Sculptor', 2e7 * 2e30, 280 * 3.086e16, 3.5e-25, 9.2),
        ('Leo I', 5e7 * 2e30, 250 * 3.086e16, 2.0e-25, 9.2),
        ('Leo II', 1e7 * 2e30, 180 * 3.086e16, 1.5e-25, 6.6),
        ('Carina', 2e7 * 2e30, 250 * 3.086e16, 1.8e-25, 6.6)
    ]
    
    print(f"{'Galaxy':10s} {'M (M☉)':>10s} {'r½ (pc)':>8s} {'ρ (kg/m³)':>10s} "
          f"{'σ_obs':>6s} {'σ_pred':>6s} {'Error':>7s}")
    print("-" * 65)
    
    for name, M, r_half, rho, obs_disp in dwarfs:
        pred_disp = solver.predict_dwarf_dispersion_new(M, r_half, rho) / 1000  # km/s
        error = (pred_disp - obs_disp) / obs_disp * 100
        
        M_solar = M / (2e30)
        r_pc = r_half / 3.086e16
        
        print(f"{name:10s} {M_solar:10.1e} {r_pc:8.0f} {rho:10.1e} "
              f"{obs_disp:6.1f} {pred_disp:6.1f} {error:+7.1f}%")
    
    # Test 2: Compare gravity at different scales
    print("\n2. Effective Gravity Comparison:")
    print("-" * 50)
    print(f"{'System Type':20s} {'r (kpc)':>10s} {'ρ (kg/m³)':>10s} {'G_eff/G_0':>10s}")
    print("-" * 50)
    
    test_cases = [
        ('Dwarf core', 0.1, 1e-25, True),
        ('Dwarf outskirts', 0.5, 1e-26, True),
        ('Disk inner', 2.0, 1e-22, False),
        ('Disk outer', 10.0, 1e-23, False),
        ('Disk far outer', 50.0, 1e-24, False)
    ]
    
    for name, r_kpc, rho, is_dwarf in test_cases:
        r = r_kpc * 3.086e19
        if is_dwarf:
            G_eff = solver.effective_gravity_dwarf(r, rho)
        else:
            G_eff = solver.effective_gravity_disk(r, rho)
        
        enhancement = G_eff / G_NEWTON
        print(f"{name:20s} {r_kpc:10.1f} {rho:10.1e} {enhancement:10.2f}")
    
    # Test 3: Screening comparison
    print("\n3. Dual Screening Mechanism:")
    print("-" * 40)
    print(f"{'ρ (kg/m³)':>12s} {'r = 0.1 kpc':>12s} {'r = 1 kpc':>12s}")
    print("-" * 40)
    
    densities = [1e-27, 1e-26, 1e-25, 1e-24, 1e-23, 1e-22]
    for rho in densities:
        s1 = solver.dual_screening(rho, 0.1 * 3.086e19)
        s2 = solver.dual_screening(rho, 1.0 * 3.086e19)
        print(f"{rho:12.1e} {s1:12.3f} {s2:12.3f}")
    
    print("\n" + "=" * 50)
    print("Key Radical Changes:")
    print("1. Scale-dependent beta: 90% reduction for r < 1 kpc")
    print("2. Dual screening: Ultra-low and scale-dependent")
    print("3. Environmental factors: Isolated systems 70% weaker")
    print("4. Three-scale kernel including dwarf scale (0.1 kpc)")
    print("5. King model and anisotropy corrections")

if __name__ == "__main__":
    test_radical_fixes() 