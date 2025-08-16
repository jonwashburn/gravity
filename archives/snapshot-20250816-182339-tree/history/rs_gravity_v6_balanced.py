#!/usr/bin/env python3
"""
RS Gravity v6 - Balanced Approach
Key insight: The issue is that we're applying disk galaxy parameters
to fundamentally different dwarf spheroidal systems.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import json

# Physical constants
G_NEWTON = 6.67430e-11  # m^3 kg^-1 s^-2
C_LIGHT = 299792458.0   # m/s
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
M_SUN = 1.989e30  # kg
PC_TO_M = 3.086e16  # m

@dataclass
class BalancedRSGravityParams:
    """Balanced parameters that work for both dwarfs and disks."""
    # Core parameters from golden ratio
    beta_0: float = -(PHI - 1) / PHI**5  # -0.055728
    
    # Recognition lengths
    l1: float = 0.97 * 3.086e19  # m (0.97 kpc)
    l2: float = 24.3 * 3.086e19  # m (24.3 kpc)
    
    # Critical insight: Different systems need different parameters
    # Disk galaxy parameters (from SPARC optimization)
    disk_params: dict = None
    
    # Dwarf spheroidal parameters (NEW calibration)
    dwarf_params: dict = None
    
    def __post_init__(self):
        if self.disk_params is None:
            self.disk_params = {
                'lambda_eff': 50.8e-6,     # m
                'beta_scale': 1.492,       # Enhancement
                'coupling_scale': 1.326,   # Overall coupling
                'vel_grad_enhance': 2.0,   # Velocity gradient effect
                'screening_threshold': 1e-23  # kg/m³
            }
        
        if self.dwarf_params is None:
            self.dwarf_params = {
                'lambda_eff': 200e-6,      # Larger scale (4x disk)
                'beta_scale': 0.3,         # Much weaker (20% of disk)
                'coupling_scale': 0.15,    # Weaker coupling
                'vel_grad_enhance': 1.0,   # No velocity enhancement
                'screening_threshold': 1e-25,  # Lower threshold
                'anisotropy_boost': 1.5    # Orbital anisotropy
            }

class BalancedRSGravitySolver:
    """Balanced solver that correctly handles different galaxy types."""
    
    def __init__(self, params: Optional[BalancedRSGravityParams] = None):
        self.params = params or BalancedRSGravityParams()
    
    def classify_system(self, M_total: float, r_half: float, 
                       v_rot: Optional[float] = None) -> str:
        """
        Classify system as disk or dwarf based on properties.
        """
        # Mass threshold
        M_threshold = 1e9 * M_SUN  # 10^9 solar masses
        
        # Size threshold
        r_threshold = 1000 * PC_TO_M  # 1 kpc
        
        # Rotation threshold
        if v_rot is not None and v_rot > 50e3:  # 50 km/s
            return 'disk'
        
        if M_total < M_threshold and r_half < r_threshold:
            return 'dwarf'
        else:
            return 'disk'
    
    def recognition_kernel(self, r: float, system_type: str) -> float:
        """Recognition kernel with system-dependent behavior."""
        def xi_func(x):
            if x < 0.1:
                return 0.6 - 0.0357 * x**2
            elif x > 50:
                return 3 * np.cos(x) / x**3
            else:
                return 3 * (np.sin(x) - x * np.cos(x)) / x**3
        
        if system_type == 'dwarf':
            # Single-scale kernel for dwarfs (simpler structure)
            return xi_func(r / self.params.l1)
        else:
            # Two-scale kernel for disks
            return xi_func(r / self.params.l1) + xi_func(r / self.params.l2)
    
    def screening_function(self, rho: float, system_type: str) -> float:
        """System-dependent screening."""
        threshold = self.params.dwarf_params['screening_threshold'] if system_type == 'dwarf' \
                   else self.params.disk_params['screening_threshold']
        
        if system_type == 'dwarf':
            # Gentler screening for dwarfs
            if rho < threshold:
                return (rho / threshold)**0.3
            else:
                return 1.0
        else:
            # Sharp screening for disks
            if rho < threshold:
                return 0.1
            else:
                return 1.0
    
    def effective_gravity(self, r: float, M_enc: float, rho: float,
                         system_type: str, grad_v: float = 0) -> float:
        """
        Calculate effective gravity with system-dependent parameters.
        """
        # Select appropriate parameters
        if system_type == 'dwarf':
            params = self.params.dwarf_params
        else:
            params = self.params.disk_params
        
        # Beta with scale factor
        beta = self.params.beta_0 * params['beta_scale']
        
        # Power law running
        power_law = (params['lambda_eff'] / r)**beta
        
        # Recognition kernel
        kernel = self.recognition_kernel(r, system_type)
        
        # Screening
        screening = self.screening_function(rho, system_type)
        
        # Velocity gradient enhancement
        vel_factor = 1.0
        if grad_v > 0 and system_type == 'disk':
            vel_factor = params['vel_grad_enhance']
        
        # Total effective G
        G_eff = G_NEWTON * power_law * kernel * screening * vel_factor
        G_eff *= params['coupling_scale']
        
        return G_eff
    
    def predict_dwarf_dispersion(self, M_total: float, r_half: float,
                                rho_central: float) -> float:
        """
        Specialized prediction for dwarf spheroidals.
        Includes corrections for pressure support and anisotropy.
        """
        # Get effective gravity
        G_eff = self.effective_gravity(r_half, M_total/2, rho_central, 'dwarf')
        
        # Base dispersion from virial theorem
        sigma_squared = G_eff * M_total / (3 * r_half)
        
        # Anisotropy correction
        beta_anisotropy = self.params.dwarf_params['anisotropy_boost']
        sigma_squared *= beta_anisotropy
        
        # Pressure support correction for ultra-low density
        if rho_central < 1e-25:
            pressure_boost = 1.3
            sigma_squared *= pressure_boost
        
        return np.sqrt(sigma_squared)
    
    def predict_disk_rotation(self, r: float, M_enc: float, rho: float,
                            grad_v: float = 1e-4) -> float:
        """
        Predict rotation velocity for disk galaxies.
        """
        G_eff = self.effective_gravity(r, M_enc, rho, 'disk', grad_v)
        v_squared = G_eff * M_enc / r
        return np.sqrt(v_squared)
    
    def validate_predictions(self):
        """Test predictions against known systems."""
        print("Validating Balanced RS Gravity v6")
        print("=" * 50)
        
        # Test dwarfs
        print("\nDwarf Spheroidal Galaxies:")
        print("-" * 50)
        print(f"{'Galaxy':10s} {'M (M☉)':>10s} {'r½ (pc)':>8s} {'ρ (kg/m³)':>10s} "
              f"{'σ_obs':>6s} {'σ_pred':>6s} {'Error':>7s}")
        print("-" * 65)
        
        dwarfs = [
            ('Draco', 3e7, 200, 2.7e-25, 9.1),
            ('Fornax', 4e8, 700, 1.5e-25, 11.7),
            ('Sculptor', 2e7, 280, 3.5e-25, 9.2),
            ('Leo I', 5e7, 250, 2.0e-25, 9.2),
            ('Leo II', 1e7, 180, 1.5e-25, 6.6),
            ('Carina', 2e7, 250, 1.8e-25, 6.6)
        ]
        
        chi2_total = 0
        for name, M_solar, r_pc, rho, obs_disp in dwarfs:
            M = M_solar * M_SUN
            r_half = r_pc * PC_TO_M
            
            pred_disp = self.predict_dwarf_dispersion(M, r_half, rho) / 1000  # km/s
            error = (pred_disp - obs_disp) / obs_disp * 100
            chi2 = ((pred_disp - obs_disp) / 1.0)**2  # Assume 1 km/s error
            chi2_total += chi2
            
            print(f"{name:10s} {M_solar:10.1e} {r_pc:8.0f} {rho:10.1e} "
                  f"{obs_disp:6.1f} {pred_disp:6.1f} {error:+7.1f}%")
        
        print(f"\nTotal χ² for dwarfs: {chi2_total:.1f} (χ²/N = {chi2_total/6:.1f})")
        
        # Test disk galaxy scales
        print("\nDisk Galaxy Rotation (typical values):")
        print("-" * 50)
        print(f"{'r (kpc)':>8s} {'M_enc (M☉)':>12s} {'v_pred (km/s)':>14s}")
        print("-" * 35)
        
        disk_tests = [
            (2, 1e10),
            (5, 3e10),
            (10, 6e10),
            (20, 8e10)
        ]
        
        for r_kpc, M_enc_solar in disk_tests:
            r = r_kpc * 3.086e19
            M_enc = M_enc_solar * M_SUN
            rho = 1e-23  # Typical disk density
            
            v_pred = self.predict_disk_rotation(r, M_enc, rho) / 1000  # km/s
            print(f"{r_kpc:8.1f} {M_enc_solar:12.1e} {v_pred:14.1f}")
        
        # Compare enhancements
        print("\nGravity Enhancement Factors:")
        print("-" * 50)
        print(f"{'System':15s} {'r (kpc)':>8s} {'G_eff/G_0':>10s}")
        print("-" * 35)
        
        test_cases = [
            ('Dwarf (0.2 kpc)', 0.2, 1e8 * M_SUN, 1e-25, 'dwarf'),
            ('Dwarf (0.5 kpc)', 0.5, 1e8 * M_SUN, 1e-25, 'dwarf'),
            ('Disk (2 kpc)', 2.0, 1e10 * M_SUN, 1e-23, 'disk'),
            ('Disk (10 kpc)', 10.0, 5e10 * M_SUN, 1e-23, 'disk')
        ]
        
        for name, r_kpc, M, rho, sys_type in test_cases:
            r = r_kpc * 3.086e19
            G_eff = self.effective_gravity(r, M, rho, sys_type)
            enhancement = G_eff / G_NEWTON
            print(f"{name:15s} {r_kpc:8.1f} {enhancement:10.2f}")

def save_balanced_parameters():
    """Save the balanced parameters for future use."""
    solver = BalancedRSGravitySolver()
    
    params_dict = {
        'description': 'Balanced RS Gravity v6 parameters',
        'disk_params': solver.params.disk_params,
        'dwarf_params': solver.params.dwarf_params,
        'core_params': {
            'beta_0': solver.params.beta_0,
            'l1_kpc': 0.97,
            'l2_kpc': 24.3
        }
    }
    
    with open('rs_gravity_v6_balanced_params.json', 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    print("\nParameters saved to rs_gravity_v6_balanced_params.json")

if __name__ == "__main__":
    solver = BalancedRSGravitySolver()
    solver.validate_predictions()
    save_balanced_parameters() 