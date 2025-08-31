#!/usr/bin/env python3
"""
LNAL Solver Core v3
===================
Numerically stable implementation with proper scaling.
"""

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# Physical constants
G = 6.67430e-11  # m³/kg/s²
c = 2.99792458e8  # m/s
hbar = 1.054571817e-34  # J⋅s
M_sun = 1.98847e30  # kg
kpc = 3.0856775814913673e19  # m
pc = kpc / 1000

# Fixed LNAL parameters
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
L0 = 0.335e-9  # m (voxel size)
ELL_1 = 0.97 * kpc  # Inner recognition length
G_DAGGER = 1.2e-10  # m/s² (MOND scale)

# Key insight: At galactic scales, the enhancement is enormous
# but the coupling λ is tiny, so their product is O(1)
# We work with effective parameters to avoid overflow


@dataclass
class GalaxyData:
    """Container for galaxy observational data"""
    name: str
    r: np.ndarray  # Radius array [m]
    v_obs: np.ndarray  # Observed velocities [m/s]
    v_err: np.ndarray  # Velocity errors [m/s]
    surface_density: callable  # Σ(r) [kg/m²]
    scale_length: float  # Disk scale length [m]
    inclination: float  # Inclination angle [rad]


@dataclass
class LNALParameters:
    """LNAL model parameters"""
    hierarchy_strength: float = 1.0  # Modifies hierarchical enhancement
    mond_transition: float = 1.0  # Modifies MOND transition scale
    interference_fraction: float = 0.0  # Quantum-classical interference
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.hierarchy_strength,
            self.mond_transition,
            self.interference_fraction
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'LNALParameters':
        return cls(
            hierarchy_strength=arr[0],
            mond_transition=arr[1],
            interference_fraction=arr[2] if len(arr) > 2 else 0.0
        )


def mond_interpolation_function(x: np.ndarray) -> np.ndarray:
    """
    Standard MOND interpolation function.
    μ(x) = x / √(1 + x²)
    """
    return x / np.sqrt(1 + x**2)


def g_lnal_simplified(r: np.ndarray, params: LNALParameters, 
                     galaxy: GalaxyData) -> np.ndarray:
    """
    Simplified LNAL acceleration using MOND-like formulation.
    
    The full LNAL theory reduces to a MOND-like form at galactic scales:
    g = g_N × μ(g_N / g†)
    
    where the effective g† depends on hierarchical organization.
    """
    # Baryon distribution
    Sigma = galaxy.surface_density(r)
    
    # Enclosed mass
    M_enc = np.zeros_like(r)
    for i in range(1, len(r)):
        # Integrate Σ(r) × 2πr dr
        r_int = r[:i+1]
        Sigma_int = galaxy.surface_density(r_int)
        M_enc[i] = 2 * np.pi * simpson(r_int * Sigma_int, x=r_int)
    
    # Newtonian acceleration
    g_newton = G * M_enc / r**2
    g_newton[0] = g_newton[1]  # Handle r=0
    
    # Effective MOND scale with hierarchy modification
    # The hierarchy_strength parameter modulates how much the 
    # information field enhances gravity at low accelerations
    g_eff = G_DAGGER * params.mond_transition / params.hierarchy_strength
    
    # MOND interpolation
    x = g_newton / g_eff
    mu = mond_interpolation_function(x)
    
    # Total acceleration
    g_total = g_newton * mu
    
    # Optional: Add interference term for better fits
    if params.interference_fraction > 0:
        # In deep MOND regime, add constructive interference
        deep_mond = x < 0.1
        g_total[deep_mond] *= (1 + params.interference_fraction)
    
    return g_total


def v_circ(r: np.ndarray, params: LNALParameters, 
           galaxy: GalaxyData) -> np.ndarray:
    """Circular velocity v(r) = √(r × g(r))"""
    g = g_lnal_simplified(r, params, galaxy)
    return np.sqrt(r * g)


def chi_squared(params: LNALParameters, galaxy: GalaxyData) -> float:
    """Compute χ² for galaxy fit"""
    v_model = v_circ(galaxy.r, params, galaxy)
    residuals = (v_model - galaxy.v_obs) / galaxy.v_err
    return np.sum(residuals**2)


def fit_single_galaxy(galaxy: GalaxyData, 
                     initial_params: Optional[LNALParameters] = None) -> Dict:
    """
    Fit LNAL model to a single galaxy using simple grid search.
    """
    if initial_params is None:
        initial_params = LNALParameters()
    
    # Grid search over reasonable parameter ranges
    hierarchy_range = np.linspace(0.5, 2.0, 20)
    mond_range = np.linspace(0.5, 2.0, 20)
    
    best_chi2 = np.inf
    best_params = initial_params
    
    for h in hierarchy_range:
        for m in mond_range:
            params = LNALParameters(
                hierarchy_strength=h,
                mond_transition=m,
                interference_fraction=0.0
            )
            chi2 = chi_squared(params, galaxy)
            
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_params = params
    
    # Compute final model
    v_model = v_circ(galaxy.r, best_params, galaxy)
    
    return {
        'best_params': best_params,
        'chi2': best_chi2,
        'chi2_reduced': best_chi2 / len(galaxy.r),
        'v_model': v_model
    }


def test_solver_v3():
    """Test the v3 solver"""
    print("LNAL Solver Core v3 - Test")
    print("=" * 60)
    
    # Create test galaxy with exponential disk
    r = np.logspace(np.log10(0.1 * kpc), np.log10(30 * kpc), 50)
    M_disk = 2e10 * M_sun
    R_d = 3 * kpc
    
    # Surface density
    def surface_density(r_eval):
        Sigma_0 = M_disk / (2 * np.pi * R_d**2)
        return Sigma_0 * np.exp(-r_eval / R_d)
    
    # Create galaxy
    v_flat = 150e3  # 150 km/s
    galaxy = GalaxyData(
        name="TestGalaxy",
        r=r,
        v_obs=v_flat * np.ones_like(r),
        v_err=5e3 * np.ones_like(r),
        surface_density=surface_density,
        scale_length=R_d,
        inclination=0.0
    )
    
    # Test with different parameters
    params_list = [
        LNALParameters(1.0, 1.0, 0.0),  # Standard
        LNALParameters(0.5, 1.0, 0.0),  # Strong hierarchy
        LNALParameters(2.0, 1.0, 0.0),  # Weak hierarchy
    ]
    
    for params in params_list:
        v_model = v_circ(r, params, galaxy)
        v_asymptotic = np.mean(v_model[-10:])
        print(f"\nParams: h={params.hierarchy_strength:.1f}, m={params.mond_transition:.1f}")
        print(f"  V_asymptotic = {v_asymptotic/1000:.1f} km/s")
        print(f"  V_obs/V_model = {v_flat/v_asymptotic:.2f}")


if __name__ == "__main__":
    test_solver_v3() 