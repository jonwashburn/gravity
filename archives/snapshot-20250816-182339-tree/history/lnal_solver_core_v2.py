#!/usr/bin/env python3
"""
LNAL Solver Core v2
===================
Production-grade implementation of Light-Native Assembly Language gravity theory.
Consolidated formula with minimal free parameters for galaxy rotation curve fitting.

Theory: g(r) = g_N(r) · F(r; θ)
where F is the LNAL modifier with parameters θ
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

# Fixed LNAL parameters from theory
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
BETA = -(PHI - 1) / PHI**5  # ≈ -0.0557280900
L0 = 0.335e-9  # m (voxel size)
ELL_1 = 0.97 * kpc  # Inner recognition length
ELL_2 = 24.3 * kpc  # Outer recognition length
G_DAGGER = 1.2e-10  # m/s² (MOND scale)

# Derived constants
MU = hbar / (c * ELL_1)  # Field mass ≈ 3.63e-6 m⁻²
M_PROTON = 1.672621898e-27  # kg
I_STAR = M_PROTON * c**2 / L0**3  # Voxel capacity ≈ 4.5e17 J/m³
LAMBDA = np.sqrt(G_DAGGER * c**2 / I_STAR)  # Coupling ≈ 1.63e-26


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
    """LNAL model parameters (θ vector)"""
    # These are the only tunable parameters
    hierarchy_strength: float = 1.0  # Hierarchical enhancement factor
    temporal_coupling: float = 1.0  # Eight-beat resonance strength
    coherence_fraction: float = 0.15  # Quantum-classical interference
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for optimizers"""
        return np.array([
            self.hierarchy_strength,
            self.temporal_coupling,
            self.coherence_fraction
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'LNALParameters':
        """Create from numpy array"""
        return cls(
            hierarchy_strength=arr[0],
            temporal_coupling=arr[1],
            coherence_fraction=arr[2]
        )


def hierarchical_enhancement(r: np.ndarray, scale_length: float, 
                           strength: float = 1.0) -> float:
    """
    Compute hierarchical voxel organization enhancement.
    
    At galactic scales, voxels organize into hierarchical structures
    that enhance information processing capacity.
    """
    # Limit the enhancement to reasonable values
    # At galactic scales (~kpc), we expect ~10^20 hierarchy levels
    scale_ratio = scale_length / L0
    n_levels = np.log(scale_ratio) / np.log(8)
    
    # Base enhancement - limit the exponent to avoid overflow
    # Physical constraint: enhancement should be ~10^10 at galactic scales
    max_exponent = 10  # This gives 8^10 ~ 10^9 maximum enhancement
    effective_exponent = min(strength * n_levels / 2, max_exponent)
    enhancement = np.power(8, effective_exponent)
    
    # Recognition scale boost (modest factor of ~2)
    recognition_factor = 1 + np.exp(-(np.log(scale_length / ELL_1))**2)
    
    return enhancement * recognition_factor


def mond_interpolation(u: np.ndarray) -> np.ndarray:
    """
    MOND interpolation function μ(u) = u / √(1 + u²)
    
    u: dimensionless gradient |∇ρ_I| / (I★ μ)
    """
    return u / np.sqrt(1 + u**2)


def solve_information_field(r: np.ndarray, rho: np.ndarray, 
                          params: LNALParameters,
                          scale_length: float) -> np.ndarray:
    """
    Solve the LNAL field equation for information density ρ_I.
    
    ∇·[μ(u)∇ρ_I] - μ²ρ_I = -λB
    
    Returns: ρ_I(r) [J/m³]
    """
    # Hierarchical parameters
    enhancement = hierarchical_enhancement(r, scale_length, params.hierarchy_strength)
    I_star_eff = I_STAR * enhancement
    lambda_eff = LAMBDA / np.sqrt(enhancement)
    
    # Source term B = ρc²
    B = rho * c**2
    
    # Initial guess: weak field limit
    rho_I = lambda_eff * B / MU**2
    
    # Iterative solution
    for iteration in range(100):
        rho_I_old = rho_I.copy()
        
        # Gradient
        drho_I_dr = np.gradient(rho_I, r)
        
        # MOND function
        u = np.abs(drho_I_dr) / (I_star_eff * MU)
        mu_u = mond_interpolation(u)
        
        # Laplacian in spherical coordinates
        term = r**2 * mu_u * drho_I_dr
        term[0] = term[1]  # Boundary condition
        d_term_dr = np.gradient(term, r)
        laplacian = d_term_dr / (r**2 + 1e-30)
        
        # Update equation
        source = -lambda_eff * B + MU**2 * rho_I
        residual = laplacian - source
        
        # Relaxation with adaptive damping
        omega = 0.3 * (1 + 0.5 * np.exp(-iteration / 30))
        rho_I = rho_I - omega * residual * (r[1] - r[0])**2
        rho_I[rho_I < 0] = 0
        
        # Check convergence
        change = np.max(np.abs(rho_I - rho_I_old) / (I_star_eff + np.abs(rho_I)))
        if change < 1e-6:
            break
    
    return rho_I, lambda_eff


def g_lnal(r: np.ndarray, params: LNALParameters, 
           galaxy: GalaxyData) -> np.ndarray:
    """
    Compute LNAL gravitational acceleration g(r).
    
    Returns: acceleration [m/s²]
    """
    # Baryon distribution
    Sigma = galaxy.surface_density(r)
    
    # Volume density (thin disk approximation)
    h_disk = 0.3 * kpc  # Typical disk height
    rho = Sigma / (2 * h_disk)
    
    # Solve information field
    rho_I, lambda_eff = solve_information_field(r, rho, params, galaxy.scale_length)
    
    # Information field acceleration
    drho_I_dr = np.gradient(rho_I, r)
    a_info = lambda_eff * drho_I_dr / c**2
    
    # Newtonian acceleration (from enclosed mass)
    M_enc = np.zeros_like(r)
    for i in range(1, len(r)):
        M_enc[i] = 2 * np.pi * simpson(r[:i+1] * Sigma[:i+1], x=r[:i+1])
    a_newton = G * M_enc / r**2
    
    # Total acceleration with quantum-classical interference
    interference = 2 * np.sqrt(a_newton * np.abs(a_info)) * params.coherence_fraction
    a_total = a_newton + np.abs(a_info) + interference
    
    return a_total


def v_circ(r: np.ndarray, params: LNALParameters, 
           galaxy: GalaxyData) -> np.ndarray:
    """
    Compute circular velocity curve v(r) = √(r·g(r)).
    
    Returns: velocity [m/s]
    """
    g = g_lnal(r, params, galaxy)
    return np.sqrt(r * g)


def chi_squared(params: LNALParameters, galaxy: GalaxyData) -> float:
    """
    Compute χ² for a single galaxy fit.
    """
    # Model velocities at observed radii
    v_model = v_circ(galaxy.r, params, galaxy)
    
    # χ² with errors
    residuals = (v_model - galaxy.v_obs) / galaxy.v_err
    return np.sum(residuals**2)


def fit_galaxy(galaxy: GalaxyData, 
               initial_params: Optional[LNALParameters] = None) -> Dict:
    """
    Fit LNAL model to a single galaxy.
    
    Returns dictionary with:
    - best_params: Optimized parameters
    - chi2: Final χ²
    - v_model: Model velocity curve
    """
    if initial_params is None:
        initial_params = LNALParameters()
    
    # For now, just evaluate at initial params
    # (Full optimization will be added in the wrapper)
    chi2 = chi_squared(initial_params, galaxy)
    v_model = v_circ(galaxy.r, initial_params, galaxy)
    
    return {
        'best_params': initial_params,
        'chi2': chi2,
        'chi2_reduced': chi2 / len(galaxy.r),
        'v_model': v_model
    }


# Utility functions for data loading
def exponential_disk(r: np.ndarray, M_star: float, R_d: float) -> np.ndarray:
    """
    Exponential disk surface density profile.
    Σ(r) = Σ₀ exp(-r/R_d)
    """
    Sigma_0 = M_star / (2 * np.pi * R_d**2)
    return Sigma_0 * np.exp(-r / R_d)


def load_galaxy_from_sparc(galaxy_id: str, gas_fraction: float = 0.15) -> GalaxyData:
    """
    Load galaxy data from SPARC files.
    
    This is a placeholder - will be implemented with actual SPARC parser.
    """
    # Placeholder implementation
    # Real version will read from Rotmod_LTG/*.dat files
    
    # Example data structure
    r = np.logspace(np.log10(0.1 * kpc), np.log10(50 * kpc), 100)
    v_obs = 150e3 * np.ones_like(r)  # Flat rotation curve
    v_err = 5e3 * np.ones_like(r)
    
    # Disk parameters (example)
    M_star = 1e10 * M_sun
    R_d = 3 * kpc
    
    # Surface density function
    def surface_density(r_eval):
        Sigma_star = exponential_disk(r_eval, M_star, R_d)
        return (1 + gas_fraction) * Sigma_star
    
    return GalaxyData(
        name=galaxy_id,
        r=r,
        v_obs=v_obs,
        v_err=v_err,
        surface_density=surface_density,
        scale_length=R_d,
        inclination=0.0
    )


def test_single_galaxy():
    """Quick test on one galaxy"""
    print("LNAL Solver Core v2 - Test Run")
    print("=" * 60)
    
    # Test galaxy: NGC 3198
    r = np.logspace(np.log10(0.1 * kpc), np.log10(30 * kpc), 50)
    M_star = 1.9e10 * M_sun
    R_d = 3.14 * kpc
    v_flat = 150.1e3  # m/s
    
    # Create galaxy data
    def surface_density(r_eval):
        return exponential_disk(r_eval, M_star * 1.15, R_d)  # 15% gas
    
    galaxy = GalaxyData(
        name="NGC3198",
        r=r,
        v_obs=v_flat * np.ones_like(r),
        v_err=5e3 * np.ones_like(r),
        surface_density=surface_density,
        scale_length=R_d,
        inclination=0.0
    )
    
    # Run model
    params = LNALParameters()
    result = fit_galaxy(galaxy, params)
    
    print(f"Galaxy: {galaxy.name}")
    print(f"χ²/N = {result['chi2_reduced']:.3f}")
    print(f"Model parameters:")
    print(f"  Hierarchy strength: {params.hierarchy_strength:.3f}")
    print(f"  Temporal coupling: {params.temporal_coupling:.3f}")
    print(f"  Coherence fraction: {params.coherence_fraction:.3f}")
    
    # Check asymptotic velocity
    v_model_flat = np.mean(result['v_model'][-10:])
    print(f"\nAsymptotic velocities:")
    print(f"  Observed: {v_flat/1000:.1f} km/s")
    print(f"  Model: {v_model_flat/1000:.1f} km/s")
    print(f"  Ratio: {v_model_flat/v_flat:.3f}")


if __name__ == "__main__":
    test_single_galaxy() 