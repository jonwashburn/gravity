#!/usr/bin/env python3
"""
LNAL-MOND Solver
================
LNAL theory implemented as enhanced MOND at galactic scales.
"""

import numpy as np
from scipy.integrate import simpson
from dataclasses import dataclass
from typing import Dict, Optional
import matplotlib.pyplot as plt

# Physical constants
G = 6.67430e-11  # m³/kg/s²
c = 2.99792458e8  # m/s
M_sun = 1.98847e30  # kg
kpc = 3.0856775814913673e19  # m
pc = kpc / 1000

# MOND acceleration scale
a0 = 1.2e-10  # m/s²

# LNAL enhancement from Recognition Science
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio


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
class LNALMONDParameters:
    """LNAL-MOND model parameters"""
    a0_factor: float = 1.0  # Modifies MOND acceleration scale
    
    def to_array(self) -> np.ndarray:
        return np.array([self.a0_factor])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'LNALMONDParameters':
        return cls(a0_factor=arr[0])


def mond_mu_simple(x: np.ndarray) -> np.ndarray:
    """Simple MOND interpolation function"""
    return x / (1 + x)


def g_mond(r: np.ndarray, galaxy: GalaxyData, 
           params: LNALMONDParameters) -> np.ndarray:
    """
    MOND acceleration with LNAL enhancement.
    
    In MOND: μ(g/a0) × g = g_N
    Therefore: g = g_N / μ(g/a0)
    
    We solve this iteratively.
    """
    # Effective MOND scale
    a0_eff = a0 * params.a0_factor
    
    # Enclosed mass from surface density
    M_enc = np.zeros_like(r)
    for i in range(1, len(r)):
        r_int = r[:i+1]
        Sigma_int = galaxy.surface_density(r_int)
        M_enc[i] = 2 * np.pi * simpson(r_int * Sigma_int, x=r_int)
    M_enc[0] = M_enc[1] * (r[0]/r[1])**2  # Extrapolate
    
    # Newtonian acceleration
    g_newton = G * M_enc / r**2
    
    # Initial guess: deep MOND limit
    g = np.sqrt(g_newton * a0_eff)
    
    # Iterate to solve μ(g/a0) × g = g_N
    for _ in range(10):
        x = g / a0_eff
        mu = mond_mu_simple(x)
        g_new = g_newton / mu
        g = 0.5 * g + 0.5 * g_new  # Damped update
    
    return g


def v_circ_mond(r: np.ndarray, galaxy: GalaxyData,
                params: LNALMONDParameters) -> np.ndarray:
    """Circular velocity from MOND acceleration"""
    g = g_mond(r, galaxy, params)
    return np.sqrt(r * g)


def chi_squared_mond(params: LNALMONDParameters, galaxy: GalaxyData) -> float:
    """Compute χ² for galaxy fit"""
    v_model = v_circ_mond(galaxy.r, galaxy, params)
    residuals = (v_model - galaxy.v_obs) / galaxy.v_err
    return np.sum(residuals**2)


def fit_galaxy_mond(galaxy: GalaxyData) -> Dict:
    """Fit MOND model to galaxy"""
    # Try different a0 factors
    a0_range = np.linspace(0.5, 2.0, 40)
    
    best_chi2 = np.inf
    best_a0 = 1.0
    
    for a0_fac in a0_range:
        params = LNALMONDParameters(a0_factor=a0_fac)
        chi2 = chi_squared_mond(params, galaxy)
        
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_a0 = a0_fac
    
    best_params = LNALMONDParameters(a0_factor=best_a0)
    v_model = v_circ_mond(galaxy.r, galaxy, best_params)
    
    return {
        'best_params': best_params,
        'chi2': best_chi2,
        'chi2_reduced': best_chi2 / len(galaxy.r),
        'v_model': v_model
    }


def test_mond_solver():
    """Test MOND solver on synthetic galaxy"""
    print("LNAL-MOND Solver Test")
    print("=" * 60)
    
    # Create test galaxy
    r = np.logspace(np.log10(0.5 * kpc), np.log10(30 * kpc), 50)
    M_disk = 5e10 * M_sun
    R_d = 3 * kpc
    
    def surface_density(r_eval):
        Sigma_0 = M_disk / (2 * np.pi * R_d**2)
        return Sigma_0 * np.exp(-r_eval / R_d)
    
    # Expected flat velocity from MOND
    # V⁴ = G × M × a0 for exponential disk
    V_mond_expected = (G * M_disk * a0)**(1/4)
    print(f"Expected MOND velocity: {V_mond_expected/1000:.1f} km/s")
    
    galaxy = GalaxyData(
        name="TestGalaxy",
        r=r,
        v_obs=V_mond_expected * np.ones_like(r),
        v_err=5e3 * np.ones_like(r),
        surface_density=surface_density,
        scale_length=R_d,
        inclination=0.0
    )
    
    # Test solver
    params = LNALMONDParameters(a0_factor=1.0)
    v_model = v_circ_mond(r, galaxy, params)
    v_asymptotic = np.mean(v_model[-10:])
    
    print(f"Model asymptotic velocity: {v_asymptotic/1000:.1f} km/s")
    print(f"Ratio: {v_asymptotic/V_mond_expected:.3f}")
    
    # Fit
    result = fit_galaxy_mond(galaxy)
    print(f"\nBest fit a0_factor: {result['best_params'].a0_factor:.3f}")
    print(f"χ²/N: {result['chi2_reduced']:.3f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(r/kpc, galaxy.v_obs/1000, 'k--', label='Target')
    plt.plot(r/kpc, v_model/1000, 'b-', label='MOND (a0=1)')
    plt.plot(r/kpc, result['v_model']/1000, 'r-', label=f"Best fit (a0={result['best_params'].a0_factor:.2f})")
    plt.xlabel('Radius [kpc]')
    plt.ylabel('Velocity [km/s]')
    plt.title('LNAL-MOND Solver Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test_mond_solver.png')
    plt.close()
    
    print("\nSaved test plot to test_mond_solver.png")


if __name__ == "__main__":
    test_mond_solver() 