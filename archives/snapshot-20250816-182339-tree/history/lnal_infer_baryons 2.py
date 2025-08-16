#!/usr/bin/env python3
"""
LNAL Baryon Inference
=====================
Infer surface density Σ(r) from incomplete observational data
using astrophysical priors and scaling relations.
"""

import numpy as np
from scipy.stats import truncnorm, norm, lognorm
from scipy.integrate import simpson
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass

# Constants
kpc = 3.0856775814913673e19  # m
pc = kpc / 1000
M_sun = 1.98847e30  # kg
L_sun = 3.828e26  # W


@dataclass
class GalaxyObservables:
    """Observed galaxy properties from catalogs"""
    name: str
    luminosity_3p6: float  # L_3.6μm [L_sun]
    axis_ratio: float  # b/a
    redshift: float  # z
    color_gi: Optional[float] = None  # g-i color
    HI_flux: Optional[float] = None  # Integrated HI flux [Jy km/s]
    distance_prior: Optional[Tuple[float, float]] = None  # (mean, std) [Mpc]


@dataclass
class BaryonParameters:
    """Inferred baryon distribution parameters"""
    distance: float  # Mpc
    inclination: float  # radians
    M_star: float  # kg
    R_star: float  # m (stellar scale length)
    M_HI: float  # kg
    R_HI: float  # m (HI scale length)
    M_H2: float  # kg
    
    def to_array(self) -> np.ndarray:
        """Convert to array for MCMC"""
        return np.array([
            self.distance,
            self.inclination,
            np.log10(self.M_star / M_sun),
            self.R_star / kpc,
            np.log10(self.M_HI / M_sun),
            self.R_HI / kpc,
            np.log10(self.M_H2 / M_sun) if self.M_H2 > 0 else 0
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'BaryonParameters':
        """Create from array"""
        return cls(
            distance=arr[0],
            inclination=arr[1],
            M_star=10**arr[2] * M_sun,
            R_star=arr[3] * kpc,
            M_HI=10**arr[4] * M_sun,
            R_HI=arr[5] * kpc,
            M_H2=10**arr[6] * M_sun if len(arr) > 6 and arr[6] > 0 else 0
        )


class BaryonPriors:
    """Astrophysical priors and scaling relations"""
    
    @staticmethod
    def distance_from_redshift(z: float, H0: float = 73.0) -> Tuple[float, float]:
        """
        Distance from redshift (simple Hubble flow).
        Returns (mean, std) in Mpc.
        """
        c_km = 299792.458  # km/s
        D = c_km * z / H0  # Mpc
        sigma_D = D * 0.05  # 5% uncertainty
        return D, sigma_D
    
    @staticmethod
    def inclination_from_axis_ratio(q: float) -> Tuple[float, float]:
        """
        Inclination from axis ratio q = b/a.
        Returns (mean, std) in radians.
        """
        # Assume intrinsic thickness q0 = 0.2
        q0 = 0.2
        if q < q0:
            q = q0
        cos_i = np.sqrt((q**2 - q0**2) / (1 - q0**2))
        i = np.arccos(cos_i)
        sigma_i = 5 * np.pi / 180  # 5 degree uncertainty
        return i, sigma_i
    
    @staticmethod
    def stellar_mass_to_light_3p6(color_gi: Optional[float] = None) -> Tuple[float, float]:
        """
        Stellar M/L ratio at 3.6μm.
        Based on Bell et al. (2003) and Meidt et al. (2014).
        Returns (mean, std) in M_sun/L_sun.
        """
        if color_gi is not None:
            # Color-dependent M/L (simplified)
            log_ML = -0.15 + 0.2 * color_gi
            ML = 10**log_ML
            sigma_ML = ML * 0.1 * np.log(10)  # 0.1 dex uncertainty
        else:
            # Default for disk galaxies
            ML = 0.6
            sigma_ML = 0.1
        return ML, sigma_ML
    
    @staticmethod
    def stellar_scale_length(L_star: float) -> Tuple[float, float]:
        """
        Stellar disk scale length from luminosity.
        Based on size-luminosity relation.
        L_star in L_sun, returns (mean, std) in kpc.
        """
        # R_d ∝ L^0.3 (approximate)
        R_d = 3.0 * (L_star / 1e10)**0.3  # kpc
        sigma_R_d = R_d * 0.2  # 20% scatter
        return R_d, sigma_R_d
    
    @staticmethod
    def HI_mass_from_luminosity(L_star: float) -> Tuple[float, float]:
        """
        HI mass from stellar luminosity.
        Based on M_HI-M_* relation for late-type galaxies.
        L_star in L_sun, returns (mean, std) in M_sun.
        """
        M_star = 0.6 * L_star  # Assume M/L = 0.6
        log_MHI = 9.0 + 0.4 * (np.log10(M_star) - 10)  # Simplified relation
        M_HI = 10**log_MHI
        sigma_log_MHI = 0.3  # 0.3 dex scatter
        return M_HI, M_HI * sigma_log_MHI * np.log(10)
    
    @staticmethod
    def HI_scale_length(M_HI: float) -> Tuple[float, float]:
        """
        HI disk scale length from HI mass.
        Based on Wang et al. (2016).
        M_HI in M_sun, returns (mean, std) in kpc.
        """
        # R_HI ∝ M_HI^0.5
        R_HI = 10.0 * (M_HI / 1e9)**0.5  # kpc
        sigma_R_HI = R_HI * 0.25  # 25% scatter
        return R_HI, sigma_R_HI
    
    @staticmethod
    def H2_mass_from_stellar(M_star: float, galaxy_type: str = 'late') -> Tuple[float, float]:
        """
        Molecular gas mass from stellar mass.
        M_star in M_sun, returns (mean, std) in M_sun.
        """
        if galaxy_type == 'late':
            # Late-type galaxies
            f_H2 = 0.1  # M_H2/M_star ~ 0.1
            M_H2 = f_H2 * M_star
            sigma_M_H2 = M_H2 * 0.5  # Factor of 2 uncertainty
        else:
            # Early-type or gas-poor
            M_H2 = 0
            sigma_M_H2 = 0
        return M_H2, sigma_M_H2


def sample_baryon_parameters(obs: GalaxyObservables, 
                           n_samples: int = 1) -> np.ndarray:
    """
    Sample baryon parameters from priors given observables.
    
    Returns array of shape (n_samples, n_params).
    """
    priors = BaryonPriors()
    samples = []
    
    for _ in range(n_samples):
        # Distance
        if obs.distance_prior is not None:
            d_mean, d_std = obs.distance_prior
        else:
            d_mean, d_std = priors.distance_from_redshift(obs.redshift)
        distance = np.random.normal(d_mean, d_std)
        distance = max(distance, 0.1)  # Minimum 0.1 Mpc
        
        # Inclination
        i_mean, i_std = priors.inclination_from_axis_ratio(obs.axis_ratio)
        inclination = np.random.normal(i_mean, i_std)
        inclination = np.clip(inclination, 0, np.pi/2)
        
        # Stellar mass
        ML_mean, ML_std = priors.stellar_mass_to_light_3p6(obs.color_gi)
        ML = np.random.normal(ML_mean, ML_std)
        ML = max(ML, 0.1)  # Minimum M/L
        M_star = ML * obs.luminosity_3p6 * M_sun
        
        # Stellar scale length
        R_d_mean, R_d_std = priors.stellar_scale_length(obs.luminosity_3p6)
        R_star = np.random.normal(R_d_mean, R_d_std) * kpc
        R_star = max(R_star, 0.1 * kpc)
        
        # HI mass
        if obs.HI_flux is not None:
            # From HI flux: M_HI = 2.36e5 * D^2 * S_HI
            M_HI = 2.36e5 * distance**2 * obs.HI_flux * M_sun
            M_HI *= np.random.lognormal(0, 0.1)  # 10% calibration uncertainty
        else:
            MHI_mean, MHI_std = priors.HI_mass_from_luminosity(obs.luminosity_3p6)
            M_HI = np.random.normal(MHI_mean, MHI_std) * M_sun
            M_HI = max(M_HI, 1e7 * M_sun)
        
        # HI scale length
        RHI_mean, RHI_std = priors.HI_scale_length(M_HI / M_sun)
        R_HI = np.random.normal(RHI_mean, RHI_std) * kpc
        R_HI = max(R_HI, R_star)  # HI extends beyond stars
        
        # H2 mass
        MH2_mean, MH2_std = priors.H2_mass_from_stellar(M_star / M_sun)
        if MH2_mean > 0:
            M_H2 = np.random.normal(MH2_mean, MH2_std) * M_sun
            M_H2 = max(M_H2, 0)
        else:
            M_H2 = 0
        
        # Create parameter object
        params = BaryonParameters(
            distance=distance,
            inclination=inclination,
            M_star=M_star,
            R_star=R_star,
            M_HI=M_HI,
            R_HI=R_HI,
            M_H2=M_H2
        )
        
        samples.append(params.to_array())
    
    return np.array(samples)


def build_surface_density(params: BaryonParameters) -> Callable:
    """
    Build surface density function Σ(r) from baryon parameters.
    
    Returns callable Σ(r) in kg/m².
    """
    def surface_density(r):
        """Total baryonic surface density"""
        # Stellar exponential disk
        Sigma_star_0 = params.M_star / (2 * np.pi * params.R_star**2)
        Sigma_star = Sigma_star_0 * np.exp(-r / params.R_star)
        
        # HI exponential disk with flaring
        Sigma_HI_0 = params.M_HI / (2 * np.pi * params.R_HI**2)
        flare_factor = 1 + (r / (5 * params.R_star))**2  # Flaring at large radii
        Sigma_HI = Sigma_HI_0 * np.exp(-r / params.R_HI) * flare_factor
        
        # Include helium
        Sigma_gas = 1.33 * Sigma_HI
        
        # H2 (concentrated in inner regions)
        if params.M_H2 > 0:
            R_H2 = 0.5 * params.R_star  # H2 more concentrated
            Sigma_H2_0 = params.M_H2 / (2 * np.pi * R_H2**2)
            Sigma_H2 = Sigma_H2_0 * np.exp(-r / R_H2)
            Sigma_gas += Sigma_H2
        
        return Sigma_star + Sigma_gas
    
    return surface_density


def infer_rotation_curve(obs: GalaxyObservables, 
                        r_eval: np.ndarray,
                        n_samples: int = 100) -> Dict:
    """
    Infer rotation curve with uncertainty from incomplete data.
    
    Parameters:
    -----------
    obs : GalaxyObservables
        Observed galaxy properties
    r_eval : array_like
        Radii to evaluate [m]
    n_samples : int
        Number of Monte Carlo samples
    
    Returns:
    --------
    dict with keys:
        'r': radius array [m]
        'v_mean': mean velocity [m/s]
        'v_std': velocity std dev [m/s]
        'v_percentiles': dict of percentiles
        'params_mean': mean baryon parameters
    """
    from lnal_pure_formula import lnal_circular_velocity
    
    # Sample baryon parameters
    param_samples = sample_baryon_parameters(obs, n_samples)
    
    # Compute velocity for each sample
    v_samples = []
    for i in range(n_samples):
        params = BaryonParameters.from_array(param_samples[i])
        Sigma = build_surface_density(params)
        v = lnal_circular_velocity(r_eval, Sigma)
        v_samples.append(v)
    
    v_samples = np.array(v_samples)
    
    # Statistics
    v_mean = np.mean(v_samples, axis=0)
    v_std = np.std(v_samples, axis=0)
    v_percentiles = {
        'p16': np.percentile(v_samples, 16, axis=0),
        'p50': np.percentile(v_samples, 50, axis=0),
        'p84': np.percentile(v_samples, 84, axis=0),
        'p05': np.percentile(v_samples, 5, axis=0),
        'p95': np.percentile(v_samples, 95, axis=0)
    }
    
    # Mean parameters
    params_mean = BaryonParameters.from_array(np.mean(param_samples, axis=0))
    
    return {
        'r': r_eval,
        'v_mean': v_mean,
        'v_std': v_std,
        'v_percentiles': v_percentiles,
        'params_mean': params_mean,
        'param_samples': param_samples
    }


# Test the module
if __name__ == "__main__":
    # Example galaxy: NGC 3198-like
    obs = GalaxyObservables(
        name="NGC3198-like",
        luminosity_3p6=2e10,  # L_sun
        axis_ratio=0.3,  # Edge-on
        redshift=0.00215,
        color_gi=0.8,  # Spiral galaxy color
        HI_flux=300  # Jy km/s
    )
    
    # Evaluate rotation curve
    r = np.logspace(np.log10(0.5 * kpc), np.log10(30 * kpc), 50)
    result = infer_rotation_curve(obs, r, n_samples=100)
    
    print("Baryon Inference Test")
    print("=" * 50)
    print(f"Galaxy: {obs.name}")
    print(f"Inferred distance: {result['params_mean'].distance:.1f} Mpc")
    print(f"Inferred M_star: {result['params_mean'].M_star/M_sun:.2e} M_sun")
    print(f"Inferred M_HI: {result['params_mean'].M_HI/M_sun:.2e} M_sun")
    print(f"V_flat (mean ± std): {result['v_mean'][-1]/1000:.1f} ± {result['v_std'][-1]/1000:.1f} km/s") 