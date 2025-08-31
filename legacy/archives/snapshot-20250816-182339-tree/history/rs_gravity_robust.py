#!/usr/bin/env python3
"""
Robust Recognition Science Gravity Solver
=========================================
Final optimized version without recursion issues
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Physical constants (SI units)
c = 2.998e8  # m/s
G_inf = 6.674e-11  # m³/kg/s²
hbar = 1.055e-34  # J·s
m_p = 1.673e-27  # kg
e = 1.602e-19  # C

# Unit conversions
kpc_to_m = 3.086e19
pc_to_m = 3.086e16
km_to_m = 1000
Msun = 1.989e30
nm_to_m = 1e-9
um_to_m = 1e-6

# Recognition Science constants
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
chi = phi / np.pi
beta_0 = -(phi - 1) / phi**5  # Base beta value

# Recognition lengths
lambda_eff_0 = 63.0 * um_to_m
ell_1 = 0.97 * kpc_to_m
ell_2 = 24.3 * kpc_to_m

# Information field parameters
L_0 = 0.335 * nm_to_m
g_dagger = 1.2e-10


@dataclass
class GalaxyData:
    """Galaxy rotation curve data"""
    name: str
    R_kpc: np.ndarray
    v_obs: np.ndarray
    v_err: np.ndarray
    sigma_gas: np.ndarray
    sigma_disk: np.ndarray
    sigma_bulge: Optional[np.ndarray] = None


@dataclass
class GalaxyParameters:
    """Per-galaxy tunable parameters"""
    ML_disk: float = 0.5
    ML_bulge: float = 0.7
    gas_factor: float = 1.33
    h_scale: float = 300


class RobustGravitySolver:
    """Robust RS gravity solver"""
    
    def __init__(self, 
                 lambda_eff: float = lambda_eff_0, 
                 beta_scale: float = 1.0,
                 mu_scale: float = 1.0,
                 coupling_scale: float = 1.0):
        """Initialize with global parameters"""
        self.lambda_eff = lambda_eff
        self.beta = beta_0 * beta_scale
        
        # Derived parameters
        self.I_star = m_p * c**2 / L_0**3
        self.mu_field = (hbar / (c * ell_1)) * mu_scale
        self.lambda_coupling = np.sqrt(g_dagger * c**2 / self.I_star) * coupling_scale
    
    def Xi_kernel(self, u):
        """Kernel function - vectorized and stable"""
        u = np.atleast_1d(u)
        result = np.ones_like(u, dtype=float)
        
        # Valid range
        mask = (u > -1) & (np.abs(u) > 1e-10)
        if np.any(mask):
            with np.errstate(over='ignore', invalid='ignore'):
                result[mask] = (np.power(1 + u[mask], self.beta) - 1) / (self.beta * u[mask])
                # Handle any NaN/inf
                bad = ~np.isfinite(result[mask])
                if np.any(bad):
                    result[mask][bad] = 1.0
        
        # u <= -1 gives NaN
        result[u <= -1] = np.nan
        
        return result if len(result) > 1 else result[0]
    
    def F_kernel_vectorized(self, r):
        """Recognition kernel - fully vectorized"""
        r = np.atleast_1d(r)
        
        u1 = r / ell_1
        u2 = r / ell_2
        
        # Compute Xi values
        Xi1 = self.Xi_kernel(u1)
        Xi2 = self.Xi_kernel(u2)
        
        # Numerical derivatives
        eps = 1e-6
        dXi1 = np.zeros_like(u1)
        dXi2 = np.zeros_like(u2)
        
        # Only compute derivatives where needed
        mask1 = np.abs(u1) > 1e-10
        if np.any(mask1):
            Xi1_plus = self.Xi_kernel(u1[mask1] + eps)
            Xi1_minus = self.Xi_kernel(u1[mask1] - eps)
            dXi1[mask1] = (Xi1_plus - Xi1_minus) / (2 * eps)
        
        mask2 = np.abs(u2) > 1e-10
        if np.any(mask2):
            Xi2_plus = self.Xi_kernel(u2[mask2] + eps)
            Xi2_minus = self.Xi_kernel(u2[mask2] - eps)
            dXi2[mask2] = (Xi2_plus - Xi2_minus) / (2 * eps)
        
        F1 = Xi1 - u1 * dXi1
        F2 = Xi2 - u2 * dXi2
        
        result = F1 + F2
        
        # Clean up
        result[~np.isfinite(result)] = 1.0
        
        return result if len(result) > 1 else result[0]
    
    def G_running_vectorized(self, r):
        """Scale-dependent Newton constant - vectorized"""
        r = np.atleast_1d(r)
        G = np.ones_like(r) * G_inf
        
        # Nanoscale enhancement
        mask_nano = r < 100 * nm_to_m
        if np.any(mask_nano):
            G[mask_nano] = G_inf * np.power(self.lambda_eff / r[mask_nano], -self.beta)
        
        # Transition region
        mask_trans = (r >= 100 * nm_to_m) & (r < 0.1 * ell_1)
        if np.any(mask_trans):
            G[mask_trans] = G_inf * np.power(self.lambda_eff / r[mask_trans], self.beta/2)
        
        # Galactic scales
        mask_gal = r >= 0.1 * ell_1
        if np.any(mask_gal):
            G_gal = G_inf * np.power(ell_1 / r[mask_gal], self.beta)
            F = self.F_kernel_vectorized(r[mask_gal])
            G[mask_gal] = G_gal * F
        
        return G if len(G) > 1 else G[0]
    
    def mond_interpolation(self, u):
        """MOND function - vectorized"""
        return u / np.sqrt(1 + u**2)
    
    def solve_information_field(self, R_kpc, B_R):
        """Solve information field equation"""
        R = R_kpc * kpc_to_m
        
        def field_equation(y, r):
            rho_I = max(y[0], 1e-50)
            drho_dr = y[1]
            
            if r < R[0]:
                return [0, 0]
            
            u = abs(drho_dr) / (self.I_star * self.mu_field)
            mu_u = self.mond_interpolation(u)
            
            B_local = np.interp(r, R, B_R, left=B_R[0], right=0)
            
            if mu_u > 1e-10 and r > 0:
                d2rho = (self.mu_field**2 * rho_I - self.lambda_coupling * B_local) / mu_u
                d2rho -= (2/r) * drho_dr
            else:
                d2rho = 0
            
            return [drho_dr, d2rho]
        
        # Initial conditions
        rho_I_0 = B_R[0] * self.lambda_coupling / self.mu_field**2
        y0 = [rho_I_0, 0]
        
        # Solve
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            solution = odeint(field_equation, y0, R, rtol=1e-8, atol=1e-10)
        
        rho_I = solution[:, 0]
        drho_dr = solution[:, 1]
        
        return np.maximum(rho_I, 0), drho_dr
    
    def solve_galaxy(self, galaxy: GalaxyData, params: Optional[GalaxyParameters] = None):
        """Solve galaxy rotation curve"""
        if params is None:
            params = GalaxyParameters()
        
        R = galaxy.R_kpc * kpc_to_m
        n_points = len(R)
        
        # Surface density with per-galaxy parameters
        sigma_gas = galaxy.sigma_gas * params.gas_factor
        sigma_disk = galaxy.sigma_disk * params.ML_disk
        sigma_total = sigma_gas + sigma_disk
        
        if galaxy.sigma_bulge is not None:
            sigma_bulge = galaxy.sigma_bulge * params.ML_bulge
            sigma_total += sigma_bulge
        
        sigma_SI = sigma_total * Msun / pc_to_m**2
        
        # Enclosed mass - vectorized
        M_enc = np.zeros(n_points)
        for i in range(n_points):
            if i == 0:
                M_enc[i] = np.pi * R[i]**2 * sigma_SI[i]
            else:
                r_vals = R[:i+1]
                sigma_vals = sigma_SI[:i+1]
                integrand = 2 * np.pi * r_vals * sigma_vals
                M_enc[i] = np.trapz(integrand, r_vals)
        
        # Baryon density
        h_scale_m = params.h_scale * pc_to_m
        rho_baryon = np.zeros(n_points)
        
        mask = R > 0
        if np.any(mask):
            z_scale = h_scale_m * np.exp(-R[mask]/(3*kpc_to_m))
            V_eff = 2 * np.pi * R[mask]**2 * (2 * z_scale)
            rho_baryon[mask] = M_enc[mask] / V_eff
        
        # Solve field
        B_R = rho_baryon * c**2
        rho_I, drho_dr = self.solve_information_field(galaxy.R_kpc, B_R)
        
        # Get G values - vectorized
        G_values = self.G_running_vectorized(R)
        
        # Accelerations - vectorized
        a_newton = np.zeros(n_points)
        mask = (R > 0) & (M_enc > 0)
        a_newton[mask] = G_values[mask] * M_enc[mask] / R[mask]**2
        
        a_info = (self.lambda_coupling / c**2) * np.abs(drho_dr)
        
        # Total acceleration with MOND interpolation
        x = a_newton / g_dagger
        
        # Deep MOND regime
        mask_deep = x < 0.01
        a_total = np.zeros(n_points)
        a_total[mask_deep] = np.sqrt(a_newton[mask_deep] * g_dagger)
        
        # Transition and Newtonian regime
        mask_trans = ~mask_deep
        if np.any(mask_trans):
            u = np.abs(drho_dr[mask_trans]) / (self.I_star * self.mu_field)
            mu_u = self.mond_interpolation(u)
            nu = self.mond_interpolation(x[mask_trans])
            a_mond = np.sqrt(a_newton[mask_trans] * g_dagger)
            a_total[mask_trans] = nu * a_newton[mask_trans] + (1 - nu) * a_mond + a_info[mask_trans] * mu_u
        
        # Velocities
        v_model = np.sqrt(np.maximum(a_total * R, 0)) / km_to_m
        v_newton = np.sqrt(np.maximum(a_newton * R, 0)) / km_to_m
        
        # Chi-squared
        residuals = galaxy.v_obs - v_model
        chi2 = np.sum((residuals / galaxy.v_err)**2)
        chi2_reduced = chi2 / len(galaxy.v_obs)
        
        return {
            'v_model': v_model,
            'v_newton': v_newton,
            'a_newton': a_newton,
            'a_info': a_info,
            'a_total': a_total,
            'rho_I': rho_I,
            'chi2': chi2,
            'chi2_reduced': chi2_reduced,
            'residuals': residuals,
            'M_enc': M_enc,
            'rho_baryon': rho_baryon
        }


def optimize_galaxy_params(galaxy: GalaxyData, solver: RobustGravitySolver, 
                         max_iter: int = 100) -> Tuple[GalaxyParameters, float]:
    """Optimize per-galaxy parameters"""
    
    def objective(x):
        if galaxy.sigma_bulge is not None:
            params = GalaxyParameters(
                ML_disk=x[0],
                ML_bulge=x[1],
                gas_factor=x[2],
                h_scale=x[3]
            )
        else:
            params = GalaxyParameters(
                ML_disk=x[0],
                ML_bulge=0.7,
                gas_factor=x[1],
                h_scale=x[2]
            )
        
        try:
            result = solver.solve_galaxy(galaxy, params)
            return result['chi2_reduced']
        except:
            return 1e6
    
    # Setup
    if galaxy.sigma_bulge is not None:
        x0 = [0.5, 0.7, 1.33, 300]
        bounds = [(0.3, 1.0), (0.3, 0.9), (1.25, 1.40), (100, 600)]
    else:
        x0 = [0.5, 1.33, 300]
        bounds = [(0.3, 1.0), (1.25, 1.40), (100, 600)]
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', 
                     bounds=bounds, options={'maxiter': max_iter})
    
    # Extract parameters
    if galaxy.sigma_bulge is not None:
        opt_params = GalaxyParameters(
            ML_disk=result.x[0],
            ML_bulge=result.x[1],
            gas_factor=result.x[2],
            h_scale=result.x[3]
        )
    else:
        opt_params = GalaxyParameters(
            ML_disk=result.x[0],
            ML_bulge=0.7,
            gas_factor=result.x[1],
            h_scale=result.x[2]
        )
    
    return opt_params, result.fun 