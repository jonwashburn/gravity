#!/usr/bin/env python3
"""
Enhanced Tunable Recognition Science Gravity Solver
===================================================
Version with both global and per-galaxy tunable parameters
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import warnings

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

# Energy and time scales
E_coh = 0.090 * e
tau_0 = 7.33e-15

# Base recognition lengths
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
    ML_disk: float = 0.5      # Disk mass-to-light ratio
    ML_bulge: float = 0.7     # Bulge mass-to-light ratio
    gas_factor: float = 1.33  # Gas helium correction factor
    h_scale: float = 300      # Scale height in pc
    
    def to_dict(self):
        return {
            'ML_disk': self.ML_disk,
            'ML_bulge': self.ML_bulge,
            'gas_factor': self.gas_factor,
            'h_scale': self.h_scale
        }


class EnhancedGravitySolver:
    """Enhanced RS gravity solver with full parameter control"""
    
    def __init__(self, 
                 # Global parameters
                 lambda_eff: float = lambda_eff_0, 
                 beta_scale: float = 1.0,
                 mu_scale: float = 1.0,
                 coupling_scale: float = 1.0,
                 # Solver options
                 use_stiff_solver: bool = True,
                 inner_radius_kpc: float = 1.0):
        """
        Initialize with global parameters
        
        Parameters:
        -----------
        lambda_eff : float
            Effective recognition length (m)
        beta_scale : float
            Scaling factor for beta exponent
        mu_scale : float
            Scaling factor for field mass parameter
        coupling_scale : float
            Scaling factor for information field coupling
        use_stiff_solver : bool
            Use stiff ODE solver for r < inner_radius_kpc
        inner_radius_kpc : float
            Radius below which to use stiff solver
        """
        self.lambda_eff = lambda_eff
        self.beta = beta_0 * beta_scale
        self.use_stiff_solver = use_stiff_solver
        self.inner_radius_kpc = inner_radius_kpc
        
        # Derived parameters
        self.I_star = m_p * c**2 / L_0**3
        self.mu_field = (hbar / (c * ell_1)) * mu_scale
        self.lambda_coupling = np.sqrt(g_dagger * c**2 / self.I_star) * coupling_scale
    
    # Core functions
    
    def J_cost(self, x):
        """Self-dual cost functional"""
        return 0.5 * (x + 1.0/x)
    
    def Xi_kernel(self, u):
        """Kernel function"""
        if abs(u) < 1e-10:
            return 1.0
        elif u <= -1:
            return np.nan
        else:
            try:
                # Avoid numerical issues
                if abs(self.beta) < 1e-10:
                    return 1.0
                result = (np.power(1 + u, self.beta) - 1) / (self.beta * u)
                if np.isnan(result) or np.isinf(result):
                    return 1.0
                return result
            except:
                return 1.0
    
    def F_kernel(self, r):
        """Recognition kernel"""
        try:
            u1 = r / ell_1
            u2 = r / ell_2
            
            Xi1 = self.Xi_kernel(u1)
            Xi2 = self.Xi_kernel(u2)
            
            # Avoid recursion in derivative calculation
            eps = 1e-6
            if abs(u1) > 1e-10:
                Xi1_plus = self.Xi_kernel(u1 + eps)
                Xi1_minus = self.Xi_kernel(u1 - eps)
                dXi1 = (Xi1_plus - Xi1_minus) / (2 * eps)
            else:
                dXi1 = 0
                
            if abs(u2) > 1e-10:
                Xi2_plus = self.Xi_kernel(u2 + eps)
                Xi2_minus = self.Xi_kernel(u2 - eps)
                dXi2 = (Xi2_plus - Xi2_minus) / (2 * eps)
            else:
                dXi2 = 0
            
            F1 = Xi1 - u1 * dXi1
            F2 = Xi2 - u2 * dXi2
            
            result = F1 + F2
            if np.isnan(result) or np.isinf(result):
                return 1.0
            return result
        except:
            return 1.0
    
    def G_running(self, r):
        """Scale-dependent Newton constant"""
        if r < 100 * nm_to_m:
            # Nanoscale enhancement
            return G_inf * np.power(self.lambda_eff / r, -self.beta)
        elif r < 0.1 * ell_1:
            # Transition
            return G_inf * np.power(self.lambda_eff / r, self.beta/2)
        else:
            # Galactic
            G_gal = G_inf * np.power(ell_1 / r, self.beta)
            return G_gal * self.F_kernel(r)
    
    def mond_interpolation(self, u):
        """MOND function"""
        return u / np.sqrt(1 + u**2)
    
    # Enhanced information field solver
    
    def solve_information_field(self, R_kpc, B_R):
        """Solve information field equation with adaptive solver"""
        R = R_kpc * kpc_to_m
        
        def field_equation(r, y):
            """Field equation for solve_ivp (note argument order)"""
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
        
        def field_equation_odeint(y, r):
            """Field equation for odeint (note argument order)"""
            return field_equation(r, y)
        
        # Initial conditions
        rho_I_0 = B_R[0] * self.lambda_coupling / self.mu_field**2
        y0 = [rho_I_0, 0]
        
        # Split solution at inner radius if using stiff solver
        if self.use_stiff_solver and R_kpc[0] < self.inner_radius_kpc:
            # Find split point
            split_idx = np.searchsorted(R_kpc, self.inner_radius_kpc)
            if split_idx > 0 and split_idx < len(R):
                # Inner region - stiff solver
                R_inner = R[:split_idx+1]
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    sol_inner = solve_ivp(field_equation, 
                                        [R_inner[0], R_inner[-1]], 
                                        y0,
                                        t_eval=R_inner,
                                        method='Radau',
                                        rtol=1e-10,
                                        atol=1e-12)
                
                # Outer region - standard solver
                if split_idx < len(R) - 1:
                    R_outer = R[split_idx:]
                    y0_outer = [sol_inner.y[0, -1], sol_inner.y[1, -1]]
                    sol_outer = odeint(field_equation_odeint, y0_outer, R_outer, 
                                     rtol=1e-8, atol=1e-10)
                    
                    # Combine solutions
                    rho_I = np.concatenate([sol_inner.y[0, :-1], sol_outer[:, 0]])
                    drho_dr = np.concatenate([sol_inner.y[1, :-1], sol_outer[:, 1]])
                else:
                    rho_I = sol_inner.y[0]
                    drho_dr = sol_inner.y[1]
            else:
                # All in outer region
                solution = odeint(field_equation_odeint, y0, R, rtol=1e-8, atol=1e-10)
                rho_I = solution[:, 0]
                drho_dr = solution[:, 1]
        else:
            # Standard solver for all
            solution = odeint(field_equation_odeint, y0, R, rtol=1e-8, atol=1e-10)
            rho_I = solution[:, 0]
            drho_dr = solution[:, 1]
        
        return np.maximum(rho_I, 0), drho_dr
    
    # Galaxy solver with per-galaxy parameters
    
    def solve_galaxy(self, galaxy: GalaxyData, params: Optional[GalaxyParameters] = None):
        """Solve galaxy rotation curve with optional per-galaxy parameters"""
        if params is None:
            params = GalaxyParameters()
        
        R = galaxy.R_kpc * kpc_to_m
        
        # Surface density with per-galaxy parameters
        sigma_gas = galaxy.sigma_gas * params.gas_factor
        sigma_disk = galaxy.sigma_disk * params.ML_disk
        sigma_total = sigma_gas + sigma_disk
        
        if galaxy.sigma_bulge is not None:
            sigma_bulge = galaxy.sigma_bulge * params.ML_bulge
            sigma_total += sigma_bulge
        
        sigma_SI = sigma_total * Msun / pc_to_m**2
        
        # Enclosed mass (improved)
        M_enc = np.zeros_like(R)
        for i in range(len(R)):
            if i == 0:
                M_enc[i] = np.pi * R[i]**2 * sigma_SI[i]
            else:
                # Trapezoidal integration
                r_vals = R[:i+1]
                sigma_vals = sigma_SI[:i+1]
                integrand = 2 * np.pi * r_vals * sigma_vals
                M_enc[i] = np.trapz(integrand, r_vals)
        
        # Baryon density with scale height
        h_scale_m = params.h_scale * pc_to_m
        rho_baryon = np.zeros_like(R)
        for i in range(len(R)):
            if R[i] > 0:
                # Exponential disk profile
                z_scale = h_scale_m * np.exp(-R[i]/(3*kpc_to_m))
                V_eff = 2 * np.pi * R[i]**2 * (2 * z_scale)
                rho_baryon[i] = M_enc[i] / V_eff
        
        # Solve field
        B_R = rho_baryon * c**2
        rho_I, drho_dr = self.solve_information_field(galaxy.R_kpc, B_R)
        
        # Accelerations
        a_newton = np.zeros_like(R)
        a_info = np.zeros_like(R)
        
        for i, r in enumerate(R):
            if r > 0 and M_enc[i] > 0:
                G_r = self.G_running(r)
                a_newton[i] = G_r * M_enc[i] / r**2
                a_info[i] = (self.lambda_coupling / c**2) * abs(drho_dr[i])
        
        # Total acceleration
        a_total = np.zeros_like(R)
        for i in range(len(R)):
            x = a_newton[i] / g_dagger
            
            if x < 0.01:
                a_total[i] = np.sqrt(a_newton[i] * g_dagger)
            else:
                u = abs(drho_dr[i]) / (self.I_star * self.mu_field)
                mu_u = self.mond_interpolation(u)
                nu = self.mond_interpolation(x)
                a_mond = np.sqrt(a_newton[i] * g_dagger)
                # Smooth transition
                a_total[i] = nu * a_newton[i] + (1 - nu) * a_mond + a_info[i] * mu_u
        
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
            'rho_baryon': rho_baryon,
            'params_used': params.to_dict()
        } 