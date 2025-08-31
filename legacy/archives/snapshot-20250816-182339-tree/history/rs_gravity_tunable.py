#!/usr/bin/env python3
"""
Tunable Recognition Science Gravity Solver
==========================================
Version with adjustable parameters for optimization
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

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


class TunableGravitySolver:
    """RS gravity solver with tunable parameters"""
    
    def __init__(self, lambda_eff: float = lambda_eff_0, 
                 h_scale: float = 300 * pc_to_m,
                 beta_scale: float = 1.0,
                 mu_scale: float = 1.0,
                 coupling_scale: float = 1.0):
        """
        Initialize with tunable parameters
        
        Parameters:
        -----------
        lambda_eff : float
            Effective recognition length (m)
        h_scale : float
            Disk scale height (m)
        beta_scale : float
            Scaling factor for beta exponent
        mu_scale : float
            Scaling factor for field mass parameter
        coupling_scale : float
            Scaling factor for information field coupling
        """
        self.lambda_eff = lambda_eff
        self.h_scale = h_scale
        self.beta = beta_0 * beta_scale
        
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
            return (np.power(1 + u, self.beta) - 1) / (self.beta * u)
    
    def F_kernel(self, r):
        """Recognition kernel"""
        u1 = r / ell_1
        u2 = r / ell_2
        
        Xi1 = self.Xi_kernel(u1)
        Xi2 = self.Xi_kernel(u2)
        
        eps = 1e-6
        dXi1 = (self.Xi_kernel(u1 + eps) - self.Xi_kernel(u1 - eps)) / (2 * eps)
        dXi2 = (self.Xi_kernel(u2 + eps) - self.Xi_kernel(u2 - eps)) / (2 * eps)
        
        F1 = Xi1 - u1 * dXi1
        F2 = Xi2 - u2 * dXi2
        
        return F1 + F2
    
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
    
    # Information field solver
    
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
        
        rho_I_0 = B_R[0] * self.lambda_coupling / self.mu_field**2
        y0 = [rho_I_0, 0]
        
        solution = odeint(field_equation, y0, R, rtol=1e-8, atol=1e-10)
        rho_I = solution[:, 0]
        drho_dr = solution[:, 1]
        
        return np.maximum(rho_I, 0), drho_dr
    
    # Galaxy solver
    
    def solve_galaxy(self, galaxy):
        """Solve galaxy rotation curve"""
        R = galaxy.R_kpc * kpc_to_m
        
        # Surface density
        sigma_total = galaxy.sigma_gas + galaxy.sigma_disk
        if galaxy.sigma_bulge is not None:
            sigma_total += galaxy.sigma_bulge
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
        rho_baryon = np.zeros_like(R)
        for i in range(len(R)):
            if R[i] > 0:
                # Exponential disk profile
                z_scale = self.h_scale * np.exp(-R[i]/(3*kpc_to_m))
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
            'residuals': residuals
        } 