#!/usr/bin/env python3
"""
LNAL SPARC Solver Adapter
========================
Bridges the corrected LNALGravity framework with the SPARC analysis pipeline.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from lnal_complete_corrected import LNALGravity

@dataclass
class GalaxyData:
    """Galaxy data structure expected by SPARC pipeline"""
    name: str
    R_kpc: np.ndarray
    v_obs: np.ndarray
    v_err: np.ndarray
    sigma_gas: np.ndarray
    sigma_disk: np.ndarray
    sigma_bulge: Optional[np.ndarray] = None

class UnifiedGravitySolver:
    """Adapter wrapping LNALGravity for SPARC analysis"""
    
    def __init__(self):
        self.lnal = LNALGravity()
        
    def solve_galaxy(self, galaxy: GalaxyData) -> Dict:
        """
        Solve for rotation curve using LNAL gravity
        
        Returns dict with required fields:
        - v_model: model velocities
        - v_baryon: Newtonian baryon velocities  
        - a_baryon: Newtonian accelerations
        - a_total: Total accelerations
        - chi2_reduced: Reduced chi-squared
        """
        # Convert surface densities to masses
        # Integrate Sigma(R) to get M(<R) assuming exponential disk
        R_kpc = galaxy.R_kpc
        
        # Disk mass from surface density
        # For exponential disk: Sigma(R) = Sigma_0 * exp(-R/R_d)
        # Total mass M_disk = 2*pi*Sigma_0*R_d^2
        
        # Find scale length by fitting exponential to outer disk
        if len(R_kpc) > 5:
            # Use outer half for fit
            mid = len(R_kpc) // 2
            R_fit = R_kpc[mid:]
            sigma_fit = galaxy.sigma_disk[mid:]
            
            # Log-linear fit for R_d
            mask = sigma_fit > 0
            if np.sum(mask) > 3:
                p = np.polyfit(R_fit[mask], np.log(sigma_fit[mask]), 1)
                R_d = -1.0 / p[0]  # Scale length in kpc
                Sigma_0 = np.exp(p[1])
                M_disk = 2 * np.pi * Sigma_0 * R_d**2 * 1e6  # Convert to M_sun
            else:
                # Fallback: assume R_d = R_max/4
                R_d = R_kpc[-1] / 4
                M_disk = np.sum(galaxy.sigma_disk) * np.pi * (R_kpc[1] - R_kpc[0])**2 * 1e6
        else:
            R_d = R_kpc[-1] / 4
            M_disk = 1e10  # Default 10^10 M_sun
            
        # Gas mass (similar treatment)
        if np.any(galaxy.sigma_gas > 0):
            # Gas typically more extended
            R_gas = R_d * 2.0
            M_gas = np.sum(galaxy.sigma_gas) * np.pi * (R_kpc[1] - R_kpc[0])**2 * 1e6
        else:
            M_gas = 0
            R_gas = R_d
            
        # Ensure reasonable bounds
        M_disk = np.clip(M_disk, 1e8, 1e12)
        M_gas = np.clip(M_gas, 0, 1e11)
        R_d = np.clip(R_d, 0.5, 20.0)
        
        # Get rotation curves from LNAL
        v_newton, v_total, mu_values = self.lnal.galaxy_rotation_curve(
            R_kpc, M_disk, R_d, M_gas, R_gas
        )
        
        # Calculate accelerations
        G = 6.67430e-11
        kpc = 3.086e19
        M_sun = 1.989e30
        
        # Enclosed mass for Newtonian acceleration
        M_enc = []
        for i, r in enumerate(R_kpc):
            # Disk contribution
            x_d = r / R_d
            M_enc_disk = M_disk * (1 - (1 + x_d) * np.exp(-x_d))
            
            # Gas contribution  
            if M_gas > 0 and r < R_gas:
                M_enc_gas = M_gas * (r / R_gas)**2 * (3 - 2*r/R_gas)
            else:
                M_enc_gas = M_gas if M_gas > 0 else 0
                
            M_enc.append(M_enc_disk + M_enc_gas)
            
        M_enc = np.array(M_enc) * M_sun
        r_m = R_kpc * kpc
        
        a_newton = G * M_enc / r_m**2
        a_total = (v_total * 1000)**2 / r_m
        
        # Chi-squared calculation
        residuals = galaxy.v_obs - v_total
        chi2 = np.sum((residuals / galaxy.v_err)**2)
        chi2_reduced = chi2 / (len(galaxy.v_obs) - 2)  # 2 effective parameters
        
        return {
            'v_model': v_total,
            'v_baryon': v_newton,
            'a_baryon': a_newton,
            'a_total': a_total,
            'chi2_reduced': chi2_reduced,
            'M_disk': M_disk,
            'R_d': R_d,
            'M_gas': M_gas
        } 