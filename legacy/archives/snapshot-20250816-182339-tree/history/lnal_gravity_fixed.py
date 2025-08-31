#!/usr/bin/env python3
"""
LNAL Gravity Fixed: Proper unit handling
========================================
Fixes the unit conversion issues in galaxy rotation curves.
"""

import numpy as np
from scipy.integrate import quad

# Recognition Science constants
phi = (1 + np.sqrt(5)) / 2
E_coh_eV = 0.090  # eV
tau_0 = 7.33e-15  # s
T_8beat = 8 * tau_0

# Physical constants
c = 299792458  # m/s
G = 6.67430e-11  # m³/kg/s²
H_0 = 70e3 / 3.086e22  # Hubble constant in 1/s
t_Hubble = 1 / H_0

# Convert units
kpc = 3.086e19  # m
M_sun = 1.989e30  # kg

class LNALGravityFixed:
    """LNAL gravity with corrected unit handling"""
    
    def __init__(self):
        # Base calculation
        a_0_base = c**2 * T_8beat / t_Hubble
        
        # 4D voxel counting correction
        voxel_factor = 8**4  # 4096
        metric_factor = (10/8)**4  # 2.441406...
        
        # Corrected value
        self.a_0 = a_0_base * voxel_factor * metric_factor
        
        # Recognition lengths (hop kernel poles)
        self.L_1 = 0.97 * kpc
        self.L_2 = 24.3 * kpc
    
    def mu(self, x):
        """MOND interpolation function"""
        return x / np.sqrt(1 + x**2)
    
    def galaxy_rotation_curve(self, r_kpc, M_disk, R_d, M_gas=0, R_gas=None):
        """
        Calculate rotation curve with proper units
        
        Parameters:
        -----------
        r_kpc : array, radii in kpc
        M_disk : float, disk mass in M_sun
        R_d : float, disk scale length in kpc
        M_gas : float, gas mass in M_sun
        R_gas : float, gas scale length in kpc
        
        Returns:
        --------
        v_newton : array, Newtonian velocities in km/s
        v_total : array, LNAL velocities in km/s
        mu_values : array, interpolation function values
        """
        # Convert to SI units
        r = r_kpc * kpc  # m
        
        # Calculate enclosed mass for exponential disk
        # M_enc(r) = M_disk * [1 - (1 + r/R_d) * exp(-r/R_d)]
        M_enc_disk = M_disk * (1 - (1 + r_kpc/R_d) * np.exp(-r_kpc/R_d))
        
        # Add gas contribution if present
        if M_gas > 0 and R_gas is not None:
            # Similar exponential profile for gas
            M_enc_gas = M_gas * (1 - (1 + r_kpc/R_gas) * np.exp(-r_kpc/R_gas))
        else:
            M_enc_gas = 0
        
        # Total enclosed mass in kg
        M_enc = (M_enc_disk + M_enc_gas) * M_sun
        
        # Newtonian acceleration
        a_newton = G * M_enc / r**2
        
        # LNAL modification
        x = a_newton / self.a_0
        mu_val = self.mu(x)
        
        # Total acceleration - pure MOND-like behavior
        a_total = a_newton / mu_val
        
        # Convert to velocities in km/s
        v_newton = np.sqrt(a_newton * r) / 1000
        v_total = np.sqrt(a_total * r) / 1000
        
        return v_newton, v_total, mu_val 