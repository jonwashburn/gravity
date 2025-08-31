#!/usr/bin/env python3
"""
SPARC-LNAL Analysis: Testing Information Gradient Dark Matter
against observed galaxy rotation curves
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
from scipy.optimize import curve_fit

# Physical constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 299792458    # m/s
pc_to_m = 3.086e16  # parsec to meters
kpc_to_m = 1000 * pc_to_m
M_sun = 1.989e30  # kg

# LNAL parameters
L0 = 3.35e-10    # voxel size in meters
tau0 = 7.33e-15  # fundamental time in seconds

class LNALGalaxyModel:
    """Model galaxy rotation curves using LNAL information gradients"""
    
    def __init__(self, M_stars, M_gas, R_stars, R_gas):
        """
        Initialize with baryonic parameters
        M_stars, M_gas: stellar and gas mass (solar masses)
        R_stars, R_gas: scale radii (kpc)
        """
        self.M_stars = M_stars * M_sun
        self.M_gas = M_gas * M_sun
        self.R_stars = R_stars * kpc_to_m
        self.R_gas = R_gas * kpc_to_m
        
    def baryon_density(self, r):
        """Baryonic mass density profile (exponential disk)"""
        rho_stars = self.M_stars / (2*np.pi*self.R_stars**2) * np.exp(-r/self.R_stars)
        rho_gas = self.M_gas / (2*np.pi*self.R_gas**2) * np.exp(-r/self.R_gas)
        return rho_stars + rho_gas
    
    def information_density(self, r):
        """
        Information density from baryonic matter
        Key assumption: I ∝ ρ_baryon × complexity_factor
        """
        rho_b = self.baryon_density(r)
        
        # Complexity factor: higher in disk, lower in halo
        # This encodes pattern maintenance cost
        z_scale = 0.1 * self.R_stars  # vertical scale height
        complexity = 1 + 10 * np.exp(-r/(5*self.R_stars))
        
        # Information density in natural units
        I_density = rho_b * c**2 * complexity
        return I_density
    
    def information_gradient_dm(self, r):
        """
        Dark matter density from information gradients
        ρ_DM = (c²/8πG) |∇I|²/I
        """
        # Numerical gradient
        dr = 0.01 * self.R_stars
        if r < dr:
            r = dr
            
        I_plus = self.information_density(r + dr)
        I_minus = self.information_density(r - dr)
        I_center = self.information_density(r)
        
        grad_I = (I_plus - I_minus) / (2 * dr)
        
        # Avoid division by zero
        if I_center < 1e-10:
            return 0
            
        rho_dm = (c**2 / (8*np.pi*G)) * (grad_I**2 / I_center)
        return rho_dm
    
    def enclosed_mass(self, r):
        """Total enclosed mass including dark matter"""
        # Integrate densities
        r_values = np.linspace(0, r, 1000)
        
        M_baryon = 0
        M_dark = 0
        
        for i in range(len(r_values)-1):
            r_mid = (r_values[i] + r_values[i+1]) / 2
            dr = r_values[i+1] - r_values[i]
            
            # Cylindrical approximation for disk
            dV = 2 * np.pi * r_mid * dr * self.R_stars  # scale height
            
            M_baryon += self.baryon_density(r_mid) * dV
            M_dark += self.information_gradient_dm(r_mid) * dV
            
        return M_baryon + M_dark
    
    def rotation_curve(self, r_kpc):
        """Calculate rotation velocity at radius r (in kpc)"""
        r = r_kpc * kpc_to_m
        M_enc = self.enclosed_mass(r)
        v_rot = np.sqrt(G * M_enc / r)
        return v_rot / 1000  # return in km/s

def analyze_sparc_galaxy(galaxy_data):
    """
    Analyze a single SPARC galaxy
    galaxy_data should contain:
    - R: radii in kpc
    - V_obs: observed velocities in km/s
    - V_gas: gas contribution
    - V_stars: stellar contribution
    - morphology: galaxy type
    """
    
    # Extract baryonic parameters from V_stars and V_gas
    # This is simplified - real analysis would fit profiles
    M_stars_est = 1e10  # solar masses (example)
    M_gas_est = 1e9
    R_stars_est = 3.0   # kpc
    R_gas_est = 5.0
    
    # Create LNAL model
    model = LNALGalaxyModel(M_stars_est, M_gas_est, R_stars_est, R_gas_est)
    
    # Calculate predicted rotation curve
    R = galaxy_data['R']
    V_pred = np.array([model.rotation_curve(r) for r in R])
    
    # Compare to observations
    V_obs = galaxy_data['V_obs']
    chi2 = np.sum((V_pred - V_obs)**2 / V_obs**2)
    
    return {
        'V_pred': V_pred,
        'chi2': chi2,
        'model': model
    }

def test_morphology_dependence(sparc_catalog):
    """
    Test LNAL prediction that morphology affects dark matter content
    """
    results_by_type = {
        'spiral': [],
        'elliptical': [],
        'irregular': []
    }
    
    for galaxy in sparc_catalog:
        result = analyze_sparc_galaxy(galaxy)
        galaxy_type = galaxy['morphology']
        results_by_type[galaxy_type].append(result['chi2'])
    
    # LNAL predicts spirals should have best fits (lowest chi2)
    # because their pattern complexity is well-defined
    
    return results_by_type

def plot_lnal_vs_mond(galaxy_data):
    """
    Compare LNAL predictions to MOND and observations
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Top panel: rotation curves
    R = galaxy_data['R']
    ax1.plot(R, galaxy_data['V_obs'], 'ko', label='Observed')
    ax1.plot(R, galaxy_data['V_newton'], 'b--', label='Newtonian')
    
    # Add LNAL prediction
    model = LNALGalaxyModel(1e10, 1e9, 3.0, 5.0)  # example parameters
    V_lnal = [model.rotation_curve(r) for r in R]
    ax1.plot(R, V_lnal, 'r-', label='LNAL')
    
    ax1.set_xlabel('Radius (kpc)')
    ax1.set_ylabel('V_rot (km/s)')
    ax1.legend()
    ax1.set_title('Galaxy Rotation Curve')
    
    # Bottom panel: information gradient
    r_fine = np.linspace(0.1, max(R), 100)
    rho_dm = [model.information_gradient_dm(r * kpc_to_m) for r in r_fine]
    
    ax2.semilogy(r_fine, rho_dm)
    ax2.set_xlabel('Radius (kpc)')
    ax2.set_ylabel('Dark Matter Density (kg/m³)')
    ax2.set_title('LNAL Information Gradient Dark Matter')
    
    plt.tight_layout()
    plt.savefig('lnal_galaxy_test.png')
    
def main():
    """
    Main analysis workflow
    """
    print("LNAL-SPARC Analysis Framework")
    print("="*40)
    
    # Load SPARC data (placeholder)
    # In reality, would load from SPARC database files
    sample_galaxy = {
        'name': 'NGC_test',
        'R': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # kpc
        'V_obs': np.array([80, 120, 145, 155, 160, 162, 163, 163, 162, 160]),  # km/s
        'V_newton': np.array([80, 95, 85, 75, 65, 58, 52, 47, 43, 40]),
        'morphology': 'spiral'
    }
    
    # Analyze
    result = analyze_sparc_galaxy(sample_galaxy)
    print(f"Chi-squared: {result['chi2']:.2f}")
    
    # Plot
    plot_lnal_vs_mond(sample_galaxy)
    print("Saved plot to lnal_galaxy_test.png")
    
    # Key insight for LNAL development
    print("\nKey Challenge for LNAL:")
    print("The information gradient formula needs a universal scale")
    print("to match the observed g† = 1.2e-10 m/s² in SPARC data.")
    print("This might come from the cosmic information background.")

if __name__ == "__main__":
    main() 