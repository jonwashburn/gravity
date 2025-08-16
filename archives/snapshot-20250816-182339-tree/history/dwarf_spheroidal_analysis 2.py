#!/usr/bin/env python3
"""
Dwarf Spheroidal Galaxy Analysis with Recognition Science Gravity
Tests the RS framework on pressure-supported systems without rotation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import json

# Physical constants
G_SI = 6.67430e-11  # m^3/kg/s^2
c = 299792458.0     # m/s
pc = 3.0857e16      # meters
kpc = 1000 * pc
M_sun = 1.989e30    # kg

# Recognition Science constants
phi = (1 + np.sqrt(5)) / 2
beta_0 = -(phi - 1) / phi**5
lambda_micro = 7.23e-36  # meters
lambda_eff = 50.8e-6     # meters (optimized)
ell_1 = 0.97 * kpc
ell_2 = 24.3 * kpc

# Optimized scale factors from SPARC analysis
beta_scale = 1.492
mu_scale = 1.644
coupling_scale = 1.326

# Known dwarf spheroidals with velocity dispersions
DWARF_SPHEROIDALS = {
    'Draco': {
        'sigma_v': 9.1,      # km/s velocity dispersion
        'r_half': 0.221,     # kpc half-light radius
        'M_V': -8.8,         # absolute V magnitude
        'distance': 76,      # kpc
        'references': 'Walker et al. 2009'
    },
    'Sculptor': {
        'sigma_v': 9.2,
        'r_half': 0.283,
        'M_V': -11.1,
        'distance': 86,
        'references': 'Walker et al. 2009'
    },
    'Carina': {
        'sigma_v': 6.6,
        'r_half': 0.250,
        'M_V': -9.1,
        'distance': 105,
        'references': 'Walker et al. 2009'
    },
    'Fornax': {
        'sigma_v': 11.7,
        'r_half': 0.710,
        'M_V': -13.4,
        'distance': 147,
        'references': 'Walker et al. 2009'
    },
    'Leo_I': {
        'sigma_v': 9.2,
        'r_half': 0.251,
        'M_V': -12.0,
        'distance': 254,
        'references': 'Mateo et al. 2008'
    },
    'Leo_II': {
        'sigma_v': 6.6,
        'r_half': 0.176,
        'M_V': -9.8,
        'distance': 233,
        'references': 'Koch et al. 2007'
    }
}

class DwarfSpheroidalSolver:
    """Solver for dwarf spheroidal galaxies using RS gravity"""
    
    def __init__(self, name, properties):
        self.name = name
        self.sigma_v = properties['sigma_v'] * 1000  # Convert to m/s
        self.r_half = properties['r_half'] * kpc     # Convert to meters
        self.M_V = properties['M_V']
        self.distance = properties['distance'] * kpc
        
        # Estimate stellar mass from M_V
        # Using M/L_V ~ 2 for old stellar populations
        L_V = 10**(-0.4 * (self.M_V - 4.83))  # Solar luminosities
        self.M_stellar = 2.0 * L_V * M_sun     # Assume M/L = 2
        
        # Plummer profile scale radius (related to half-light radius)
        self.a_plummer = self.r_half / 1.305
        
        # Recognition Science parameters
        self.beta = beta_scale * beta_0
        self.mu_0 = mu_scale * np.sqrt(c**2 / (8 * np.pi * G_SI))
        self.lambda_c = coupling_scale * G_SI / c**2
        
    def Xi_kernel(self, x):
        """Xi kernel for scale transitions (vectorized)"""
        scalar_input = np.isscalar(x)
        x = np.atleast_1d(x)
        result = np.zeros_like(x)
        
        # Low-x expansion
        low_mask = x < 0.01
        if np.any(low_mask):
            x_low = x[low_mask]
            result[low_mask] = (3/5)*x_low**2 - (3/7)*x_low**4 + (9/35)*x_low**6
        
        # High-x expansion
        high_mask = x > 100
        if np.any(high_mask):
            x_high = x[high_mask]
            result[high_mask] = (1 - 6/x_high**2 + 30/x_high**4 - 140/x_high**6)
        
        # Middle range - direct calculation
        mid_mask = ~(low_mask | high_mask)
        if np.any(mid_mask):
            x_mid = x[mid_mask]
            result[mid_mask] = 3 * (np.sin(x_mid) - x_mid * np.cos(x_mid)) / x_mid**3
        
        return float(result[0]) if scalar_input else result
    
    def F_kernel(self, r):
        """Recognition kernel F(r) (vectorized)"""
        scalar_input = np.isscalar(r)
        r = np.atleast_1d(r)
        
        # Galactic transition functions
        F1 = self.Xi_kernel(r / ell_1)
        F2 = self.Xi_kernel(r / ell_2)
        
        result = F1 + F2
        return float(result) if scalar_input else result
    
    def G_of_r(self, r):
        """Scale-dependent Newton constant"""
        # G(r) = G_inf * (lambda_eff/r)^beta * F(r)
        G_inf = G_SI
        
        # Power law component
        power_factor = (lambda_eff / r) ** self.beta
        
        # Recognition kernel
        F = self.F_kernel(r)
        
        return G_inf * power_factor * F
    
    def stellar_density(self, r):
        """Plummer profile for stellar density"""
        return (3 * self.M_stellar / (4 * np.pi * self.a_plummer**3)) * \
               (1 + (r/self.a_plummer)**2)**(-5/2)
    
    def jeans_equation_ode(self, y, r):
        """ODE for spherical Jeans equation with RS gravity
        
        d(ln sigma^2)/dr = -G(r) * M(<r) / (r^2 * sigma^2) - 2 * d(ln rho)/dr
        """
        ln_sigma2 = y[0]
        sigma2 = np.exp(ln_sigma2)
        
        # Avoid singularity at r=0
        if r < 1e-10:
            return [0.0]
        
        # Stellar density and its derivative
        rho = self.stellar_density(r)
        a = self.a_plummer
        dlnrho_dr = -5 * r / (a**2 + r**2)
        
        # Enclosed mass (Plummer profile)
        M_enc = self.M_stellar * r**3 / (r**2 + a**2)**(3/2)
        
        # Scale-dependent gravity - ensure scalar r
        G = float(self.G_of_r(np.array([r]))[0])
        
        # Jeans equation for d(ln sigma^2)/dr
        dlnsigma2_dr = -G * M_enc / (r**2 * sigma2) - 2 * dlnrho_dr
        
        return [dlnsigma2_dr]
    
    def solve_jeans(self, r_max=None):
        """Solve the Jeans equation to get velocity dispersion profile"""
        if r_max is None:
            r_max = 10 * self.r_half
        
        # Radial grid - start from small but non-zero radius
        r_min = 0.001 * self.a_plummer
        r = np.logspace(np.log10(r_min), np.log10(r_max), 1000)
        
        # Initial guess for central velocity dispersion
        # Use virial estimate: sigma^2 ~ G * M / R
        G_central = float(self.G_of_r(np.array([self.r_half]))[0])
        sigma_central = np.sqrt(G_central * self.M_stellar / self.r_half)
        
        # Start with this central value
        ln_sigma2_0 = np.log(sigma_central**2)
        y0 = [ln_sigma2_0]
        
        # Integrate outward
        sol = odeint(self.jeans_equation_ode, y0, r)
        ln_sigma2 = sol[:, 0]
        
        # Extract velocity dispersion
        sigma = np.sqrt(np.exp(ln_sigma2))
        
        # Check for numerical issues
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
        
        return r, sigma
    
    def calculate_mass_profile(self, r):
        """Calculate total dynamical mass profile from velocity dispersion"""
        # For pressure-supported system:
        # M_dyn(<r) = -r * sigma^2 / G * d(ln(rho*sigma^2))/d(ln r)
        
        r_prof, sigma_prof = self.solve_jeans()
        
        # Interpolate to desired radii
        sigma_interp = interp1d(r_prof, sigma_prof, 
                               bounds_error=False, fill_value='extrapolate')
        sigma = sigma_interp(r)
        
        # Use average G for simplicity (more accurate would integrate)
        G_avg = self.G_of_r(r)
        
        # Simple isothermal approximation for mass
        M_dyn = 3 * sigma**2 * r / G_avg
        
        return M_dyn
    
    def predict_dispersion(self):
        """Predict central velocity dispersion"""
        r, sigma = self.solve_jeans()
        
        # Average within half-light radius
        mask = r < self.r_half
        if np.any(mask):
            sigma_avg = np.mean(sigma[mask])
        else:
            sigma_avg = sigma[0]
        
        return sigma_avg / 1000  # Convert to km/s
    
    def plot_analysis(self, save=True):
        """Plot the analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Velocity dispersion profile
        ax = axes[0, 0]
        r, sigma = self.solve_jeans()
        ax.loglog(r/kpc, sigma/1000, 'b-', linewidth=2, label='RS gravity')
        ax.axhline(self.sigma_v/1000, color='r', linestyle='--', 
                  label=f'Observed: {self.sigma_v/1000:.1f} km/s')
        ax.axvline(self.r_half/kpc, color='gray', linestyle=':', alpha=0.5,
                  label='Half-light radius')
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Velocity Dispersion (km/s)')
        ax.set_title(f'{self.name} - Velocity Dispersion Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Enclosed mass profile
        ax = axes[0, 1]
        r_mass = np.logspace(np.log10(0.1*self.r_half), np.log10(10*self.r_half), 100)
        M_dyn = self.calculate_mass_profile(r_mass)
        M_stellar_enc = self.M_stellar * r_mass**3 / (r_mass**2 + self.a_plummer**2)**(3/2)
        
        ax.loglog(r_mass/kpc, M_dyn/M_sun, 'b-', linewidth=2, label='Dynamical (RS)')
        ax.loglog(r_mass/kpc, M_stellar_enc/M_sun, 'r--', linewidth=2, label='Stellar')
        ax.axvline(self.r_half/kpc, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Enclosed Mass (M☉)')
        ax.set_title('Mass Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. "Dark matter" fraction (actually RS gravity effect)
        ax = axes[1, 0]
        f_dm = 1 - M_stellar_enc / M_dyn
        ax.semilogx(r_mass/kpc, f_dm, 'g-', linewidth=2)
        ax.axvline(self.r_half/kpc, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('1 - M★/M_dyn')
        ax.set_title('RS Gravity Enhancement')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # 4. G(r) profile
        ax = axes[1, 1]
        r_g = np.logspace(np.log10(0.01*kpc), np.log10(100*kpc), 1000)
        G_r = self.G_of_r(r_g)
        ax.loglog(r_g/kpc, G_r/G_SI, 'purple', linewidth=2)
        ax.axvline(self.r_half/kpc, color='gray', linestyle=':', alpha=0.5, 
                  label='Half-light radius')
        ax.axhline(1, color='black', linestyle='--', alpha=0.5, label='G₀')
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('G(r)/G₀')
        ax.set_title('Scale-dependent Newton Constant')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'dwarf_spheroidal_{self.name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save numerical results
        results = {
            'name': self.name,
            'observed_sigma': self.sigma_v / 1000,  # km/s
            'predicted_sigma': self.predict_dispersion(),  # km/s
            'r_half': self.r_half / kpc,  # kpc
            'M_stellar': float(self.M_stellar / M_sun),  # M_sun
            'M_dyn_rhalf': float(M_dyn[np.argmin(np.abs(r_mass - self.r_half))] / M_sun),
            'M_to_L': float(M_dyn[np.argmin(np.abs(r_mass - self.r_half))] / (self.M_stellar/2)),
            'enhancement_rhalf': float(M_dyn[np.argmin(np.abs(r_mass - self.r_half))] / M_stellar_enc[np.argmin(np.abs(r_mass - self.r_half))])
        }
        
        return results

def analyze_all_dwarfs():
    """Analyze all dwarf spheroidals"""
    print("=== Dwarf Spheroidal Analysis with RS Gravity ===\n")
    
    all_results = {}
    
    for name, props in DWARF_SPHEROIDALS.items():
        print(f"Analyzing {name}...")
        
        solver = DwarfSpheroidalSolver(name, props)
        results = solver.plot_analysis()
        all_results[name] = results
        
        print(f"  Observed σ_v: {results['observed_sigma']:.1f} km/s")
        print(f"  Predicted σ_v: {results['predicted_sigma']:.1f} km/s")
        print(f"  Ratio: {results['predicted_sigma']/results['observed_sigma']:.2f}")
        print(f"  M_dyn/M_★ at r_half: {results['enhancement_rhalf']:.1f}")
        print(f"  Effective M/L: {results['M_to_L']:.1f}\n")
    
    # Summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    names = list(all_results.keys())
    obs_sigma = [all_results[n]['observed_sigma'] for n in names]
    pred_sigma = [all_results[n]['predicted_sigma'] for n in names]
    r_half = [all_results[n]['r_half'] for n in names]
    enhancement = [all_results[n]['enhancement_rhalf'] for n in names]
    
    # 1. Predicted vs Observed
    ax = axes[0, 0]
    ax.scatter(obs_sigma, pred_sigma, s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax.annotate(name.replace('_', ' '), (obs_sigma[i], pred_sigma[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Perfect agreement line
    sigma_range = [min(obs_sigma)*0.8, max(obs_sigma)*1.2]
    ax.plot(sigma_range, sigma_range, 'r--', alpha=0.5, label='Perfect agreement')
    ax.set_xlabel('Observed σ_v (km/s)')
    ax.set_ylabel('Predicted σ_v (km/s)')
    ax.set_title('RS Gravity Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Enhancement vs size
    ax = axes[0, 1]
    ax.scatter(r_half, enhancement, s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax.annotate(name.replace('_', ' '), (r_half[i], enhancement[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Half-light radius (kpc)')
    ax.set_ylabel('M_dyn/M_★ at r_half')
    ax.set_title('RS Gravity Enhancement vs Size')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 3. Residuals
    ax = axes[1, 0]
    residuals = [(p-o)/o for p, o in zip(pred_sigma, obs_sigma)]
    ax.bar(range(len(names)), residuals, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', ' ') for n in names], rotation=45)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_ylabel('(Predicted - Observed) / Observed')
    ax.set_title('Relative Residuals')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Statistics
    ax = axes[1, 1]
    ax.text(0.1, 0.9, f"Number of galaxies: {len(names)}", transform=ax.transAxes)
    ax.text(0.1, 0.8, f"Mean σ_pred/σ_obs: {np.mean([p/o for p,o in zip(pred_sigma, obs_sigma)]):.2f}", 
            transform=ax.transAxes)
    ax.text(0.1, 0.7, f"Median σ_pred/σ_obs: {np.median([p/o for p,o in zip(pred_sigma, obs_sigma)]):.2f}", 
            transform=ax.transAxes)
    ax.text(0.1, 0.6, f"Mean M_dyn/M_★: {np.mean(enhancement):.1f}", transform=ax.transAxes)
    ax.text(0.1, 0.5, f"Range M_dyn/M_★: {min(enhancement):.1f} - {max(enhancement):.1f}", 
            transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig('dwarf_spheroidal_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open('dwarf_spheroidal_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n=== Summary ===")
    print(f"Mean σ_pred/σ_obs: {np.mean([p/o for p,o in zip(pred_sigma, obs_sigma)]):.2f}")
    print(f"RS gravity typically enhances mass by factor {np.mean(enhancement):.1f} at r_half")
    print("\nResults saved to dwarf_spheroidal_results.json")
    print("Plots saved as dwarf_spheroidal_*.png")

if __name__ == "__main__":
    analyze_all_dwarfs() 