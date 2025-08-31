#!/usr/bin/env python3
"""
RS Gravity v7 SPARC-based Comprehensive Tuning
Uses actual SPARC data and includes ξ-mode screening for all predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Physical constants
G_NEWTON = 6.67430e-11  # m^3 kg^-1 s^-2
C_LIGHT = 299792458.0   # m/s
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
M_SUN = 1.989e30  # kg
PC_TO_M = 3.086e16  # m
KPC_TO_M = 3.086e19  # m

@dataclass
class OptimizedRSGravityParams:
    """Optimized parameters for RS Gravity v7 with screening."""
    # Core parameters (tunable)
    beta_scale: float = 0.5  # From initial tuning
    lambda_eff: float = 118.6e-6  # m (from initial tuning)
    coupling_scale: float = 1.97
    
    # Recognition lengths (fixed from theory)
    l1: float = 0.97 * KPC_TO_M  # m
    l2: float = 24.3 * KPC_TO_M  # m
    
    # ξ-mode screening parameters
    rho_gap: float = 1.46e-24  # kg/m³ (from initial tuning)
    screening_alpha: float = 1.91
    screening_amp: float = 1.145
    
    # Velocity gradient parameters
    alpha_grad_0: float = 2.24e6  # m
    vel_threshold: float = 37.8e3  # m/s
    
    # System-specific modifiers
    dwarf_anisotropy: float = 1.0
    udg_modifier: float = 0.646
    
    # Disk-specific parameters
    disk_boost: float = 1.2  # Additional boost for disk galaxies
    bulge_suppression: float = 0.8  # Suppression in bulge-dominated regions

class SPARCGalaxyData:
    """Container for SPARC galaxy data."""
    def __init__(self, name: str):
        self.name = name
        self.data = None
        self.quality = 1
        self.distance = 10.0  # Mpc
        self.inc = 60.0  # degrees
        self.M_star = 1e10 * M_SUN
        self.M_gas = 1e9 * M_SUN
        self.scale_length = 3.0 * KPC_TO_M
        
    def load_rotmod(self, directory='Rotmod_LTG'):
        """Load rotation curve data."""
        filename = f"{directory}/{self.name}_rotmod.dat"
        if os.path.exists(filename):
            # Format: r(kpc) v_obs(km/s) v_err(km/s) v_gas v_disk v_bul
            data = pd.read_csv(filename, sep=r'\s+', comment='#',
                             names=['r_kpc', 'v_obs', 'v_err', 'v_gas', 'v_disk', 'v_bul'])
            self.data = data
            return True
        return False
    
    def get_density_profile(self, r: np.ndarray) -> np.ndarray:
        """Estimate density profile from baryonic components."""
        # Simple exponential disk model
        Sigma_0 = self.M_star / (2 * np.pi * self.scale_length**2)
        Sigma = Sigma_0 * np.exp(-r / self.scale_length)
        
        # Convert to volume density (thin disk approximation)
        h_z = 0.1 * self.scale_length  # Scale height
        rho = Sigma / (2 * h_z)
        
        # Add gas contribution
        rho_gas = 0.1 * rho  # Simplified
        
        return rho + rho_gas

class ComprehensiveSPARCTuner:
    """Tuner using actual SPARC data."""
    
    def __init__(self):
        self.params = OptimizedRSGravityParams()
        self.galaxies = []
        self.dwarf_data = None
        self.load_data()
        
    def load_data(self):
        """Load SPARC galaxies and dwarf spheroidals."""
        # Load SPARC sample
        sparc_names = [
            'NGC3198', 'NGC2403', 'NGC6503', 'DDO154', 'NGC7814',
            'NGC2841', 'NGC3521', 'UGC2885', 'NGC5055', 'IC2574',
            'NGC0300', 'NGC2976', 'NGC4736', 'NGC5457', 'NGC7331'
        ]
        
        for name in sparc_names:
            galaxy = SPARCGalaxyData(name)
            if galaxy.load_rotmod():
                self.galaxies.append(galaxy)
                print(f"Loaded {name}")
            else:
                print(f"Warning: Could not load {name}")
        
        # Load dwarf spheroidals
        self.dwarf_data = pd.DataFrame({
            'name': ['Draco', 'Fornax', 'Sculptor', 'Leo I', 'Leo II', 
                     'Carina', 'Sextans', 'Ursa Minor'],
            'M_solar': [3e7, 4e8, 2e7, 5e7, 1e7, 2e7, 1e7, 3e7],
            'r_half_pc': [200, 700, 280, 250, 180, 250, 300, 300],
            'sigma_obs': [9.1, 11.7, 9.2, 9.2, 6.6, 6.6, 7.9, 9.5],
            'sigma_err': [1.2, 0.9, 1.4, 1.4, 0.7, 1.2, 1.3, 1.2],
            'rho_central': [2.7e-25, 1.5e-25, 3.5e-25, 2.0e-25, 
                           1.5e-25, 1.8e-25, 1.0e-25, 2.5e-25]
        })
    
    def xi_mode_screening(self, rho: np.ndarray, params: OptimizedRSGravityParams) -> np.ndarray:
        """Calculate ξ-mode screening factor."""
        rho = np.atleast_1d(rho)
        ratio = params.rho_gap / np.maximum(rho, 1e-30)
        screening = params.screening_amp / (1.0 + ratio**params.screening_alpha)
        return screening
    
    def recognition_kernel(self, r: np.ndarray, params: OptimizedRSGravityParams) -> np.ndarray:
        """Recognition kernel K(r)."""
        def xi_func(x):
            x = np.atleast_1d(x)
            result = np.zeros_like(x)
            
            small = x < 0.1
            if np.any(small):
                result[small] = 0.6 - 0.0357 * x[small]**2
            
            large = x > 50
            if np.any(large):
                result[large] = 3 * np.cos(x[large]) / x[large]**3
            
            standard = ~(small | large)
            if np.any(standard):
                x_std = x[standard]
                result[standard] = 3 * (np.sin(x_std) - x_std * np.cos(x_std)) / x_std**3
            
            return result
        
        r = np.atleast_1d(r)
        return xi_func(r / params.l1) + xi_func(r / params.l2)
    
    def effective_gravity(self, r: np.ndarray, rho: np.ndarray, v_rot: float,
                         params: OptimizedRSGravityParams, is_disk: bool = True) -> np.ndarray:
        """Calculate effective gravity with all effects."""
        # Base beta
        beta = -(PHI - 1) / PHI**5 * params.beta_scale
        
        # Power law
        power_law = (params.lambda_eff / r)**beta
        
        # Recognition kernel
        kernel = self.recognition_kernel(r, params)
        
        # Screening
        screening = self.xi_mode_screening(rho, params)
        
        # Velocity gradient enhancement
        if v_rot > params.vel_threshold:
            vel_enhance = 1.0 + params.alpha_grad_0 * v_rot / (r * C_LIGHT)
        else:
            vel_enhance = 1.0
        
        # System-specific modifiers
        if is_disk:
            system_mod = params.disk_boost
        else:
            system_mod = 1.0
        
        # Total
        G_eff = G_NEWTON * power_law * kernel * screening * vel_enhance * system_mod
        G_eff *= params.coupling_scale
        
        return G_eff
    
    def solve_rotation_curve(self, galaxy: SPARCGalaxyData, 
                           params: OptimizedRSGravityParams) -> np.ndarray:
        """Solve for rotation curve using RS gravity."""
        if galaxy.data is None:
            return np.zeros(1)
        
        r = galaxy.data['r_kpc'].values * KPC_TO_M
        
        # Get density profile
        rho = galaxy.get_density_profile(r)
        
        # Estimate typical rotation velocity
        v_typ = np.median(galaxy.data['v_obs'].values) * 1000  # m/s
        
        # Calculate G_eff at each radius
        G_eff = self.effective_gravity(r, rho, v_typ, params, is_disk=True)
        
        # Calculate enclosed mass (simplified)
        M_enc = np.zeros_like(r)
        for i in range(len(r)):
            if i == 0:
                M_enc[i] = 4 * np.pi * rho[i] * r[i]**3 / 3
            else:
                # Trapezoidal integration
                dr = r[i] - r[i-1]
                M_enc[i] = M_enc[i-1] + 2 * np.pi * r[i]**2 * rho[i] * dr
        
        # Calculate rotation velocity
        v_pred = np.sqrt(G_eff * M_enc / r)
        
        return v_pred / 1000  # km/s
    
    def calculate_chi2_galaxy(self, galaxy: SPARCGalaxyData, 
                            params: OptimizedRSGravityParams) -> float:
        """Calculate χ² for a single galaxy."""
        v_pred = self.solve_rotation_curve(galaxy, params)
        v_obs = galaxy.data['v_obs'].values
        v_err = galaxy.data['v_err'].values
        
        # Mask bad data
        mask = (v_err > 0) & (v_obs > 0)
        
        chi2 = np.sum(((v_pred[mask] - v_obs[mask]) / v_err[mask])**2)
        return chi2 / np.sum(mask)
    
    def calculate_total_chi2(self, param_array: np.ndarray) -> float:
        """Calculate total χ² for optimization."""
        # Unpack parameters
        params = OptimizedRSGravityParams(
            beta_scale=param_array[0],
            lambda_eff=param_array[1] * 1e-6,
            coupling_scale=param_array[2],
            rho_gap=10**param_array[3],
            screening_alpha=param_array[4],
            screening_amp=param_array[5],
            disk_boost=param_array[6],
            dwarf_anisotropy=param_array[7]
        )
        
        chi2_total = 0
        n_total = 0
        
        # 1. SPARC galaxies
        for galaxy in self.galaxies:
            chi2 = self.calculate_chi2_galaxy(galaxy, params)
            chi2_total += chi2
            n_total += 1
        
        # 2. Dwarf spheroidals
        for _, dwarf in self.dwarf_data.iterrows():
            M = dwarf['M_solar'] * M_SUN
            r_half = dwarf['r_half_pc'] * PC_TO_M
            rho = dwarf['rho_central']
            
            G_eff = self.effective_gravity(
                np.array([r_half]), np.array([rho]), 0, params, is_disk=False
            )[0]
            
            sigma_pred = np.sqrt(params.dwarf_anisotropy * G_eff * M / (3 * r_half))
            sigma_pred_kms = sigma_pred / 1000
            
            chi2 = ((sigma_pred_kms - dwarf['sigma_obs']) / dwarf['sigma_err'])**2
            chi2_total += chi2
            n_total += 1
        
        # Add penalties
        if params.screening_alpha < 0.5 or params.screening_alpha > 3:
            chi2_total += 100
        if params.beta_scale < 0.1 or params.beta_scale > 2:
            chi2_total += 100
        
        return chi2_total / n_total
    
    def optimize_parameters(self):
        """Optimize all parameters."""
        print("\nStarting SPARC-based parameter optimization...")
        print("=" * 60)
        print(f"Using {len(self.galaxies)} SPARC galaxies and {len(self.dwarf_data)} dwarf spheroidals")
        
        # Define bounds
        bounds = [
            (0.3, 1.5),      # beta_scale
            (50, 200),       # lambda_eff (microns)
            (1.0, 3.0),      # coupling_scale
            (-25, -23),      # log10(rho_gap)
            (1.0, 3.0),      # screening_alpha
            (0.8, 1.5),      # screening_amp
            (0.8, 1.5),      # disk_boost
            (0.8, 1.5)       # dwarf_anisotropy
        ]
        
        # Initial guess from previous tuning
        x0 = np.array([0.5, 118.6, 1.97, np.log10(1.46e-24), 1.91, 1.145, 1.2, 1.0])
        
        # Optimize
        result = differential_evolution(
            self.calculate_total_chi2,
            bounds,
            x0=x0,
            maxiter=50,
            popsize=10,
            disp=True,
            seed=42,
            workers=-1  # Use all cores
        )
        
        # Extract best parameters
        best_params = OptimizedRSGravityParams(
            beta_scale=result.x[0],
            lambda_eff=result.x[1] * 1e-6,
            coupling_scale=result.x[2],
            rho_gap=10**result.x[3],
            screening_alpha=result.x[4],
            screening_amp=result.x[5],
            disk_boost=result.x[6],
            dwarf_anisotropy=result.x[7]
        )
        
        print(f"\nOptimization complete!")
        print(f"Best χ²/N = {result.fun:.3f}")
        
        return best_params, result.fun
    
    def analyze_results(self, params: OptimizedRSGravityParams):
        """Detailed analysis of results."""
        print("\nDetailed Analysis:")
        print("=" * 60)
        
        # 1. Best and worst SPARC fits
        print("\nSPARC Galaxies:")
        print("-" * 50)
        print(f"{'Galaxy':12s} {'χ²/N':>8s} {'Quality':>8s}")
        print("-" * 50)
        
        chi2_list = []
        for galaxy in self.galaxies:
            chi2 = self.calculate_chi2_galaxy(galaxy, params)
            chi2_list.append((galaxy.name, chi2))
            print(f"{galaxy.name:12s} {chi2:8.2f} {galaxy.quality:8d}")
        
        chi2_list.sort(key=lambda x: x[1])
        print(f"\nBest fit: {chi2_list[0][0]} (χ²/N = {chi2_list[0][1]:.2f})")
        print(f"Worst fit: {chi2_list[-1][0]} (χ²/N = {chi2_list[-1][1]:.2f})")
        
        # 2. Dwarf spheroidals
        print("\nDwarf Spheroidals:")
        print("-" * 60)
        print(f"{'Galaxy':12s} {'σ_obs':>8s} {'σ_pred':>8s} {'Error':>8s} {'χ²':>8s}")
        print("-" * 60)
        
        for _, dwarf in self.dwarf_data.iterrows():
            M = dwarf['M_solar'] * M_SUN
            r_half = dwarf['r_half_pc'] * PC_TO_M
            rho = dwarf['rho_central']
            
            G_eff = self.effective_gravity(
                np.array([r_half]), np.array([rho]), 0, params, is_disk=False
            )[0]
            
            sigma_pred = np.sqrt(params.dwarf_anisotropy * G_eff * M / (3 * r_half)) / 1000
            chi2 = ((sigma_pred - dwarf['sigma_obs']) / dwarf['sigma_err'])**2
            error = (sigma_pred / dwarf['sigma_obs'] - 1) * 100
            
            print(f"{dwarf['name']:12s} {dwarf['sigma_obs']:8.1f} "
                  f"{sigma_pred:8.1f} {error:+7.1f}% {chi2:8.2f}")
    
    def create_final_plots(self, params: OptimizedRSGravityParams):
        """Create final visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Example rotation curves
        ax = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, 4))
        
        for i, galaxy in enumerate(self.galaxies[:4]):
            if galaxy.data is not None:
                r = galaxy.data['r_kpc'].values
                v_obs = galaxy.data['v_obs'].values
                v_err = galaxy.data['v_err'].values
                v_pred = self.solve_rotation_curve(galaxy, params)
                
                ax.errorbar(r, v_obs, yerr=v_err, fmt='o', alpha=0.5, 
                           color=colors[i], markersize=3)
                ax.plot(r, v_pred, '-', color=colors[i], linewidth=2,
                       label=f"{galaxy.name} (χ²/N={self.calculate_chi2_galaxy(galaxy, params):.1f})")
        
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Rotation Velocity (km/s)')
        ax.set_title('Example SPARC Fits')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Screening function with galaxy positions
        ax = axes[0, 1]
        rho = np.logspace(-27, -21, 1000)
        screening = self.xi_mode_screening(rho, params)
        
        ax.semilogx(rho, screening, 'b-', linewidth=2)
        ax.axvline(params.rho_gap, color='red', linestyle='--', 
                   label=f'ρ_gap = {params.rho_gap:.2e}')
        
        # Mark different galaxy types
        for _, dwarf in self.dwarf_data.iterrows():
            ax.axvline(dwarf['rho_central'], color='orange', alpha=0.3)
        
        ax.text(1e-25, 0.8, 'Dwarfs', color='orange', fontsize=10)
        ax.text(1e-23, 0.8, 'Disks', color='green', fontsize=10)
        
        ax.set_xlabel('Density (kg/m³)')
        ax.set_ylabel('Screening Factor S(ρ)')
        ax.set_title(f'ξ-mode Screening (α = {params.screening_alpha:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.2)
        
        # Plot 3: Chi-squared distribution
        ax = axes[1, 0]
        
        sparc_chi2s = [self.calculate_chi2_galaxy(g, params) for g in self.galaxies]
        
        ax.hist(sparc_chi2s, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.median(sparc_chi2s), color='red', linestyle='--',
                  label=f'Median = {np.median(sparc_chi2s):.1f}')
        ax.set_xlabel('χ²/N')
        ax.set_ylabel('Number of Galaxies')
        ax.set_title('SPARC Fit Quality Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Parameter summary
        ax = axes[1, 1]
        ax.axis('off')
        
        param_text = f"""Optimized RS Gravity v7 Parameters:
        
β_scale = {params.beta_scale:.3f}
λ_eff = {params.lambda_eff*1e6:.1f} μm
coupling = {params.coupling_scale:.3f}

ρ_gap = {params.rho_gap:.2e} kg/m³
α = {params.screening_alpha:.2f}
S_amp = {params.screening_amp:.3f}

disk_boost = {params.disk_boost:.3f}
dwarf_aniso = {params.dwarf_anisotropy:.3f}

Total galaxies: {len(self.galaxies) + len(self.dwarf_data)}
Median SPARC χ²/N: {np.median(sparc_chi2s):.1f}
"""
        
        ax.text(0.1, 0.9, param_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('rs_gravity_v7_sparc_tuning_results.png', dpi=150)
        plt.close()
    
    def save_results(self, params: OptimizedRSGravityParams, chi2: float):
        """Save optimized parameters and results."""
        results = {
            'description': 'RS Gravity v7 SPARC-optimized Parameters',
            'n_sparc_galaxies': len(self.galaxies),
            'n_dwarf_spheroidals': len(self.dwarf_data),
            'total_chi2_per_n': chi2,
            'parameters': asdict(params),
            'sparc_results': {}
        }
        
        # Add individual galaxy results
        for galaxy in self.galaxies:
            results['sparc_results'][galaxy.name] = {
                'chi2_per_n': self.calculate_chi2_galaxy(galaxy, params),
                'n_points': len(galaxy.data) if galaxy.data is not None else 0
            }
        
        with open('rs_gravity_v7_sparc_optimized.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to rs_gravity_v7_sparc_optimized.json")

def main():
    """Run SPARC-based tuning."""
    tuner = ComprehensiveSPARCTuner()
    
    # Optimize
    best_params, best_chi2 = tuner.optimize_parameters()
    
    # Analyze
    tuner.analyze_results(best_params)
    
    # Visualize
    tuner.create_final_plots(best_params)
    print("\nVisualization saved to rs_gravity_v7_sparc_tuning_results.png")
    
    # Save
    tuner.save_results(best_params, best_chi2)
    
    print("\n" + "=" * 60)
    print("SPARC TUNING COMPLETE!")
    print(f"Final χ²/N = {best_chi2:.3f}")
    print("=" * 60)

if __name__ == "__main__":
    main() 