#!/usr/bin/env python3
"""
RS Gravity v7 Comprehensive Tuning Framework
Includes ξ-mode screening in all predictions and optimizes parameters
to get the best possible fit across all galaxy types.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy import stats
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
class TunableRSGravityParams:
    """All tunable parameters for RS Gravity v7."""
    # Core parameters (derived from golden ratio but with scale factors)
    beta_scale: float = 1.492
    lambda_eff: float = 50.8e-6  # m
    coupling_scale: float = 1.326
    
    # Recognition lengths (fixed from theory)
    l1: float = 0.97 * KPC_TO_M  # m
    l2: float = 24.3 * KPC_TO_M  # m
    
    # ξ-mode screening parameters (key tuning targets)
    rho_gap: float = 1.1e-24  # kg/m^3
    screening_alpha: float = 1.0  # Exponent in screening function
    screening_amp: float = 1.0  # Overall screening amplitude
    
    # Velocity gradient parameters
    alpha_grad_0: float = 1.5e6  # m
    vel_threshold: float = 50e3  # m/s (threshold for rotation)
    
    # System-specific modifiers
    dwarf_anisotropy: float = 1.3  # Anisotropy boost for dwarfs
    udg_modifier: float = 0.8  # Modifier for ultra-diffuse galaxies
    
    def to_array(self) -> np.ndarray:
        """Convert to array for optimization."""
        return np.array([
            self.beta_scale,
            self.lambda_eff * 1e6,  # Convert to microns
            self.coupling_scale,
            np.log10(self.rho_gap),  # Log space
            self.screening_alpha,
            self.screening_amp,
            self.alpha_grad_0 / 1e6,  # Convert to Mm
            self.vel_threshold / 1e3,  # Convert to km/s
            self.dwarf_anisotropy,
            self.udg_modifier
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'TunableRSGravityParams':
        """Create from array during optimization."""
        return cls(
            beta_scale=arr[0],
            lambda_eff=arr[1] * 1e-6,
            coupling_scale=arr[2],
            rho_gap=10**arr[3],
            screening_alpha=arr[4],
            screening_amp=arr[5],
            alpha_grad_0=arr[6] * 1e6,
            vel_threshold=arr[7] * 1e3,
            dwarf_anisotropy=arr[8],
            udg_modifier=arr[9]
        )

class ComprehensiveTuner:
    """Comprehensive tuning framework for RS Gravity v7."""
    
    def __init__(self):
        self.params = TunableRSGravityParams()
        self.load_all_data()
        
    def load_all_data(self):
        """Load data for all galaxy types."""
        # SPARC disk galaxies
        self.sparc_data = self.load_sparc_sample()
        
        # Dwarf spheroidals
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
        
        # Ultra-diffuse galaxies (limited data)
        self.udg_data = pd.DataFrame({
            'name': ['NGC1052-DF2', 'NGC1052-DF4', 'VCC1287'],
            'M_solar': [2e8, 1.5e8, 4.5e8],
            'r_eff_kpc': [2.2, 1.9, 3.1],
            'sigma_obs': [8.5, 4.2, 19],
            'sigma_err': [2.3, 1.1, 2],
            'rho_mean': [3e-25, 2e-25, 5e-25]
        })
        
    def load_sparc_sample(self) -> List[Dict]:
        """Load a representative SPARC sample."""
        # For tuning, use a subset of well-measured galaxies
        sample_galaxies = [
            'NGC3198', 'NGC2403', 'NGC6503', 'DDO154', 'NGC7814',
            'NGC2841', 'NGC3521', 'UGC2885', 'NGC5055', 'IC2574'
        ]
        
        sparc_sample = []
        # This would load actual SPARC data
        # For now, create representative examples
        for i, name in enumerate(sample_galaxies):
            sparc_sample.append({
                'name': name,
                'quality': 1 if i < 5 else 2,
                'r_kpc': np.linspace(0.5, 20, 20),
                'v_obs': 150 + 50 * np.random.rand(20),  # Placeholder
                'v_err': 5 + 5 * np.random.rand(20),
                'M_star': 10**(10 + 0.5 * np.random.rand()) * M_SUN,
                'rho_mean': 10**(-23 + 0.5 * np.random.rand())
            })
        
        return sparc_sample
    
    def xi_mode_screening(self, rho: np.ndarray, params: TunableRSGravityParams) -> np.ndarray:
        """Calculate screening with tunable parameters."""
        rho = np.atleast_1d(rho)
        ratio = params.rho_gap / np.maximum(rho, 1e-30)
        screening = params.screening_amp / (1.0 + ratio**params.screening_alpha)
        return screening
    
    def recognition_kernel(self, r: np.ndarray, params: TunableRSGravityParams) -> np.ndarray:
        """Standard recognition kernel."""
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
    
    def effective_gravity(self, r: float, rho: float, v_rot: float,
                         params: TunableRSGravityParams) -> float:
        """Calculate effective gravity with all effects."""
        # Base beta
        beta = -(PHI - 1) / PHI**5 * params.beta_scale
        
        # Power law
        power_law = (params.lambda_eff / r)**beta
        
        # Recognition kernel
        kernel = self.recognition_kernel(np.array([r]), params)[0]
        
        # Screening
        screening = self.xi_mode_screening(np.array([rho]), params)[0]
        
        # Velocity gradient enhancement
        if v_rot > params.vel_threshold:
            vel_enhance = 1.0 + params.alpha_grad_0 * 1e-4 / C_LIGHT
        else:
            vel_enhance = 1.0
        
        # Total
        G_eff = G_NEWTON * power_law * kernel * screening * vel_enhance
        G_eff *= params.coupling_scale
        
        return G_eff
    
    def predict_dwarf_dispersion(self, M: float, r_half: float, rho: float,
                                params: TunableRSGravityParams) -> float:
        """Predict dwarf spheroidal velocity dispersion."""
        G_eff = self.effective_gravity(r_half, rho, 0, params)
        
        # Include anisotropy
        sigma_squared = params.dwarf_anisotropy * G_eff * M / (3 * r_half)
        
        return np.sqrt(sigma_squared)
    
    def predict_udg_dispersion(self, M: float, r_eff: float, rho: float,
                              params: TunableRSGravityParams) -> float:
        """Predict UDG velocity dispersion."""
        G_eff = self.effective_gravity(r_eff, rho, 10e3, params)
        
        # UDGs may have different structure
        sigma_squared = params.udg_modifier * G_eff * M / (2 * r_eff)
        
        return np.sqrt(sigma_squared)
    
    def predict_disk_rotation(self, r: float, M_enc: float, rho: float,
                             params: TunableRSGravityParams) -> float:
        """Predict disk galaxy rotation velocity."""
        G_eff = self.effective_gravity(r, rho, 200e3, params)
        
        if r > 0 and M_enc > 0:
            return np.sqrt(G_eff * M_enc / r)
        return 0
    
    def calculate_chi2_all(self, param_array: np.ndarray) -> float:
        """Calculate total χ² across all galaxy types."""
        params = TunableRSGravityParams.from_array(param_array)
        
        chi2_total = 0
        n_total = 0
        
        # 1. Dwarf spheroidals
        for _, dwarf in self.dwarf_data.iterrows():
            M = dwarf['M_solar'] * M_SUN
            r_half = dwarf['r_half_pc'] * PC_TO_M
            rho = dwarf['rho_central']
            
            sigma_pred = self.predict_dwarf_dispersion(M, r_half, rho, params) / 1000  # km/s
            chi2 = ((sigma_pred - dwarf['sigma_obs']) / dwarf['sigma_err'])**2
            chi2_total += chi2
            n_total += 1
        
        # 2. Ultra-diffuse galaxies
        for _, udg in self.udg_data.iterrows():
            M = udg['M_solar'] * M_SUN
            r_eff = udg['r_eff_kpc'] * KPC_TO_M
            rho = udg['rho_mean']
            
            sigma_pred = self.predict_udg_dispersion(M, r_eff, rho, params) / 1000  # km/s
            chi2 = ((sigma_pred - udg['sigma_obs']) / udg['sigma_err'])**2
            chi2_total += chi2
            n_total += 1
        
        # 3. Disk galaxies (simplified for speed)
        # In reality, would fit full rotation curves
        for galaxy in self.sparc_data[:5]:  # Just use first 5 for tuning
            # Simple check at r = 10 kpc
            r = 10 * KPC_TO_M
            M_enc = 0.5 * galaxy['M_star']  # Rough estimate
            rho = galaxy['rho_mean']
            
            v_pred = self.predict_disk_rotation(r, M_enc, rho, params) / 1000  # km/s
            v_obs = 200  # Typical value
            chi2 = ((v_pred - v_obs) / 10)**2  # 10 km/s error
            chi2_total += chi2
            n_total += 1
        
        # Add penalty for unphysical parameters
        if params.screening_alpha < 0.1 or params.screening_alpha > 3:
            chi2_total += 1000
        if params.rho_gap < 1e-26 or params.rho_gap > 1e-22:
            chi2_total += 1000
        
        return chi2_total / n_total
    
    def optimize_parameters(self, method='differential_evolution'):
        """Optimize all parameters simultaneously."""
        print("Starting comprehensive parameter optimization...")
        print("=" * 60)
        
        # Define bounds
        bounds = [
            (0.5, 2.0),      # beta_scale
            (10, 200),       # lambda_eff (microns)
            (0.5, 2.0),      # coupling_scale
            (-26, -22),      # log10(rho_gap)
            (0.3, 2.0),      # screening_alpha
            (0.5, 1.5),      # screening_amp
            (0.1, 10),       # alpha_grad_0 (Mm)
            (10, 100),       # vel_threshold (km/s)
            (1.0, 2.0),      # dwarf_anisotropy
            (0.5, 1.5)       # udg_modifier
        ]
        
        # Initial guess
        x0 = self.params.to_array()
        
        if method == 'differential_evolution':
            result = differential_evolution(
                self.calculate_chi2_all,
                bounds,
                maxiter=100,
                popsize=15,
                disp=True,
                seed=42
            )
        else:
            result = minimize(
                self.calculate_chi2_all,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'disp': True}
            )
        
        # Extract best parameters
        best_params = TunableRSGravityParams.from_array(result.x)
        best_chi2 = result.fun
        
        print(f"\nOptimization complete!")
        print(f"Best χ²/N = {best_chi2:.3f}")
        
        return best_params, best_chi2
    
    def analyze_results(self, params: TunableRSGravityParams):
        """Analyze results with optimized parameters."""
        print("\nDetailed Results with Optimized Parameters:")
        print("=" * 60)
        
        # 1. Dwarf spheroidals
        print("\nDwarf Spheroidals:")
        print("-" * 50)
        print(f"{'Galaxy':12s} {'σ_obs':>8s} {'σ_pred':>8s} {'Error':>8s} {'S(ρ)':>8s}")
        print("-" * 50)
        
        dwarf_chi2 = 0
        for _, dwarf in self.dwarf_data.iterrows():
            M = dwarf['M_solar'] * M_SUN
            r_half = dwarf['r_half_pc'] * PC_TO_M
            rho = dwarf['rho_central']
            
            sigma_pred = self.predict_dwarf_dispersion(M, r_half, rho, params) / 1000
            error = (sigma_pred - dwarf['sigma_obs']) / dwarf['sigma_obs'] * 100
            screening = self.xi_mode_screening(np.array([rho]), params)[0]
            
            print(f"{dwarf['name']:12s} {dwarf['sigma_obs']:8.1f} "
                  f"{sigma_pred:8.1f} {error:+7.1f}% {screening:8.3f}")
            
            dwarf_chi2 += ((sigma_pred - dwarf['sigma_obs']) / dwarf['sigma_err'])**2
        
        print(f"\nDwarf spheroidal χ²/N = {dwarf_chi2/len(self.dwarf_data):.2f}")
        
        # 2. UDGs
        print("\nUltra-Diffuse Galaxies:")
        print("-" * 50)
        print(f"{'Galaxy':15s} {'σ_obs':>8s} {'σ_pred':>8s} {'Error':>8s} {'S(ρ)':>8s}")
        print("-" * 50)
        
        udg_chi2 = 0
        for _, udg in self.udg_data.iterrows():
            M = udg['M_solar'] * M_SUN
            r_eff = udg['r_eff_kpc'] * KPC_TO_M
            rho = udg['rho_mean']
            
            sigma_pred = self.predict_udg_dispersion(M, r_eff, rho, params) / 1000
            error = (sigma_pred - udg['sigma_obs']) / udg['sigma_obs'] * 100
            screening = self.xi_mode_screening(np.array([rho]), params)[0]
            
            print(f"{udg['name']:15s} {udg['sigma_obs']:8.1f} "
                  f"{sigma_pred:8.1f} {error:+7.1f}% {screening:8.3f}")
            
            udg_chi2 += ((sigma_pred - udg['sigma_obs']) / udg['sigma_err'])**2
        
        print(f"\nUDG χ²/N = {udg_chi2/len(self.udg_data):.2f}")
        
        # 3. Parameter comparison
        print("\nOptimized Parameters:")
        print("-" * 50)
        print(f"{'Parameter':20s} {'Original':>12s} {'Optimized':>12s} {'Change':>10s}")
        print("-" * 50)
        
        original = TunableRSGravityParams()
        param_names = [
            ('beta_scale', '', 1),
            ('lambda_eff', 'μm', 1e6),
            ('coupling_scale', '', 1),
            ('rho_gap', 'kg/m³', 1),
            ('screening_alpha', '', 1),
            ('screening_amp', '', 1),
            ('alpha_grad_0', 'Mm', 1e-6),
            ('vel_threshold', 'km/s', 1e-3),
            ('dwarf_anisotropy', '', 1),
            ('udg_modifier', '', 1)
        ]
        
        for name, unit, scale in param_names:
            orig_val = getattr(original, name) * scale
            new_val = getattr(params, name) * scale
            change = (new_val / orig_val - 1) * 100
            
            if name == 'rho_gap':
                print(f"{name:20s} {orig_val:12.2e} {new_val:12.2e} {change:+9.1f}%")
            else:
                print(f"{name:20s} {orig_val:12.3f} {new_val:12.3f} {change:+9.1f}%")
    
    def create_visualization(self, params: TunableRSGravityParams):
        """Create visualization of tuning results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Dwarf spheroidal fits
        ax = axes[0, 0]
        obs = self.dwarf_data['sigma_obs'].values
        pred = []
        for _, dwarf in self.dwarf_data.iterrows():
            M = dwarf['M_solar'] * M_SUN
            r_half = dwarf['r_half_pc'] * PC_TO_M
            rho = dwarf['rho_central']
            sigma_pred = self.predict_dwarf_dispersion(M, r_half, rho, params) / 1000
            pred.append(sigma_pred)
        
        pred = np.array(pred)
        ax.scatter(obs, pred, s=100, alpha=0.7)
        ax.plot([0, 20], [0, 20], 'k--', alpha=0.5)
        ax.set_xlabel('Observed σ (km/s)')
        ax.set_ylabel('Predicted σ (km/s)')
        ax.set_title('Dwarf Spheroidals')
        ax.text(0.05, 0.95, f'Mean error: {np.mean(np.abs(pred/obs - 1))*100:.1f}%',
                transform=ax.transAxes, va='top')
        
        # Plot 2: Screening function
        ax = axes[0, 1]
        rho = np.logspace(-27, -21, 1000)
        screening = self.xi_mode_screening(rho, params)
        
        ax.semilogx(rho, screening, 'b-', linewidth=2)
        ax.axvline(params.rho_gap, color='red', linestyle='--', 
                   label=f'ρ_gap = {params.rho_gap:.1e}')
        
        # Mark galaxy types
        for _, dwarf in self.dwarf_data.iterrows():
            ax.axvline(dwarf['rho_central'], color='orange', alpha=0.3)
        
        ax.set_xlabel('Density (kg/m³)')
        ax.set_ylabel('Screening Factor S(ρ)')
        ax.set_title(f'Optimized Screening (α = {params.screening_alpha:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Parameter changes
        ax = axes[1, 0]
        original = TunableRSGravityParams()
        param_labels = ['β_scale', 'λ_eff', 'coupling', 'ρ_gap', 'α', 'S_amp']
        orig_vals = [original.beta_scale, original.lambda_eff*1e6, 
                     original.coupling_scale, np.log10(original.rho_gap),
                     original.screening_alpha, original.screening_amp]
        new_vals = [params.beta_scale, params.lambda_eff*1e6,
                    params.coupling_scale, np.log10(params.rho_gap),
                    params.screening_alpha, params.screening_amp]
        
        changes = [(n/o - 1) * 100 for n, o in zip(new_vals, orig_vals)]
        
        y_pos = np.arange(len(param_labels))
        ax.barh(y_pos, changes, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_labels)
        ax.set_xlabel('Change (%)')
        ax.set_title('Parameter Optimization')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Chi-squared by galaxy type
        ax = axes[1, 1]
        
        # Calculate chi2 for each type
        dwarf_chi2s = []
        for _, dwarf in self.dwarf_data.iterrows():
            M = dwarf['M_solar'] * M_SUN
            r_half = dwarf['r_half_pc'] * PC_TO_M
            rho = dwarf['rho_central']
            sigma_pred = self.predict_dwarf_dispersion(M, r_half, rho, params) / 1000
            chi2 = ((sigma_pred - dwarf['sigma_obs']) / dwarf['sigma_err'])**2
            dwarf_chi2s.append(chi2)
        
        udg_chi2s = []
        for _, udg in self.udg_data.iterrows():
            M = udg['M_solar'] * M_SUN
            r_eff = udg['r_eff_kpc'] * KPC_TO_M
            rho = udg['rho_mean']
            sigma_pred = self.predict_udg_dispersion(M, r_eff, rho, params) / 1000
            chi2 = ((sigma_pred - udg['sigma_obs']) / udg['sigma_err'])**2
            udg_chi2s.append(chi2)
        
        types = ['Dwarfs', 'UDGs']
        chi2_means = [np.mean(dwarf_chi2s), np.mean(udg_chi2s)]
        chi2_stds = [np.std(dwarf_chi2s), np.std(udg_chi2s)]
        
        x_pos = np.arange(len(types))
        ax.bar(x_pos, chi2_means, yerr=chi2_stds, alpha=0.7, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(types)
        ax.set_ylabel('Mean χ²')
        ax.set_title('Fit Quality by Galaxy Type')
        ax.axhline(1, color='red', linestyle='--', alpha=0.5, label='χ² = 1')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('rs_gravity_v7_tuning_results.png', dpi=150)
        plt.close()
        
        return fig
    
    def save_optimized_parameters(self, params: TunableRSGravityParams, chi2: float):
        """Save optimized parameters to file."""
        results = {
            'description': 'RS Gravity v7 Optimized Parameters with ξ-mode Screening',
            'optimization_date': pd.Timestamp.now().isoformat(),
            'total_chi2_per_n': chi2,
            'parameters': asdict(params),
            'parameter_changes': {},
            'fit_statistics': {
                'n_dwarfs': len(self.dwarf_data),
                'n_udgs': len(self.udg_data),
                'n_disks': len(self.sparc_data)
            }
        }
        
        # Calculate parameter changes
        original = TunableRSGravityParams()
        for key in asdict(params).keys():
            orig_val = getattr(original, key)
            new_val = getattr(params, key)
            if isinstance(orig_val, (int, float)):
                results['parameter_changes'][key] = {
                    'original': orig_val,
                    'optimized': new_val,
                    'change_percent': (new_val / orig_val - 1) * 100
                }
        
        with open('rs_gravity_v7_optimized_params.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nOptimized parameters saved to rs_gravity_v7_optimized_params.json")

def main():
    """Run comprehensive tuning."""
    tuner = ComprehensiveTuner()
    
    # Optimize parameters
    best_params, best_chi2 = tuner.optimize_parameters(method='differential_evolution')
    
    # Analyze results
    tuner.analyze_results(best_params)
    
    # Create visualization
    tuner.create_visualization(best_params)
    print("\nVisualization saved to rs_gravity_v7_tuning_results.png")
    
    # Save results
    tuner.save_optimized_parameters(best_params, best_chi2)
    
    print("\n" + "=" * 60)
    print("TUNING COMPLETE!")
    print(f"Final χ²/N = {best_chi2:.3f}")
    print("Key findings:")
    print(f"- ρ_gap optimized to {best_params.rho_gap:.2e} kg/m³")
    print(f"- Screening exponent α = {best_params.screening_alpha:.2f}")
    print(f"- Dwarf anisotropy factor = {best_params.dwarf_anisotropy:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main() 