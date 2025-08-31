#!/usr/bin/env python3
"""
RS Gravity v9: Pressure-Based Dynamics
Implements gravity from P = J_in - J_out as shown in the manuscript
This is the correct formulation - gravity emerges from ledger flow imbalance
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import json

# Universal constants
PHI = (1 + np.sqrt(5)) / 2
C_LIGHT = 299792458.0  # m/s
HBAR = 1.054571817e-34  # J·s
G_NEWTON = 6.67430e-11  # m^3 kg^-1 s^-2
K_B = 1.380649e-23  # J/K

# Recognition Science constants
E_COH = 0.090 * 1.602176634e-19  # J
TAU_0 = 7.33e-15  # s
LAMBDA_REC = 7.23e-36  # m
PC_TO_M = 3.086e16
KPC_TO_M = 3.086e19
M_SUN = 1.989e30  # kg

@dataclass
class PressureGravityParams:
    """Parameters for pressure-based gravity."""
    # Recognition scales
    lambda_eff: float = 60e-6  # m (effective wavelength)
    l1: float = 0.97 * KPC_TO_M  # m (first kernel zero)
    l2: float = 24.3 * KPC_TO_M  # m (second kernel zero)
    
    # Pressure dynamics
    pressure_coupling: float = 1.0  # Pressure to gravity coupling
    lock_in_rate: float = 1e6  # Lock-in events per second per unit pressure
    eight_beat_amplitude: float = 0.01  # 8-beat modulation strength
    
    # Information effects
    bits_per_complexity: float = 1e23  # bits per unit complexity
    info_temperature: float = 300  # K (information temperature)
    
    # Light contribution
    light_pressure_scale: float = 0.1  # Light pressure contribution
    opcode_density_scale: float = 0.01  # LNAL opcode contribution

class PressureDynamicsGravity:
    """
    Gravity from ledger pressure P = J_in - J_out.
    This is the correct formulation from the manuscript.
    """
    
    def __init__(self, params: Optional[PressureGravityParams] = None):
        self.params = params or PressureGravityParams()
        
    def recognition_flux_in(self, r: np.ndarray, v: np.ndarray, 
                           rho: np.ndarray) -> np.ndarray:
        """
        J_in: Recognition events flowing into a region.
        Depends on velocity convergence and density.
        """
        # Velocity convergence creates inflow
        # For single values, approximate div_v from velocity scale
        if np.isscalar(v) or (hasattr(v, 'shape') and v.shape == ()):
            v = np.atleast_1d(v)
        if np.isscalar(r) or (hasattr(r, 'shape') and r.shape == ()):
            r = np.atleast_1d(r)
        if np.isscalar(rho) or (hasattr(rho, 'shape') and rho.shape == ()):
            rho = np.atleast_1d(rho)
            
        if len(v) == 1:
            # Approximate divergence for single value
            div_v = -v[0] / r[0]  # Spherical approximation
        else:
            div_v = np.gradient(v)
            
        convergence = np.maximum(-div_v, 0)  # Only inflow
        
        # Density attracts recognition events
        attraction = rho * C_LIGHT**2 / E_COH
        
        # Scale by recognition kernel
        kernel = self.recognition_kernel(r)
        
        return attraction * (1 + convergence/C_LIGHT) * kernel
    
    def recognition_flux_out(self, r: np.ndarray, v: np.ndarray, 
                            rho: np.ndarray) -> np.ndarray:
        """
        J_out: Recognition events flowing out of a region.
        Depends on velocity divergence and entropy production.
        """
        # Velocity divergence creates outflow
        # For single values, approximate div_v from velocity scale
        if np.isscalar(v) or (hasattr(v, 'shape') and v.shape == ()):
            v = np.atleast_1d(v)
        if np.isscalar(r) or (hasattr(r, 'shape') and r.shape == ()):
            r = np.atleast_1d(r)
        if np.isscalar(rho) or (hasattr(rho, 'shape') and rho.shape == ()):
            rho = np.atleast_1d(rho)
            
        if len(v) == 1:
            # Approximate divergence for single value
            div_v = v[0] / r[0]  # Spherical approximation
        else:
            div_v = np.gradient(v)
            
        divergence = np.maximum(div_v, 0)  # Only outflow
        
        # Entropy production drives outflow
        temperature = self.local_temperature(rho)
        entropy_rate = K_B * temperature / TAU_0
        
        # Information complexity reduces outflow (organized systems)
        complexity = self.complexity_factor(r, rho)
        
        return entropy_rate * (1 + divergence/C_LIGHT) / complexity
    
    def pressure_field(self, r: np.ndarray, v: np.ndarray, 
                      rho: np.ndarray) -> np.ndarray:
        """
        P = J_in - J_out: The fundamental driver of gravity.
        """
        J_in = self.recognition_flux_in(r, v, rho)
        J_out = self.recognition_flux_out(r, v, rho)
        return J_in - J_out
    
    def lock_in_events(self, P: np.ndarray, dt: float) -> np.ndarray:
        """
        Discrete lock-in events from pressure.
        Each event releases E_lock = 0.09 eV.
        """
        # Poisson process with rate proportional to pressure
        rate = self.params.lock_in_rate * np.abs(P) * dt
        
        # Limit rate to avoid numerical issues
        rate = np.minimum(rate, 1e6)  # Cap at 1 million events
        
        # For very small rates, use probability
        if np.any(rate < 0.01):
            # Bernoulli approximation for small rates
            n_events = (np.random.random(rate.shape) < rate).astype(float)
        else:
            n_events = np.random.poisson(rate)
        
        # Each event creates mass
        mass_created = n_events * E_COH / C_LIGHT**2
        return mass_created
    
    def eight_beat_modulation(self, t: float) -> float:
        """
        8-beat cosmic cycle modulates all physics.
        """
        phase = (t / TAU_0) % 8
        return 1 + self.params.eight_beat_amplitude * np.cos(2*np.pi*phase/8)
    
    def recognition_kernel(self, r: np.ndarray) -> np.ndarray:
        """Standard RS recognition kernel."""
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
        return xi_func(r / self.params.l1) + xi_func(r / self.params.l2)
    
    def local_temperature(self, rho: np.ndarray) -> np.ndarray:
        """Local temperature from density."""
        # Simple ideal gas approximation
        return (rho * C_LIGHT**2 / K_B)**(1/4)
    
    def complexity_factor(self, r: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """
        Complexity of local structure.
        Galaxies are more complex than random gas.
        """
        # Ensure arrays
        r = np.atleast_1d(r)
        rho = np.atleast_1d(rho)
        
        # Use density gradient as proxy for structure
        if len(rho) > 1:
            grad_rho = np.gradient(rho)
            structure = 1 + np.abs(grad_rho) / (rho + 1e-30)
        else:
            # For single values, use scale-based complexity
            # Galactic scales are more complex
            r_gal = 10 * KPC_TO_M
            structure = 1 + 0.5 * np.exp(-((np.log10(r) - np.log10(r_gal))/2)**2)
            
        return structure
    
    def information_mass(self, r: np.ndarray, complexity: np.ndarray) -> np.ndarray:
        """
        Information has mass: m = k_B T ln(2) / c².
        """
        bits = self.params.bits_per_complexity * complexity
        T = self.params.info_temperature
        return bits * K_B * T * np.log(2) / C_LIGHT**2
    
    def light_pressure_contribution(self, r: np.ndarray, t: float) -> np.ndarray:
        """
        Light creates pressure through momentum transfer.
        Includes LNAL opcode effects.
        """
        # Standing wave pattern (simplified)
        k = 2 * np.pi / self.params.lambda_eff
        standing_wave = np.sin(k * r) * np.cos(C_LIGHT * k * t)
        
        # Light pressure
        u_light = (standing_wave**2) * E_COH / self.params.lambda_eff**3
        p_light = u_light / 3  # Radiation pressure
        
        # LNAL opcode contribution
        opcode_factor = self.params.opcode_density_scale * (1 + standing_wave**2)
        
        return self.params.light_pressure_scale * (p_light + opcode_factor * u_light)
    
    def effective_gravity_from_pressure(self, r: float, v: float, rho: float,
                                      t: float = 0) -> Dict:
        """
        Calculate gravity from pressure dynamics.
        Returns detailed breakdown of all contributions.
        """
        # Core pressure field
        P = self.pressure_field(np.array([r]), np.array([v]), np.array([rho]))[0]
        
        # Pressure gradient drives acceleration
        # For spherical symmetry: ∇P ≈ dP/dr
        dr = 0.01 * r  # Small step for gradient
        P_plus = self.pressure_field(np.array([r + dr]), np.array([v]), np.array([rho]))[0]
        dP_dr = (P_plus - P) / dr
        
        # Base gravitational acceleration from pressure
        g_pressure = -dP_dr / (rho + 1e-30) * self.params.pressure_coupling
        
        # Lock-in events add discrete mass
        dm_lock = self.lock_in_events(np.array([P]), TAU_0)[0]
        g_lock = G_NEWTON * dm_lock / r**2
        
        # Eight-beat modulation
        modulation = self.eight_beat_modulation(t)
        
        # Information mass contribution
        complexity = self.complexity_factor(np.array([r]), np.array([rho]))[0]
        m_info = self.information_mass(np.array([r]), np.array([complexity]))[0]
        g_info = G_NEWTON * m_info / r**2
        
        # Light pressure effects
        p_light = self.light_pressure_contribution(np.array([r]), t)[0]
        g_light = p_light / (rho * r)
        
        # Total effective acceleration
        g_total = (g_pressure + g_lock + g_info + g_light) * modulation
        
        # Convert to effective G
        g_newton = G_NEWTON * rho * 4 * np.pi * r / 3  # Enclosed mass
        G_eff = g_total * r**2 / (rho * 4 * np.pi * r / 3) if g_newton > 0 else G_NEWTON
        
        return {
            'pressure': P,
            'pressure_gradient': dP_dr,
            'g_pressure': g_pressure,
            'g_lock': g_lock,
            'g_info': g_info,
            'g_light': g_light,
            'g_total': g_total,
            'G_eff': G_eff,
            'modulation': modulation,
            'J_in': self.recognition_flux_in(np.array([r]), np.array([v]), np.array([rho]))[0],
            'J_out': self.recognition_flux_out(np.array([r]), np.array([v]), np.array([rho]))[0]
        }
    
    def solve_rotation_curve(self, r_values: np.ndarray, rho_func, 
                           galaxy_name: str = "Galaxy") -> Dict:
        """
        Solve for rotation curve using pressure dynamics.
        """
        v_rot = []
        details = []
        
        for r in r_values:
            # Get density at this radius
            rho = rho_func(r)
            
            # Initial guess for velocity
            v_guess = np.sqrt(G_NEWTON * rho * 4 * np.pi * r**2 / 3)
            
            # Iterate to find self-consistent velocity
            for _ in range(10):
                result = self.effective_gravity_from_pressure(r, v_guess, rho)
                v_new = np.sqrt(abs(result['g_total'] * r))
                if abs(v_new - v_guess) < 0.01 * v_guess:
                    break
                v_guess = v_new
            
            v_rot.append(v_new)
            details.append(result)
        
        return {
            'r': r_values,
            'v': np.array(v_rot),
            'details': details,
            'galaxy': galaxy_name
        }
    
    def create_analysis_plots(self):
        """Create comprehensive analysis plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('RS Gravity v9: Pressure-Based Dynamics', fontsize=16)
        
        # Test parameters
        r_test = np.logspace(-9, 21, 100)  # 1 nm to 1 Mpc
        rho_test = 1e-24  # kg/m³
        v_test = 200e3  # m/s
        
        # 1. Pressure vs radius
        ax = axes[0, 0]
        P_values = []
        for r in r_test:
            result = self.effective_gravity_from_pressure(r, v_test, rho_test)
            P_values.append(result['pressure'])
        
        ax.loglog(r_test/PC_TO_M, np.abs(P_values))
        ax.set_xlabel('Radius (pc)')
        ax.set_ylabel('|Pressure| (recognition units)')
        ax.set_title('Pressure Field P = J_in - J_out')
        ax.grid(True, alpha=0.3)
        
        # 2. J_in vs J_out
        ax = axes[0, 1]
        J_in_values = []
        J_out_values = []
        for r in r_test:
            result = self.effective_gravity_from_pressure(r, v_test, rho_test)
            J_in_values.append(result['J_in'])
            J_out_values.append(result['J_out'])
        
        ax.loglog(r_test/PC_TO_M, J_in_values, 'b-', label='J_in')
        ax.loglog(r_test/PC_TO_M, J_out_values, 'r-', label='J_out')
        ax.set_xlabel('Radius (pc)')
        ax.set_ylabel('Recognition Flux')
        ax.set_title('Recognition Flux Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Effective G vs radius
        ax = axes[0, 2]
        G_eff_values = []
        for r in r_test:
            result = self.effective_gravity_from_pressure(r, v_test, rho_test)
            G_eff_values.append(result['G_eff'])
        
        ax.loglog(r_test/PC_TO_M, np.array(G_eff_values)/G_NEWTON)
        ax.axhline(1, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Radius (pc)')
        ax.set_ylabel('G_eff / G_Newton')
        ax.set_title('Effective Gravitational "Constant"')
        ax.grid(True, alpha=0.3)
        
        # 4. Component breakdown at 1 kpc
        ax = axes[1, 0]
        r_kpc = 1 * KPC_TO_M
        result = self.effective_gravity_from_pressure(r_kpc, v_test, rho_test)
        
        components = ['Pressure', 'Lock-in', 'Information', 'Light']
        values = [abs(result['g_pressure']), abs(result['g_lock']), 
                 abs(result['g_info']), abs(result['g_light'])]
        
        ax.bar(components, values)
        ax.set_ylabel('Acceleration (m/s²)')
        ax.set_title('Gravity Components at 1 kpc')
        ax.set_yscale('log')
        
        # 5. Eight-beat modulation
        ax = axes[1, 1]
        t_values = np.linspace(0, 8*TAU_0, 1000)
        modulation = [self.eight_beat_modulation(t) for t in t_values]
        
        ax.plot(t_values/TAU_0, modulation)
        ax.set_xlabel('Time (τ₀)')
        ax.set_ylabel('Modulation Factor')
        ax.set_title('8-Beat Gravity Modulation')
        ax.grid(True, alpha=0.3)
        
        # 6. Example rotation curve
        ax = axes[1, 2]
        r_gal = np.logspace(np.log10(0.1*KPC_TO_M), np.log10(30*KPC_TO_M), 50)
        
        # Simple exponential disk
        def rho_disk(r):
            r_d = 3 * KPC_TO_M  # disk scale length
            rho_0 = 1e-21  # central density kg/m³
            return rho_0 * np.exp(-r/r_d)
        
        result = self.solve_rotation_curve(r_gal, rho_disk, "Model Galaxy")
        
        ax.plot(result['r']/KPC_TO_M, result['v']/1e3, 'b-', linewidth=2)
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Rotation Velocity (km/s)')
        ax.set_title('Example Rotation Curve')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rs_gravity_v9_pressure_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Analysis plots saved to rs_gravity_v9_pressure_analysis.png")

def main():
    """Demonstrate pressure-based gravity."""
    print("RS GRAVITY v9: PRESSURE-BASED DYNAMICS")
    print("=" * 50)
    print("Implementing P = J_in - J_out from manuscript")
    print()
    
    # Create pressure gravity instance
    gravity = PressureDynamicsGravity()
    
    # Test at various scales
    print("PRESSURE DYNAMICS AT KEY SCALES:")
    print("-" * 50)
    
    test_cases = [
        (20e-9, 1e3, 1e5, "20 nm (molecular)"),
        (1e-6, 1e-3, 1e3, "1 μm (cellular)"),
        (1e3, 1e-10, 10, "1 km (laboratory)"),
        (1*KPC_TO_M, 1e-24, 200e3, "1 kpc (galactic)"),
        (10*KPC_TO_M, 1e-25, 150e3, "10 kpc (galactic edge)")
    ]
    
    for r, rho, v, label in test_cases:
        result = gravity.effective_gravity_from_pressure(r, v, rho)
        print(f"\n{label}:")
        print(f"  Pressure P = {result['pressure']:.2e}")
        print(f"  J_in = {result['J_in']:.2e}, J_out = {result['J_out']:.2e}")
        print(f"  Pressure gradient = {result['pressure_gradient']:.2e}")
        print(f"  G_eff/G_0 = {result['G_eff']/G_NEWTON:.3f}")
        print(f"  g_total = {result['g_total']:.2e} m/s²")
    
    # Key insights
    print("\n" + "=" * 50)
    print("KEY INSIGHTS FROM PRESSURE FORMULATION:")
    print()
    print("1. Gravity emerges from ledger imbalance P = J_in - J_out")
    print("2. Velocity fields directly affect gravity (not just density)")
    print("3. Lock-in events create discrete quantum gravity")
    print("4. 8-beat modulation links gravity to other forces")
    print("5. Information and light contribute to gravitational field")
    print()
    print("This matches the manuscript's fundamental insight:")
    print("Gravity is cosmic accounting, not geometry!")
    print("=" * 50)
    
    # Create analysis plots
    gravity.create_analysis_plots()
    
    # Save parameters
    params_dict = {
        'description': 'RS Gravity v9 Pressure-Based Parameters',
        'key_insight': 'Gravity from P = J_in - J_out, not density',
        'parameters': gravity.params.__dict__
    }
    
    with open('rs_gravity_v9_params.json', 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    print("\nParameters saved to rs_gravity_v9_params.json")

if __name__ == "__main__":
    main() 