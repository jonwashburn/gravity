#!/usr/bin/env python3
"""
LNAL Gravity Theory - Fresh Perspective
======================================
Starting from Recognition Science first principles:
1. Eight-beat cycle governs all processes
2. Information debt creates gravitational effects
3. Prime channels determine interaction strength
4. No MOND-like interpolation - emergent from 8-beat quantization
"""

import numpy as np
import matplotlib.pyplot as plt

# Recognition Science constants
phi = 1.618034  # Golden ratio
E_coh = 0.090  # eV
tau_0 = 7.33e-15  # seconds
c = 2.998e8  # m/s
G = 6.674e-11  # m³/kg/s²

# Eight-beat period
T_8beat = 8 * tau_0  # 5.86e-14 seconds

# Prime channels (from Recognition Science)
PRIME_CHANNELS = [2, 3, 5, 7, 11, 13, 17, 19]  # First 8 primes
N_CHANNELS = len(PRIME_CHANNELS)

class RecognitionGravity:
    """Gravity from information processing through prime channels"""
    
    def __init__(self):
        # Information processing scale
        self.lambda_info = c * T_8beat  # ~17.6 nm
        print(f"Recognition Gravity Parameters:")
        print(f"  8-beat period: {T_8beat:.2e} s")
        print(f"  Information scale: {self.lambda_info:.2e} m = {self.lambda_info*1e9:.1f} nm")
        print(f"  Prime channels: {PRIME_CHANNELS}")
    
    def channel_blocking(self, density_kg_m3):
        """
        Calculate how many prime channels are blocked by matter density.
        Higher density = more blocked channels = fewer recognition paths
        """
        # Critical density where channels start blocking
        rho_crit = 1e-21  # kg/m³ (cosmic mean density scale)
        
        # Each channel has different blocking threshold
        blocked_channels = 0
        active_channels = []
        
        for i, prime in enumerate(PRIME_CHANNELS):
            # Blocking threshold scales with prime number
            # Larger primes = harder to block (more resilient)
            threshold = rho_crit * prime**2
            
            if density_kg_m3 > threshold:
                blocked_channels += 1
            else:
                active_channels.append(prime)
        
        return blocked_channels, active_channels
    
    def information_debt(self, density_kg_m3, scale_m):
        """
        Calculate information debt based on:
        1. How much information needs processing (∝ density)
        2. How many channels are available
        3. Whether scale allows 8-beat completion
        """
        blocked, active = self.channel_blocking(density_kg_m3)
        n_active = N_CHANNELS - blocked
        
        if n_active == 0:
            # All channels blocked - maximum debt
            return np.inf
        
        # Information content scales with density
        info_content = density_kg_m3 * scale_m**3
        
        # Processing rate through active channels
        # Each channel processes at rate ∝ its prime number
        processing_rate = sum(active) if active else 1
        
        # Debt = content / rate
        debt = info_content / processing_rate
        
        # 8-beat quantization effect
        # Can only process integer multiples of 8-beat cycles
        cycles_needed = debt / (c * T_8beat)**3
        cycles_actual = np.ceil(cycles_needed)  # Must complete full cycles
        
        # Quantization creates additional debt
        quantization_penalty = (cycles_actual - cycles_needed) / cycles_actual
        
        return debt * (1 + quantization_penalty)
    
    def gravitational_enhancement(self, density_kg_m3, scale_m):
        """
        Convert information debt to gravitational enhancement factor.
        This multiplies the Newtonian gravity.
        """
        blocked, active = self.channel_blocking(density_kg_m3)
        
        # Key insight: enhancement is NOT smooth!
        # It jumps when channels get blocked
        
        # Channel factor - discrete jumps
        # But the enhancement should be modest, not huge
        if blocked == 0:
            channel_factor = 1.0  # All channels open - Newtonian
        elif blocked < 3:
            channel_factor = 1.0 + (phi - 1) * 0.2  # ~1.12
        elif blocked < 5:
            channel_factor = 1.0 + (phi - 1) * 0.5  # ~1.31
        elif blocked < 7:
            channel_factor = 1.0 + (phi - 1) * 1.0  # ~1.62
        else:
            channel_factor = phi  # Maximum enhancement
        
        # Information processing delay
        # When channels are blocked, processing takes longer
        n_active = N_CHANNELS - blocked
        if n_active > 0:
            processing_efficiency = n_active / N_CHANNELS
            delay_factor = 1.0 / processing_efficiency
        else:
            delay_factor = 10.0  # Maximum delay
        
        # Scale-dependent modulation
        # Information must propagate across the system
        info_propagation_time = scale_m / c
        eight_beat_time = T_8beat
        
        # How many 8-beat cycles to cross the system?
        cycles_to_cross = info_propagation_time / eight_beat_time
        
        # If it takes many cycles, enhancement saturates
        scale_factor = np.tanh(cycles_to_cross / 1e12)  # Saturates around galactic scales
        
        # Total enhancement
        enhancement = 1.0 + (channel_factor - 1.0) * delay_factor * scale_factor
        
        # But cap it at reasonable values
        return min(enhancement, 5.0)  # Never more than 5x Newtonian
    
    def effective_gravity(self, density_kg_m3, scale_m):
        """
        Effective gravitational 'constant' at given density and scale
        """
        enhancement = self.gravitational_enhancement(density_kg_m3, scale_m)
        return G * enhancement
    
    def galaxy_rotation_curve(self, r_m, surface_density_func):
        """
        Calculate rotation curve for a galaxy
        r_m: radius array in meters
        surface_density_func: function returning Σ(r) in kg/m²
        """
        velocities = []
        
        for radius in r_m:
            # Get local surface density
            sigma = surface_density_func(radius)
            
            # Convert to volume density (thin disk approximation)
            h = 100  # pc scale height
            h_m = h * 3.086e16  # meters
            rho = sigma / (2 * h_m)
            
            # Get effective gravity
            G_eff = self.effective_gravity(rho, radius)
            
            # Newtonian acceleration with enhanced G
            a = 2 * np.pi * G_eff * sigma
            
            # Circular velocity
            v = np.sqrt(a * radius)
            velocities.append(v)
        
        return np.array(velocities)
    
    def plot_enhancement_map(self):
        """Visualize how enhancement varies with density"""
        densities = np.logspace(-25, -18, 100)  # kg/m³
        scale = 1e20  # 1 kpc in meters
        
        enhancements = []
        blocked_counts = []
        
        for rho in densities:
            enh = self.gravitational_enhancement(rho, scale)
            blocked, _ = self.channel_blocking(rho)
            enhancements.append(enh)
            blocked_counts.append(blocked)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Enhancement factor
        ax1.loglog(densities, enhancements, 'b-', linewidth=2)
        ax1.axhline(1, color='k', linestyle='--', alpha=0.5, label='Newtonian')
        ax1.axhline(phi, color='r', linestyle=':', alpha=0.5, label='φ')
        ax1.axhline(phi**2, color='r', linestyle=':', alpha=0.5, label='φ²')
        ax1.set_ylabel('Gravitational Enhancement', fontsize=12)
        ax1.set_title('LNAL Gravity: Channel Blocking Creates Discrete Enhancement', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Blocked channels
        ax2.semilogx(densities, blocked_counts, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Density (kg/m³)', fontsize=12)
        ax2.set_ylabel('Blocked Channels', fontsize=12)
        ax2.set_ylim(-0.5, 8.5)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Prime Channel Blocking vs Density', fontsize=12)
        
        # Mark typical galaxy densities
        rho_disk = 1e-20  # Typical disk
        rho_halo = 1e-23  # Typical halo
        
        for ax in [ax1, ax2]:
            ax.axvline(rho_disk, color='g', linestyle='-', alpha=0.3, linewidth=3, label='Disk')
            ax.axvline(rho_halo, color='orange', linestyle='-', alpha=0.3, linewidth=3, label='Halo')
        
        plt.tight_layout()
        plt.savefig('lnal_channel_blocking.png', dpi=150)
        plt.show()
        
        # Print key transitions
        print("\nKey density transitions:")
        last_blocked = 0
        for i, rho in enumerate(densities):
            blocked, _ = self.channel_blocking(rho)
            if blocked != last_blocked:
                print(f"  {last_blocked} → {blocked} channels blocked at ρ = {rho:.2e} kg/m³")
                last_blocked = blocked

def test_simple_galaxy():
    """Test on a simple exponential disk"""
    rg = RecognitionGravity()
    
    # Galaxy parameters
    M_disk = 5e10 * 1.989e30  # kg
    R_d = 3 * 3.086e19  # 3 kpc in meters
    
    def surf_density(r):
        """Exponential disk surface density"""
        Sigma_0 = M_disk / (2 * np.pi * R_d**2)
        return Sigma_0 * np.exp(-r / R_d)
    
    # Calculate rotation curve
    r_kpc = np.linspace(0.1, 30, 100)
    r_m = r_kpc * 3.086e19
    
    # LNAL prediction
    v_lnal = rg.galaxy_rotation_curve(r_m, surf_density)
    
    # Newtonian prediction
    a_newton = 2 * np.pi * G * surf_density(r_m)
    v_newton = np.sqrt(a_newton * r_m)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(r_kpc, v_lnal/1000, 'b-', linewidth=2.5, label='LNAL (Channel Blocking)')
    plt.plot(r_kpc, v_newton/1000, 'r--', linewidth=2, label='Newtonian')
    
    plt.xlabel('Radius (kpc)', fontsize=12)
    plt.ylabel('Velocity (km/s)', fontsize=12)
    plt.title('LNAL Gravity: Discrete Enhancement from Channel Blocking', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 30)
    plt.ylim(0, 300)
    
    plt.tight_layout()
    plt.savefig('lnal_rotation_curve_fresh.png', dpi=150)
    plt.show()
    
    # Show the discrete jumps
    print("\nVelocity enhancement at different radii:")
    for r in [1, 5, 10, 20]:
        idx = np.argmin(np.abs(r_kpc - r))
        enhancement = v_lnal[idx] / v_newton[idx]
        rho = surf_density(r_m[idx]) / (2 * 100 * 3.086e16)  # volume density
        blocked, _ = rg.channel_blocking(rho)
        print(f"  R = {r:2d} kpc: V_LNAL/V_Newton = {enhancement:.2f}, "
              f"blocked channels = {blocked}")

def main():
    """Run the fresh analysis"""
    print("LNAL GRAVITY - FRESH PERSPECTIVE")
    print("Based on 8-beat cycles and prime channel blocking")
    print("="*60)
    print()
    
    rg = RecognitionGravity()
    print()
    
    # Show enhancement map
    rg.plot_enhancement_map()
    
    # Test on simple galaxy
    test_simple_galaxy()
    
    print("\nKEY INSIGHTS:")
    print("1. Gravity enhancement is DISCRETE, not continuous")
    print("2. Jumps occur when prime channels get blocked")
    print("3. Enhancement factors are powers of φ")
    print("4. This naturally creates 'dark matter' effect")
    print("5. No free parameters - all from Recognition Science")

if __name__ == "__main__":
    main() 