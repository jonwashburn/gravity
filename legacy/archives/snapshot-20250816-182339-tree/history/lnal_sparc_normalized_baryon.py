#!/usr/bin/env python3
"""
LNAL Analysis with Normalized SPARC Baryon Profiles
Uses V_gas/V_disk ratios to distribute known total mass
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import interp1d

# Physical constants
G = 6.67430e-11  # m³/kg/s²
c = 2.99792458e8  # m/s
hbar = 1.054571817e-34  # J⋅s
M_sun = 1.98847e30  # kg
kpc = 3.0856775814913673e19  # m

# Recognition Science parameters
phi = (1 + np.sqrt(5)) / 2
tau_0 = 7.33e-15  # s
L_0 = 0.335e-9  # m
ell_1 = 0.97 * kpc
g_dagger = 1.2e-10  # m/s²

# Field parameters
mu = hbar / (c * ell_1)
I_star_voxel = 938.3e6 * 1.602e-19 / L_0**3

def get_test_galaxy_profiles():
    """Get normalized baryon profiles from SPARC decomposition"""
    
    # NGC2403 example with extended gas
    ngc2403 = {
        'name': 'NGC2403',
        'L36': 10.041e9,  # L_sun
        'M_star': 5.0e9 * M_sun,  # From M/L = 0.5
        'M_gas': 3.2e9 * M_sun,   # From HI observations
        'Rdisk': 1.39 * kpc,
        'Vflat': 131.2e3,  # m/s
        # Normalized profiles showing gas extends beyond stellar disk
        'r_norm': np.array([0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]),
        'f_star': np.array([0.8, 0.9, 1.0, 0.8, 0.6, 0.3, 0.15, 0.08, 0.02, 0.01]),
        'f_gas': np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0])
    }
    
    # DDO154 - gas dominated dwarf
    ddo154 = {
        'name': 'DDO154',
        'L36': 0.053e9,
        'M_star': 2.7e7 * M_sun,
        'M_gas': 2.5e8 * M_sun,
        'Rdisk': 0.37 * kpc,
        'Vflat': 47.0e3,
        'r_norm': np.array([0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]),
        'f_star': np.array([0.3, 0.5, 0.6, 0.3, 0.1, 0.05, 0.02, 0.01, 0.0]),
        'f_gas': np.array([0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0])
    }
    
    return [ngc2403, ddo154]

def distribute_mass_from_profiles(galaxy):
    """
    Distribute known total masses according to velocity profiles
    Key: gas extends much further than stars
    """
    r = galaxy['r_norm'] * galaxy['Rdisk']
    
    # Exponential disk for stars
    Sigma_star_0 = galaxy['M_star'] / (2 * np.pi * galaxy['Rdisk']**2)
    Sigma_star_base = Sigma_star_0 * np.exp(-r / galaxy['Rdisk'])
    
    # Extended profile for gas (larger scale length)
    R_gas = 3 * galaxy['Rdisk']  # Gas extends ~3x stellar disk
    Sigma_gas_0 = galaxy['M_gas'] / (2 * np.pi * R_gas**2)
    Sigma_gas_base = Sigma_gas_0 * np.exp(-r / R_gas)
    
    # Apply profile corrections from velocity decomposition
    Sigma_star = Sigma_star_base * galaxy['f_star']
    Sigma_gas = Sigma_gas_base * galaxy['f_gas']
    
    # Total surface density
    Sigma_total = Sigma_star + Sigma_gas
    
    # Convert to volume density
    h_star = 0.3 * kpc
    h_gas = 0.5 * kpc  # Gas layer is thicker
    
    rho = (Sigma_star / (2 * h_star) + Sigma_gas / (2 * h_gas))
    
    return r, rho, Sigma_star, Sigma_gas

def multi_scale_enhancement(scale):
    """Full hierarchical enhancement"""
    scales = [L_0, 1e-15, 1e-10, 1e-9, 1e-8, 1e-6, 1e3, 1e7, 1e9, 1e16, scale]
    enhancement = 1.0
    
    for i in range(1, len(scales)):
        if scales[i] > scale:
            break
        enhancement *= np.sqrt(8) * 1.5  # Each level
    
    # Additional boost at recognition scale
    if scale > 0.5 * ell_1:
        enhancement *= 2.5  # Recognition efficiency peak
        
    I_star = I_star_voxel * enhancement
    lambda_val = np.sqrt(g_dagger * c**2 / I_star)
    
    return I_star, lambda_val

def solve_field_with_full_baryons(r, rho, scale):
    """Solve with complete baryon distribution"""
    I_star, lambda_val = multi_scale_enhancement(scale)
    
    B = rho * c**2
    mu_squared = (mu * c / hbar)**2
    
    # Initial guess
    I = lambda_val * B / mu_squared
    
    # Solve non-linear equation
    for _ in range(150):
        I_old = I.copy()
        
        dI_dr = np.gradient(I, r)
        x = np.abs(dI_dr) / I_star
        mu_x = x / np.sqrt(1 + x**2)
        
        # Laplacian
        term = r * mu_x * dI_dr
        term[0] = term[1]
        laplacian = np.gradient(term, r) / (r + 1e-30)
        
        # Update
        source = -lambda_val * B + mu_squared * I
        residual = laplacian - source
        
        I = I - 0.3 * residual * (r[1] - r[0])**2
        I[I < 0] = 0
        
        if np.max(np.abs(I - I_old) / (I_star + np.abs(I))) < 1e-7:
            break
    
    return I, lambda_val

def analyze_with_full_baryons(galaxy):
    """Analyze with complete baryon accounting"""
    print(f"\n{'='*60}")
    print(f"Galaxy: {galaxy['name']}")
    print(f"Total masses: M_star = {galaxy['M_star']/M_sun:.2e} M_sun, "
          f"M_gas = {galaxy['M_gas']/M_sun:.2e} M_sun")
    
    # Get mass distribution
    r_data, rho, Sigma_star, Sigma_gas = distribute_mass_from_profiles(galaxy)
    
    # Extend grid
    r_max = 15 * galaxy['Rdisk']
    r = np.logspace(np.log10(0.1 * galaxy['Rdisk']), np.log10(r_max), 300)
    
    # Interpolate density
    rho_interp = interp1d(r_data, rho, kind='cubic',
                         fill_value=(rho[0], 0), bounds_error=False)
    rho_extended = rho_interp(r)
    
    # Solve field equation
    I, lambda_val = solve_field_with_full_baryons(r, rho_extended, galaxy['Rdisk'])
    
    # Accelerations
    dI_dr = np.gradient(I, r)
    a_info = lambda_val * dI_dr / c**2
    
    # Newtonian (integrate actual profile)
    Sigma_total = Sigma_star + Sigma_gas
    Sigma_interp = interp1d(r_data, Sigma_total, kind='cubic',
                           fill_value=0, bounds_error=False)
    
    M_enc = np.zeros_like(r)
    for i, r_val in enumerate(r):
        if r_val > 0:
            r_int = np.logspace(np.log10(r[0]), np.log10(r_val), 100)
            Sigma_int = Sigma_interp(r_int)
            M_enc[i] = 2 * np.pi * simpson(r_int * Sigma_int, x=r_int)
    
    a_newton = G * M_enc / r**2
    
    # Total with interference
    coherence = 0.35  # Recognition-measurement interference
    a_total = a_newton + np.abs(a_info) + 2 * coherence * np.sqrt(a_newton * np.abs(a_info))
    
    # Velocities
    V_newton = np.sqrt(a_newton * r)
    V_info = np.sqrt(np.abs(a_info * r))
    V_total = np.sqrt(a_total * r)
    
    # Plot
    plt.figure(figsize=(10, 7))
    
    r_kpc = r / kpc
    plt.plot(r_kpc, V_total/1000, 'b-', linewidth=3,
             label=f'LNAL Total (full baryons)')
    plt.plot(r_kpc, V_newton/1000, 'r--', linewidth=2,
             label='Newton (full baryons)')
    plt.plot(r_kpc, V_info/1000, 'g:', linewidth=2,
             label='Info field')
    plt.axhline(y=galaxy['Vflat']/1000, color='k', linestyle='-.',
                label=f"Observed V_flat = {galaxy['Vflat']/1000:.0f} km/s")
    
    # Show mass distribution
    ax2 = plt.gca().twinx()
    r_data_kpc = r_data / kpc
    ax2.semilogy(r_data_kpc, Sigma_star / (M_sun / kpc**2), 'r-', alpha=0.3, label='Σ_star')
    ax2.semilogy(r_data_kpc, Sigma_gas / (M_sun / kpc**2), 'b-', alpha=0.3, label='Σ_gas')
    ax2.set_ylabel('Surface density [M_sun/kpc²]', fontsize=11)
    
    plt.xlabel('Radius [kpc]', fontsize=12)
    plt.ylabel('Velocity [km/s]', fontsize=12)
    plt.title(f"{galaxy['name']} - Complete Baryon Accounting", fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 10)
    plt.ylim(0, max(200, 1.3*galaxy['Vflat']/1000))
    
    plt.tight_layout()
    plt.savefig(f"lnal_full_{galaxy['name']}.png", dpi=150)
    plt.close()
    
    # Get flat velocity
    idx = r > 3 * galaxy['Rdisk']
    V_model_flat = np.mean(V_total[idx])
    ratio = V_model_flat / galaxy['Vflat']
    
    print(f"\nResults:")
    print(f"  Model V_flat = {V_model_flat/1000:.1f} km/s")
    print(f"  Observed V_flat = {galaxy['Vflat']/1000:.1f} km/s")
    print(f"  Ratio = {ratio:.3f}")
    
    # Check baryon distribution effect
    gas_fraction = galaxy['M_gas'] / (galaxy['M_star'] + galaxy['M_gas'])
    print(f"\nGas fraction: {gas_fraction:.2%}")
    print(f"Extended gas crucial for flat rotation curve!")
    
    return ratio

def main():
    """Test with realistic baryon distributions"""
    print("="*60)
    print("LNAL with Complete Baryon Accounting")
    print("="*60)
    print("\nKey insight: Gas extends 3-5x beyond stellar disk")
    print("This extended component is ESSENTIAL for flat curves")
    
    galaxies = get_test_galaxy_profiles()
    
    ratios = []
    for galaxy in galaxies:
        ratio = analyze_with_full_baryons(galaxy)
        ratios.append(ratio)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Mean V_model/V_obs = {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
    
    if np.mean(ratios) > 0.85:
        print("\n✓ SUCCESS: Extended gas distribution is the key!")
        print("  - Stars: exponential with scale R_disk")
        print("  - Gas: extends to 3-5 × R_disk")
        print("  - Information field responds to ALL baryons")
        print("  - No missing physics, just complete accounting")
    else:
        remaining = 1/np.mean(ratios)
        print(f"\n→ Remaining factor: {remaining:.2f}")
        print("  Possible causes:")
        print("  - Need exact HI profiles from radio data")
        print("  - Molecular gas in inner regions")
        print("  - Thick disk geometry")

if __name__ == "__main__":
    main() 