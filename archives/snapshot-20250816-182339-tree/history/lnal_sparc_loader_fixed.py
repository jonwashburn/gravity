#!/usr/bin/env python3
"""
Fixed LNAL SPARC Data Loader
=============================
Improved surface density computation to avoid numerical issues.
"""

import numpy as np
import os
from typing import Dict, Optional
from scipy.interpolate import UnivariateSpline

from lnal_solver_core_v2 import GalaxyData, kpc, pc, M_sun, G

# Constants for mass-to-light ratios
ML_DISK_36 = 0.5  # M/L in 3.6μm band for disk
ML_BULGE_36 = 0.7  # M/L in 3.6μm band for bulge


def parse_rotmod_file(filepath: str) -> Optional[Dict]:
    """Parse a single SPARC _rotmod.dat file."""
    data = []
    distance = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# Distance'):
                distance = float(line.split('=')[1].split()[0])
            elif line.startswith('#') or not line:
                continue
            else:
                parts = line.split()
                if len(parts) >= 8:
                    data.append([float(p) for p in parts])
    
    if not data:
        return None
        
    data = np.array(data)
    
    return {
        'distance': distance,  # Mpc
        'r': data[:, 0],       # kpc
        'V_obs': data[:, 1],   # km/s
        'e_V': data[:, 2],     # km/s
        'V_gas': data[:, 3],   # km/s (HI component)
        'V_disk': data[:, 4],  # km/s (stellar disk)
        'V_bul': data[:, 5],   # km/s (bulge)
        'SB_disk': data[:, 6], # L_sun/pc^2
        'SB_bul': data[:, 7]   # L_sun/pc^2
    }


def compute_surface_density_simple(curve_data: Dict) -> callable:
    """
    Create surface density function from velocity components.
    Uses the relation: V^2 = G * M_enc / r
    """
    r_data = curve_data['r'] * kpc  # Convert to meters
    
    # Total baryonic velocity from components
    V_gas = curve_data['V_gas'] * 1000  # m/s
    V_disk = curve_data['V_disk'] * 1000
    V_bul = curve_data['V_bul'] * 1000
    
    # Total baryonic velocity (add in quadrature)
    V_baryon = np.sqrt(V_gas**2 + V_disk**2 + V_bul**2)
    
    # Enclosed mass from velocity: M_enc = V^2 * r / G
    M_enc = V_baryon**2 * r_data / G
    
    # Surface density from mass gradient
    # For thin disk: Σ(r) ≈ (1/2πr) * dM_enc/dr
    Sigma_total = np.zeros_like(r_data)
    
    # Compute derivative using finite differences
    for i in range(1, len(r_data)-1):
        dM_dr = (M_enc[i+1] - M_enc[i-1]) / (r_data[i+1] - r_data[i-1])
        Sigma_total[i] = dM_dr / (2 * np.pi * r_data[i])
    
    # Handle boundaries
    Sigma_total[0] = Sigma_total[1]
    Sigma_total[-1] = Sigma_total[-2]
    
    # Ensure positive values
    Sigma_total = np.maximum(Sigma_total, 1e-10)
    
    # Create interpolating function
    valid = (r_data > 0) & (Sigma_total > 0)
    if np.sum(valid) < 4:
        # Fallback to constant density
        return lambda r: np.ones_like(r) * 1e-3
    
    # Use linear interpolation in log space for stability
    log_r = np.log10(r_data[valid])
    log_Sigma = np.log10(Sigma_total[valid])
    
    # Create spline with smoothing
    spline = UnivariateSpline(log_r, log_Sigma, s=0.5, k=3, ext='extrapolate')
    
    def surface_density(r):
        """Interpolated surface density [kg/m²]"""
        r = np.atleast_1d(r)
        result = np.zeros_like(r, dtype=float)
        valid_r = r > 0
        if np.any(valid_r):
            log_r_eval = np.log10(r[valid_r])
            # Clip extrapolation to reasonable range
            log_r_eval = np.clip(log_r_eval, log_r.min() - 0.5, log_r.max() + 0.5)
            result[valid_r] = 10**spline(log_r_eval)
        return result
    
    return surface_density


def estimate_scale_length_from_velocity(curve_data: Dict) -> float:
    """
    Estimate disk scale length from velocity profile.
    For exponential disk, velocity peaks at ~2.2 R_d
    """
    r = curve_data['r']
    V_disk = curve_data['V_disk']
    
    # Find where disk velocity peaks
    valid = V_disk > 0
    if np.sum(valid) < 5:
        # Fallback estimate
        return 3.0 * kpc
    
    # Find peak
    idx_peak = np.argmax(V_disk[valid])
    r_peak = r[valid][idx_peak]
    
    # Scale length estimate
    R_d = r_peak / 2.2 * kpc
    
    # Sanity check
    if R_d < 0.5 * kpc or R_d > 15 * kpc:
        R_d = 3.0 * kpc
    
    return R_d


def load_sparc_galaxy_fixed(galaxy_name: str, rotmod_dir: str = 'Rotmod_LTG') -> Optional[GalaxyData]:
    """Load a single SPARC galaxy with improved surface density."""
    filepath = os.path.join(rotmod_dir, f'{galaxy_name}_rotmod.dat')
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None
    
    # Parse rotation curve file
    curve_data = parse_rotmod_file(filepath)
    if curve_data is None:
        return None
    
    # Convert units
    r = curve_data['r'] * kpc  # kpc to m
    v_obs = curve_data['V_obs'] * 1000  # km/s to m/s
    v_err = curve_data['e_V'] * 1000  # km/s to m/s
    
    # Handle missing errors
    v_err[v_err <= 0] = 5000  # Default 5 km/s error
    
    # Create surface density function
    surface_density = compute_surface_density_simple(curve_data)
    
    # Estimate scale length
    scale_length = estimate_scale_length_from_velocity(curve_data)
    
    return GalaxyData(
        name=galaxy_name,
        r=r,
        v_obs=v_obs,
        v_err=v_err,
        surface_density=surface_density,
        scale_length=scale_length,
        inclination=0.0  # Already corrected in SPARC data
    )


def test_fixed_loader():
    """Test the fixed loader"""
    print("Testing Fixed SPARC Loader")
    print("=" * 60)
    
    galaxy = load_sparc_galaxy_fixed('NGC3198')
    
    if galaxy is not None:
        print(f"Successfully loaded {galaxy.name}")
        print(f"  Scale length: {galaxy.scale_length/kpc:.2f} kpc")
        
        # Test surface density
        r_test = np.array([1, 5, 10, 20]) * kpc
        Sigma_test = galaxy.surface_density(r_test)
        print(f"\nSurface density profile:")
        for r, s in zip(r_test/kpc, Sigma_test):
            print(f"  Σ({r:.0f} kpc) = {s:.2e} kg/m²")
            
        # Check for reasonable values
        if np.all(np.isfinite(Sigma_test)) and np.all(Sigma_test > 0):
            print("\n✓ Surface density values look reasonable")
        else:
            print("\n✗ Surface density has issues")


if __name__ == "__main__":
    test_fixed_loader() 