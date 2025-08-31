#!/usr/bin/env python3
"""
LNAL SPARC Data Loader
======================
Load and prepare SPARC galaxy data for LNAL gravity fitting.
Integrates with lnal_solver_core_v2.py
"""

import numpy as np
import os
import glob
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy.interpolate import UnivariateSpline

from lnal_solver_core_v2 import GalaxyData, kpc, pc, M_sun

# Constants for mass-to-light ratios
ML_DISK_36 = 0.5  # M/L in 3.6μm band for disk
ML_BULGE_36 = 0.7  # M/L in 3.6μm band for bulge


def parse_rotmod_file(filepath: str) -> Optional[Dict]:
    """
    Parse a single SPARC _rotmod.dat file.
    
    Returns dict with radius, velocities, errors, and baryonic components.
    """
    data = []
    distance = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# Distance'):
                # Extract distance from "# Distance = 3.16 Mpc"
                distance = float(line.split('=')[1].split()[0])
            elif line.startswith('#') or not line:
                continue
            else:
                # Data line: Rad Vobs errV Vgas Vdisk Vbul SBdisk SBbul
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


def compute_surface_density(curve_data: Dict, ml_disk: float = ML_DISK_36, 
                          ml_bulge: float = ML_BULGE_36) -> callable:
    """
    Create surface density function from SPARC components.
    
    Returns a callable Σ(r) that includes gas + disk + bulge.
    """
    r_data = curve_data['r'] * kpc  # Convert to meters
    
    # Gas surface density from velocity (assuming thin disk)
    # V_gas^2 = G * M_gas / r = G * 2π * ∫Σ_gas r dr / r
    V_gas = curve_data['V_gas'] * 1000  # Convert to m/s
    
    # For exponential disk: Σ_gas ≈ V_gas^2 / (2πGr) * correction_factor
    # This is approximate - better to use HI data directly if available
    from lnal_solver_core_v2 import G
    Sigma_gas = V_gas**2 / (2 * np.pi * G * r_data) * 0.85  # Correction factor
    
    # Stellar surface densities from surface brightness
    # Σ = M/L * 10^(0.4*(M_sun_band - SB))
    # For 3.6μm band: M_sun,3.6 = 3.24
    M_sun_36 = 3.24
    
    # Convert surface brightness to surface density
    Sigma_disk = ml_disk * 10**(0.4 * (M_sun_36 - curve_data['SB_disk'])) * M_sun / pc**2
    Sigma_bulge = ml_bulge * 10**(0.4 * (M_sun_36 - curve_data['SB_bul'])) * M_sun / pc**2
    
    # Handle zeros and invalid values
    Sigma_disk[curve_data['SB_disk'] == 0] = 0
    Sigma_bulge[curve_data['SB_bul'] == 0] = 0
    Sigma_gas[np.isnan(Sigma_gas) | np.isinf(Sigma_gas)] = 0
    
    # Total surface density
    Sigma_total = Sigma_gas + Sigma_disk + Sigma_bulge
    
    # Create interpolating function
    # Use log-spline for better behavior at small radii
    valid = (r_data > 0) & (Sigma_total > 0)
    if np.sum(valid) < 4:
        # Not enough valid points
        return lambda r: np.zeros_like(r)
    
    spline = UnivariateSpline(
        np.log(r_data[valid]), 
        np.log(Sigma_total[valid]), 
        s=0, k=3, ext='extrapolate'
    )
    
    def surface_density(r):
        """Interpolated surface density [kg/m²]"""
        r = np.atleast_1d(r)
        result = np.zeros_like(r)
        valid_r = r > 0
        if np.any(valid_r):
            result[valid_r] = np.exp(spline(np.log(r[valid_r])))
        return result
    
    return surface_density


def estimate_scale_length(curve_data: Dict) -> float:
    """
    Estimate disk scale length from surface brightness profile.
    
    For exponential disk: SB(r) = SB_0 + 1.086 * r / R_d
    """
    r = curve_data['r']
    SB_disk = curve_data['SB_disk']
    
    # Find valid disk data
    valid = (SB_disk > 0) & (r > 0.5) & (r < 10.0)  # Focus on disk-dominated region
    
    if np.sum(valid) < 5:
        # Not enough points, use rough estimate based on galaxy size
        return max(2.0 * kpc, 0.2 * r.max() * kpc)
    
    # Fit exponential profile
    from scipy.optimize import curve_fit
    
    def exp_profile(r, SB_0, R_d):
        return SB_0 + 1.086 * r / R_d
    
    try:
        # Better initial guess based on data
        r_half = r[valid][len(r[valid])//2]
        popt, _ = curve_fit(
            exp_profile, 
            r[valid], 
            SB_disk[valid], 
            p0=[np.median(SB_disk[valid]), r_half],
            bounds=([10, 0.1], [30, 20])
        )
        R_d = popt[1] * kpc
        
        # Sanity check
        if R_d < 0.1 * kpc or R_d > 20 * kpc:
            R_d = max(2.0 * kpc, 0.2 * r.max() * kpc)
    except:
        # Fallback estimate
        R_d = max(2.0 * kpc, 0.2 * r.max() * kpc)
    
    return R_d


def load_sparc_galaxy(galaxy_name: str, rotmod_dir: str = 'Rotmod_LTG') -> Optional[GalaxyData]:
    """
    Load a single SPARC galaxy as GalaxyData object.
    """
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
    surface_density = compute_surface_density(curve_data)
    
    # Estimate scale length
    scale_length = estimate_scale_length(curve_data)
    
    return GalaxyData(
        name=galaxy_name,
        r=r,
        v_obs=v_obs,
        v_err=v_err,
        surface_density=surface_density,
        scale_length=scale_length,
        inclination=0.0  # Already corrected in SPARC data
    )


def load_all_sparc_galaxies(rotmod_dir: str = 'Rotmod_LTG', 
                           quality_cut: bool = True) -> Dict[str, GalaxyData]:
    """
    Load all SPARC galaxies.
    
    Parameters:
    - rotmod_dir: Directory containing *_rotmod.dat files
    - quality_cut: If True, only include high-quality galaxies
    """
    pattern = os.path.join(rotmod_dir, '*_rotmod.dat')
    rotmod_files = glob.glob(pattern)
    
    print(f"Found {len(rotmod_files)} rotation curve files")
    
    galaxies = {}
    failed = 0
    
    for filepath in rotmod_files:
        galaxy_name = os.path.basename(filepath).replace('_rotmod.dat', '')
        
        try:
            galaxy_data = load_sparc_galaxy(galaxy_name, rotmod_dir)
            if galaxy_data is not None:
                # Quality cuts
                if quality_cut:
                    # Require at least 5 data points
                    if len(galaxy_data.r) < 5:
                        continue
                    # Require reasonable velocity range
                    if np.max(galaxy_data.v_obs) < 30000:  # 30 km/s minimum
                        continue
                    # Skip if too many bad errors
                    if np.sum(galaxy_data.v_err <= 0) > len(galaxy_data.v_err) / 2:
                        continue
                
                galaxies[galaxy_name] = galaxy_data
            else:
                failed += 1
        except Exception as e:
            print(f"Error loading {galaxy_name}: {e}")
            failed += 1
    
    print(f"Successfully loaded {len(galaxies)} galaxies ({failed} failed)")
    return galaxies


def get_galaxy_sample(sample_type: str = 'high_quality') -> List[str]:
    """
    Get predefined galaxy samples for analysis.
    """
    samples = {
        'test': ['NGC2403', 'NGC3198', 'NGC6503', 'DDO154', 'UGC02885'],
        'high_quality': [
            'NGC2403', 'NGC3198', 'NGC6503', 'NGC2841', 'NGC7814',
            'NGC0891', 'NGC5055', 'NGC3521', 'NGC7331', 'NGC0300',
            'DDO154', 'DDO168', 'UGC02885', 'UGC06399', 'UGC04499'
        ],
        'dwarfs': [
            'DDO154', 'DDO168', 'DDO126', 'DDO87', 'DDO64',
            'UGCA281', 'UGCA442', 'UGCA444', 'CVnIdwA', 'WLM'
        ],
        'massive': [
            'UGC02885', 'UGC12591', 'NGC7814', 'NGC2841', 'NGC5055',
            'NGC3521', 'NGC7331', 'NGC0891', 'NGC4736', 'NGC5457'
        ]
    }
    
    return samples.get(sample_type, samples['test'])


def test_loader():
    """Test the SPARC loader"""
    print("Testing SPARC Data Loader")
    print("=" * 60)
    
    # Test single galaxy
    galaxy = load_sparc_galaxy('NGC3198')
    
    if galaxy is not None:
        print(f"Successfully loaded {galaxy.name}")
        print(f"  Data points: {len(galaxy.r)}")
        print(f"  R range: {galaxy.r[0]/kpc:.2f} - {galaxy.r[-1]/kpc:.2f} kpc")
        print(f"  V range: {galaxy.v_obs.min()/1000:.1f} - {galaxy.v_obs.max()/1000:.1f} km/s")
        print(f"  Scale length: {galaxy.scale_length/kpc:.2f} kpc")
        
        # Test surface density
        r_test = np.array([1, 5, 10]) * kpc
        Sigma_test = galaxy.surface_density(r_test)
        print(f"  Σ(1 kpc) = {Sigma_test[0]:.2e} kg/m²")
        print(f"  Σ(5 kpc) = {Sigma_test[1]:.2e} kg/m²")
    else:
        print("Failed to load NGC3198")
    
    # Test batch loading
    print("\nTesting batch loading...")
    test_galaxies = get_galaxy_sample('test')
    loaded = 0
    
    for name in test_galaxies:
        galaxy = load_sparc_galaxy(name)
        if galaxy is not None:
            loaded += 1
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
    
    print(f"\nLoaded {loaded}/{len(test_galaxies)} test galaxies")


if __name__ == "__main__":
    test_loader() 