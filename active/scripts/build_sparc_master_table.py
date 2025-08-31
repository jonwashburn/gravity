#!/usr/bin/env python3
"""
Build master table for all SPARC galaxies with:
- Radius profiles and velocities
- True gas fractions (HI + H2 estimate)
- Dynamical times T_dyn(r)
- Surface brightness proxies
- Disk scale lengths
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Constants
kpc_to_m = 3.086e19
M_sun = 1.989e30

def load_sparc_catalog():
    """Load the main SPARC catalog with galaxy properties"""
    # For now, we'll generate synthetic catalog data
    # In real implementation, this would read the actual SPARC_Lelli2016c.mrt file
    catalog = {}
    return catalog

def estimate_H2_mass(M_HI, M_star):
    """Estimate molecular hydrogen mass from HI and stellar mass"""
    # Empirical relation: more metal-rich galaxies have more H2
    if M_star <= 0 or M_HI <= 0:
        return 0
    
    # Metallicity proxy
    Z_ratio = (M_star / 1e10) ** 0.3
    
    # H2/HI ratio increases with metallicity
    H2_HI_ratio = 0.4 * Z_ratio  # Typical values 0.1-1.0
    
    return H2_HI_ratio * M_HI

def load_rotation_curve(path):
    """Load rotation curve data from rotmod file"""
    cols = ["rad", "vobs", "verr", "vgas", "vdisk", "vbul", "sbdisk", "sbbul"]
    
    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#", names=cols)
        df = df[(df["rad"] > 0) & (df["vobs"] > 0) & (df["verr"] > 0)].reset_index(drop=True)
        
        # Extract galaxy name
        name = path.stem.replace("_rotmod", "")
        
        # Calculate some aggregate properties
        v2_gas = np.mean(df["vgas"]**2)
        v2_disk = np.mean(df["vdisk"]**2) 
        v2_bul = np.mean(df["vbul"]**2)
        v2_total = v2_gas + v2_disk + v2_bul
        
        # Estimate masses from velocities (simplified)
        # M ≈ v²R/G, use typical radius
        G_kpc = 4.302e-6  # kpc (km/s)² / M_sun
        R_typical = np.median(df["rad"])
        
        M_gas_est = v2_gas * R_typical / G_kpc if v2_gas > 0 else 1e8
        M_star_est = (v2_disk + v2_bul) * R_typical / G_kpc if (v2_disk + v2_bul) > 0 else 1e9
        
        # Estimate disk scale length from where v_disk peaks
        if np.any(df["vdisk"] > 0):
            R_d = df["rad"].iloc[np.argmax(df["vdisk"])] / 2.2  # Peak at ~2.2 R_d
        else:
            R_d = 2.0  # Default
            
        return {
            'name': name,
            'data': df,
            'M_gas_est': M_gas_est,
            'M_star_est': M_star_est,
            'R_d': R_d
        }
        
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def calculate_derived_quantities(galaxy_data):
    """Calculate T_dyn(r), true f_gas, surface brightness, etc."""
    df = galaxy_data['data']
    
    # Get radii and velocities
    r = df['rad'].values  # kpc
    v_obs = df['vobs'].values
    v_gas = df['vgas'].values
    v_disk = df['vdisk'].values
    v_bul = df['vbul'].values
    
    # Total baryonic velocity
    v_baryon = np.sqrt(v_gas**2 + v_disk**2 + v_bul**2)
    
    # Dynamical time in years
    T_dyn = 2 * np.pi * r * kpc_to_m / (v_baryon * 1000) / (365.25 * 24 * 3600)
    T_dyn = np.where(v_baryon > 0, T_dyn, 1e10)  # Avoid div by zero
    
    # Estimate H2
    M_HI = galaxy_data['M_gas_est'] 
    M_star = galaxy_data['M_star_est']
    M_H2 = estimate_H2_mass(M_HI, M_star)
    
    # True gas fraction
    M_gas_total = M_HI + M_H2
    f_gas_true = M_gas_total / (M_gas_total + M_star)
    
    # Surface brightness proxy (central surface density)
    R_d = galaxy_data['R_d']
    Sigma_0 = M_star / (2 * np.pi * R_d**2)  # M_sun/kpc²
    
    # Store everything
    galaxy_data['r'] = r
    galaxy_data['v_obs'] = v_obs
    galaxy_data['v_baryon'] = v_baryon
    galaxy_data['T_dyn'] = T_dyn
    galaxy_data['M_HI'] = M_HI
    galaxy_data['M_H2'] = M_H2
    galaxy_data['f_gas_true'] = f_gas_true
    galaxy_data['Sigma_0'] = Sigma_0
    
    return galaxy_data

def build_master_table():
    """Build comprehensive master table for all SPARC galaxies"""
    
    rotmod_dir = Path("Rotmod_LTG")
    if not rotmod_dir.exists():
        print(f"Error: {rotmod_dir} not found!")
        return None
        
    # Get all rotation curve files
    files = sorted(rotmod_dir.glob("*_rotmod.dat"))
    # Remove duplicates (files with ' 2' in name)
    files = [f for f in files if ' 2' not in str(f)]
    
    print(f"Building master table for {len(files)} galaxies...")
    
    master_table = {}
    successful = 0
    
    for i, file in enumerate(files):
        if i % 20 == 0:
            print(f"  Processing galaxy {i+1}/{len(files)}...")
            
        galaxy_data = load_rotation_curve(file)
        
        if galaxy_data is not None:
            # Calculate derived quantities
            galaxy_data = calculate_derived_quantities(galaxy_data)
            
            # Store in master table
            master_table[galaxy_data['name']] = galaxy_data
            successful += 1
    
    print(f"\nSuccessfully processed {successful} galaxies")
    
    # Save to pickle
    with open('sparc_master.pkl', 'wb') as f:
        pickle.dump(master_table, f)
        
    print("Saved master table to sparc_master.pkl")
    
    # Print some statistics
    print("\nMaster table statistics:")
    f_gas_values = [g['f_gas_true'] for g in master_table.values()]
    Sigma_0_values = [g['Sigma_0'] for g in master_table.values()]
    
    print(f"  f_gas range: {np.min(f_gas_values):.3f} - {np.max(f_gas_values):.3f}")
    print(f"  Mean f_gas: {np.mean(f_gas_values):.3f}")
    print(f"  Sigma_0 range: {np.min(Sigma_0_values):.2e} - {np.max(Sigma_0_values):.2e} M_sun/kpc²")
    
    return master_table

if __name__ == "__main__":
    master_table = build_master_table() 