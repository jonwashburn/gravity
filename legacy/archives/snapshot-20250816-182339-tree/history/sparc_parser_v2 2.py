#!/usr/bin/env python3
"""
SPARC Data Parser V2
Robust parser for SPARC_Lelli2016c.mrt file with proper handling of wrapped lines
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple
import pickle

def parse_sparc_header(filename: str) -> Tuple[int, List[str]]:
    """Parse header to find data start and column names"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find where data starts - look for the line of dashes before the data
    data_start = 0
    for i, line in enumerate(lines):
        if i > 0 and '-------' in line and len(line) > 50:
            # The actual data starts after the line of dashes
            data_start = i + 1
            break
    
    # Extract column names from the header
    # The table format starts around line 88 based on the file
    col_names = ['Galaxy', 'T', 'D', 'e_D', 'f_D', 'Inc', 'e_Inc', 
                 'L[3.6]', 'e_L[3.6]', 'Reff', 'SBeff', 'Rdisk', 'SBdisk',
                 'MHI', 'RHI', 'Vflat', 'e_Vflat', 'Q', 'Ref']
    
    return data_start, col_names

def parse_sparc_galaxy_block(lines: List[str]) -> Dict:
    """Parse a single galaxy block (handling multi-line entries)"""
    if not lines:
        return None
    
    # First line contains galaxy name and basic properties
    first_line = lines[0].strip()
    parts = first_line.split()
    
    if len(parts) < 10:
        return None
    
    galaxy = {
        'name': parts[0],
        'T': float(parts[1]),
        'D': float(parts[2]),
        'e_D': float(parts[3]),
        'f_D': float(parts[4]),
        'Inc': float(parts[5]),
        'e_Inc': float(parts[6]),
        'L36': float(parts[7]),
        'e_L36': float(parts[8]),
        'Reff': float(parts[9]) if len(parts) > 9 else np.nan,
        'SBeff': float(parts[10]) if len(parts) > 10 else np.nan,
        'Rdisk': float(parts[11]) if len(parts) > 11 else np.nan,
        'SBdisk': float(parts[12]) if len(parts) > 12 else np.nan,
        'MHI': float(parts[13]) if len(parts) > 13 else np.nan,
        'RHI': float(parts[14]) if len(parts) > 14 else np.nan,
        'Vflat': float(parts[15]) if len(parts) > 15 else np.nan,
        'e_Vflat': float(parts[16]) if len(parts) > 16 else np.nan,
        'Q': float(parts[17]) if len(parts) > 17 else np.nan,
        'Ref': parts[18] if len(parts) > 18 else ""
    }
    
    # Parse radial data from subsequent lines
    radii = []
    vobs = []
    e_vobs = []
    vgas = []
    vdisk = []
    vbul = []
    SBdisk_r = []
    SBbul_r = []
    
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a continuation of references
        if not any(char.isdigit() for char in line[:10]):
            galaxy['Ref'] += " " + line
            continue
        
        # Parse radial data
        parts = line.split()
        if len(parts) >= 6:
            try:
                radii.append(float(parts[0]))
                vobs.append(float(parts[1]))
                e_vobs.append(float(parts[2]))
                vgas.append(float(parts[3]))
                vdisk.append(float(parts[4]))
                vbul.append(float(parts[5]))
                if len(parts) > 6:
                    SBdisk_r.append(float(parts[6]))
                if len(parts) > 7:
                    SBbul_r.append(float(parts[7]))
            except ValueError:
                # Skip malformed lines
                continue
    
    galaxy['radii'] = np.array(radii)
    galaxy['vobs'] = np.array(vobs)
    galaxy['e_vobs'] = np.array(e_vobs)
    galaxy['vgas'] = np.array(vgas)
    galaxy['vdisk'] = np.array(vdisk)
    galaxy['vbul'] = np.array(vbul)
    galaxy['SBdisk_r'] = np.array(SBdisk_r) if SBdisk_r else None
    galaxy['SBbul_r'] = np.array(SBbul_r) if SBbul_r else None
    
    return galaxy

def parse_sparc_file(filename: str) -> List[Dict]:
    """Parse entire SPARC file"""
    data_start, col_names = parse_sparc_header(filename)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Process data lines
    data_lines = lines[data_start:]
    galaxies = []
    current_block = []
    
    for line in data_lines:
        line = line.strip()
        
        # Check if this is a new galaxy (starts with a letter)
        if line and line[0].isalpha() and not line.startswith('Note:'):
            # Process previous block if exists
            if current_block:
                galaxy = parse_sparc_galaxy_block(current_block)
                if galaxy and len(galaxy['radii']) > 5:  # Quality filter
                    galaxies.append(galaxy)
            
            # Start new block
            current_block = [line]
        elif line and current_block:
            # Add to current block
            current_block.append(line)
    
    # Don't forget last block
    if current_block:
        galaxy = parse_sparc_galaxy_block(current_block)
        if galaxy and len(galaxy['radii']) > 5:
            galaxies.append(galaxy)
    
    return galaxies

def compute_surface_densities(galaxy: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute surface density profiles from rotation curve decomposition
    
    Returns:
        radii: radius array in kpc
        Sigma_star: stellar surface density in M_sun/pc^2
        Sigma_gas: gas surface density in M_sun/pc^2
    """
    radii = galaxy['radii']  # kpc
    vdisk = galaxy['vdisk']  # km/s
    vgas = galaxy['vgas']   # km/s
    vbul = galaxy['vbul']   # km/s
    
    # Constants
    G = 4.302e-6  # kpc (km/s)^2 / M_sun
    
    # Stellar surface density from disk+bulge rotation curve
    # Using V^2 = G * M_enc / r and M_enc = 2π ∫ Σ(r') r' dr'
    # For exponential disk: Σ = Σ_0 exp(-r/Rd)
    
    # Total stellar
    v_star_sq = vdisk**2 + vbul**2
    
    # Approximate surface density (assumes exponential profile)
    if galaxy['Rdisk'] > 0:
        Rd = galaxy['Rdisk']  # kpc
        # Normalize using total luminosity
        M_star_total = 0.5 * galaxy['L36'] * 1e9  # M_sun (M/L = 0.5)
        Sigma_0 = M_star_total / (2 * np.pi * Rd**2)  # M_sun/kpc^2
        Sigma_star = Sigma_0 * np.exp(-radii / Rd) * 1e-6  # M_sun/pc^2
    else:
        Sigma_star = np.zeros_like(radii)
    
    # Gas surface density from gas rotation curve
    # Use isothermal approximation
    if galaxy['MHI'] > 0 and galaxy['RHI'] > 0:
        M_gas = galaxy['MHI'] * 1e9  # M_sun
        R_gas = galaxy['RHI']  # kpc
        Sigma_gas_0 = M_gas / (2 * np.pi * R_gas**2)  # M_sun/kpc^2
        Sigma_gas = Sigma_gas_0 * np.exp(-radii / R_gas) * 1e-6  # M_sun/pc^2
    else:
        # Estimate from rotation curve
        Sigma_gas = 0.2 * Sigma_star  # Default 20% gas fraction
    
    return radii, Sigma_star, Sigma_gas

def save_sparc_data(galaxies: List[Dict], output_file: str = "sparc_parsed_data.pkl"):
    """Save parsed data for quick loading"""
    with open(output_file, 'wb') as f:
        pickle.dump(galaxies, f)
    print(f"Saved {len(galaxies)} galaxies to {output_file}")

def load_sparc_data(input_file: str = "sparc_parsed_data.pkl") -> List[Dict]:
    """Load pre-parsed data"""
    with open(input_file, 'rb') as f:
        return pickle.load(f)

def main():
    """Parse SPARC data and show summary"""
    print("Parsing SPARC data...")
    
    try:
        galaxies = parse_sparc_file("SPARC_Lelli2016c.mrt")
        print(f"Successfully parsed {len(galaxies)} galaxies")
        
        # Save for future use
        save_sparc_data(galaxies)
        
        # Show sample
        print("\nSample galaxies:")
        for i, gal in enumerate(galaxies[:5]):
            print(f"{i+1}. {gal['name']:12s} - "
                  f"L[3.6]={gal['L36']:.3f}×10⁹ L_sun, "
                  f"Vflat={gal['Vflat']:.1f} km/s, "
                  f"N_points={len(gal['radii'])}")
        
        # Statistics
        print(f"\nStatistics:")
        print(f"Total galaxies: {len(galaxies)}")
        print(f"With disk scale: {sum(1 for g in galaxies if g['Rdisk'] > 0)}")
        print(f"With HI data: {sum(1 for g in galaxies if g['MHI'] > 0)}")
        print(f"High quality (Q≥2): {sum(1 for g in galaxies if g['Q'] >= 2)}")
        
    except FileNotFoundError:
        print("Error: SPARC_Lelli2016c.mrt not found!")
        print("Please download from: http://astroweb.cwru.edu/SPARC/")

if __name__ == "__main__":
    main() 