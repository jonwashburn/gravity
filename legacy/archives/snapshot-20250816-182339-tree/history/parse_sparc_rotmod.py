#!/usr/bin/env python3
"""
Parse real SPARC rotation curve files from Rotmod_LTG directory
"""

import numpy as np
import os
import glob
from parse_sparc_mrt import parse_sparc_mrt

def parse_rotmod_file(filepath):
    """
    Parse a single SPARC _rotmod.dat file
    Returns dict with radius, velocities, errors, and baryonic components
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

def load_all_sparc_curves(rotmod_dir='Rotmod_LTG', catalog_file='SPARC_Lelli2016c.mrt'):
    """
    Load all SPARC rotation curves and match with catalog data
    """
    # Parse catalog
    catalog = parse_sparc_mrt(catalog_file)
    catalog_dict = {gal['name']: gal for gal in catalog}
    
    # Find all rotmod files
    pattern = os.path.join(rotmod_dir, '*_rotmod.dat')
    rotmod_files = glob.glob(pattern)
    
    print(f"Found {len(rotmod_files)} rotation curve files")
    
    combined_data = {}
    matched = 0
    
    for filepath in rotmod_files:
        # Extract galaxy name from filename
        filename = os.path.basename(filepath)
        galaxy_name = filename.replace('_rotmod.dat', '')
        
        # Parse rotation curve
        curve_data = parse_rotmod_file(filepath)
        if curve_data is None:
            continue
            
        # Match with catalog
        if galaxy_name in catalog_dict:
            combined_data[galaxy_name] = {
                'catalog': catalog_dict[galaxy_name],
                'curve': curve_data
            }
            matched += 1
        else:
            print(f"Warning: {galaxy_name} not found in catalog")
    
    print(f"Successfully matched {matched} galaxies")
    return combined_data

def save_sparc_data(data, output_file='sparc_real_data.pkl'):
    """Save combined SPARC data"""
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved real SPARC data to {output_file}")

if __name__ == "__main__":
    # Load and save real SPARC data
    data = load_all_sparc_curves()
    save_sparc_data(data)
    
    # Print example
    if data:
        first_name = list(data.keys())[0]
        first_data = data[first_name]
        curve = first_data['curve']
        catalog = first_data['catalog']
        
        print(f"\nExample: {first_name}")
        print(f"Distance: {curve['distance']:.2f} Mpc")
        print(f"V_flat (catalog): {catalog['V_flat']:.1f} km/s")
        print(f"Curve points: {len(curve['r'])}")
        print(f"R range: {curve['r'][0]:.2f} - {curve['r'][-1]:.2f} kpc")
        print(f"V range: {curve['V_obs'].min():.1f} - {curve['V_obs'].max():.1f} km/s") 