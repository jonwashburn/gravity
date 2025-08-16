#!/usr/bin/env python3
"""
Parse SPARC data from MRT (Machine Readable Table) format
"""

import numpy as np

def parse_sparc_mrt(filename='SPARC_Lelli2016c.mrt'):
    """Parse SPARC MRT file and return list of galaxy dictionaries"""
    
    galaxies = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find where data starts (after last line of dashes)
    data_start = 0
    for i, line in enumerate(lines):
        if '-------' in line:
            data_start = i + 1
    
    print(f"Data starts at line {data_start}, total lines: {len(lines)}")
    
    # Parse each galaxy
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
            
        if i == data_start:  # Debug first line
            print(f"First data line: {repr(line[:100])}")
            
        try:
            # Split by whitespace - simpler approach
            parts = line.split()
            
            # Need at least 18 fields
            if len(parts) < 18:
                continue
            
            name = parts[0]
            T = int(parts[1])  # Hubble type
            D = float(parts[2])  # Distance in Mpc
            e_D = float(parts[3])  # Distance error
            f_D = int(parts[4])  # Distance method
            Inc = float(parts[5])  # Inclination
            e_Inc = float(parts[6])  # Inclination error
            L_36 = float(parts[7])  # Luminosity at 3.6μm (10^9 L_sun)
            e_L_36 = float(parts[8])  # Luminosity error
            R_eff = float(parts[9])  # Effective radius (kpc)
            SB_eff = float(parts[10])  # Effective surface brightness
            R_disk = float(parts[11])  # Disk scale length (kpc)
            SB_disk = float(parts[12])  # Disk central surface brightness
            M_HI = float(parts[13])  # HI mass (10^9 M_sun)
            R_HI = float(parts[14])  # HI radius (kpc)
            V_flat = float(parts[15])  # Flat rotation velocity (km/s)
            e_V_flat = float(parts[16])  # Velocity error
            Q = int(parts[17])  # Quality flag
            
            # Skip galaxies with no rotation curve
            if V_flat == 0.0:
                continue
                
            # Calculate stellar mass from 3.6μm luminosity
            # Using M*/L = 0.6 (McGaugh & Schombert 2014)
            M_star = 0.6 * L_36  # 10^9 M_sun
            
            galaxy = {
                'name': name,
                'type': T,
                'distance': D,  # Mpc
                'inclination': Inc,  # degrees
                'L_36': L_36,  # 10^9 L_sun
                'R_disk': R_disk,  # kpc
                'M_star': M_star,  # 10^9 M_sun
                'M_HI': M_HI,  # 10^9 M_sun
                'M_visible': M_star + M_HI,  # 10^9 M_sun
                'V_flat': V_flat,  # km/s
                'e_V_flat': e_V_flat,  # km/s
                'quality': Q
            }
            
            galaxies.append(galaxy)
            
        except Exception as e:
            # Skip lines that can't be parsed
            if i < data_start + 3:  # Debug first few lines
                print(f"Error on line {i}: {e}")
                print(f"Line content: {repr(line)}")
            continue
    
    return galaxies

def print_summary(galaxies):
    """Print summary statistics"""
    print(f"\nParsed {len(galaxies)} galaxies with rotation curves")
    
    # Quality distribution
    q1 = sum(1 for g in galaxies if g['quality'] == 1)
    q2 = sum(1 for g in galaxies if g['quality'] == 2)
    q3 = sum(1 for g in galaxies if g['quality'] == 3)
    print(f"Quality 1: {q1}, Quality 2: {q2}, Quality 3: {q3}")
    
    # Mass range
    masses = [g['M_visible'] for g in galaxies]
    print(f"Mass range: {min(masses):.1e} - {max(masses):.1e} × 10^9 M_sun")
    
    # Velocity range
    velocities = [g['V_flat'] for g in galaxies]
    print(f"V_flat range: {min(velocities):.1f} - {max(velocities):.1f} km/s")

if __name__ == "__main__":
    galaxies = parse_sparc_mrt()
    print_summary(galaxies)
    
    # Show a few examples
    print("\nFirst 5 galaxies:")
    for i, gal in enumerate(galaxies[:5]):
        print(f"{gal['name']:12} M_vis={gal['M_visible']:7.1f}×10^9 M_sun  "
              f"R_disk={gal['R_disk']:5.2f} kpc  V_flat={gal['V_flat']:6.1f} km/s  Q={gal['quality']}") 