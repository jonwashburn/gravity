#!/usr/bin/env python3
"""
Simple SPARC parser - extracts key galaxy properties
"""

import numpy as np
import re

def parse_sparc_simple(filename="SPARC_Lelli2016c.mrt"):
    """Extract galaxy properties from SPARC file"""
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header and find data start
    data_start = 0
    for i, line in enumerate(lines):
        if '-------' in line and i > 80:
            data_start = i + 1
            break
    
    galaxies = []
    
    # Process each line
    for line in lines[data_start:]:
        line = line.strip()
        if not line or 'Note:' in line:
            continue
            
        # Split by whitespace
        parts = line.split()
        if len(parts) < 17:
            continue
            
        try:
            # Extract key properties
            galaxy = {
                'name': parts[0],
                'T': float(parts[1]),
                'D': float(parts[2]),  # Distance in Mpc
                'Inc': float(parts[5]),  # Inclination
                'L36': float(parts[7]),  # L[3.6] in 10^9 L_sun
                'Rdisk': float(parts[11]) if parts[11] != '0.00' else None,  # kpc
                'MHI': float(parts[13]) if parts[13] != '0.000' else None,  # 10^9 M_sun
                'Vflat': float(parts[15]),  # km/s
                'Q': int(parts[17])  # Quality flag
            }
            
            # Only keep galaxies with good data
            if galaxy['Vflat'] > 0 and galaxy['L36'] > 0:
                galaxies.append(galaxy)
                
        except (ValueError, IndexError):
            continue
    
    return galaxies

def test_parser():
    """Test the parser and show statistics"""
    galaxies = parse_sparc_simple()
    
    print(f"Parsed {len(galaxies)} galaxies")
    print("\nSample galaxies:")
    
    # Show some specific galaxies we've been analyzing
    test_names = ['NGC2403', 'NGC3198', 'NGC6503', 'DDO154', 'UGC2885']
    
    for name in test_names:
        for gal in galaxies:
            if gal['name'] == name:
                print(f"{name:10s}: L36={gal['L36']:6.3f}, "
                      f"Rdisk={gal['Rdisk']:5.2f}, " 
                      f"Vflat={gal['Vflat']:6.1f}, "
                      f"Q={gal['Q']}")
                break
    
    # Statistics
    print(f"\nTotal with Rdisk: {sum(1 for g in galaxies if g['Rdisk'] is not None)}")
    print(f"High quality (Q>=2): {sum(1 for g in galaxies if g['Q'] >= 2)}")
    
    return galaxies

if __name__ == "__main__":
    test_parser() 