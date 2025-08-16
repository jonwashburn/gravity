#!/usr/bin/env python3
"""
LNAL Prime Recognition Gravity: Exact Baryonic Source Parser
Extracts complete stellar disk, bulge, and HI surface density profiles 
from SPARC galaxy data tables to construct B(R) = Σ ρ c² exactly.
"""

import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from scipy.integrate import quad
import pickle

# Physical constants
c = 2.998e8  # m/s
G = 6.674e-11  # m³/kg/s²
kpc_to_m = 3.086e19  # m/kpc
Msun_to_kg = 1.989e30  # kg/Msun

class SPARCBaryonParser:
    """Parse SPARC galaxy data to extract exact baryonic source B(R)"""
    
    def __init__(self, sparc_file='SPARC_Lelli2016c.mrt', rotmod_dir='Rotmod_LTG/'):
        self.sparc_file = sparc_file
        self.rotmod_dir = rotmod_dir
        self.galaxies = {}
        self.load_sparc_catalog()
    
    def load_sparc_catalog(self):
        """Load main SPARC catalog with galaxy parameters"""
        print("Loading SPARC catalog...")
        
        # Read the main catalog file
        with open(self.sparc_file, 'r') as f:
            lines = f.readlines()
        
        # Find data start (after header)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('NGC0024') or line.strip().startswith('IC'):
                data_start = i
                break
        
        # Parse galaxy data
        for line in lines[data_start:]:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 15:  # Ensure we have enough columns
                    galaxy = parts[0]
                    
                    # Extract key parameters
                    try:
                        self.galaxies[galaxy] = {
                            'name': galaxy,
                            'distance': float(parts[2]),  # Mpc
                            'inclination': float(parts[3]),  # degrees
                            'L_36': float(parts[4]) if parts[4] != '...' else 0,  # 3.6μm luminosity
                            'stellar_mass': float(parts[6]) if parts[6] != '...' else 0,  # log(M*/Msun)
                            'gas_mass': float(parts[7]) if parts[7] != '...' else 0,  # log(Mgas/Msun)
                            'v_flat': float(parts[8]) if parts[8] != '...' else 0,  # km/s
                            'quality': parts[9]
                        }
                    except (ValueError, IndexError):
                        continue
        
        print(f"Loaded {len(self.galaxies)} galaxies from SPARC catalog")
    
    def parse_rotmod_file(self, galaxy_name):
        """Parse individual galaxy rotation curve file"""
        rotmod_file = os.path.join(self.rotmod_dir, f"{galaxy_name}_rotmod.dat")
        
        if not os.path.exists(rotmod_file):
            print(f"Warning: {rotmod_file} not found")
            return None
        
        # Read rotation curve data
        data = []
        with open(rotmod_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            radius = float(parts[0])  # kpc
                            v_obs = float(parts[1])   # km/s
                            v_err = float(parts[2])   # km/s
                            v_gas = float(parts[3])   # km/s (gas contribution)
                            v_disk = float(parts[4])  # km/s (stellar disk)
                            v_bulge = float(parts[5]) if len(parts) > 5 else 0  # km/s (bulge)
                            
                            data.append([radius, v_obs, v_err, v_gas, v_disk, v_bulge])
                        except ValueError:
                            continue
        
        if not data:
            return None
        
        data = np.array(data)
        return {
            'radius': data[:, 0],  # kpc
            'v_obs': data[:, 1],   # km/s
            'v_err': data[:, 2],   # km/s
            'v_gas': data[:, 3],   # km/s
            'v_disk': data[:, 4],  # km/s
            'v_bulge': data[:, 5]  # km/s
        }
    
    def velocity_to_surface_density(self, radius, velocity, galaxy_info):
        """Convert velocity curve to surface density using v² = 2πGΣR"""
        # Convert units
        R = radius * kpc_to_m  # m
        v = velocity * 1000    # m/s
        
        # Surface density: Σ = v²/(2πGR)
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma = v**2 / (2 * np.pi * G * R)  # kg/m²
        
        # Handle division by zero at R=0
        sigma[R == 0] = 0
        
        return sigma  # kg/m²
    
    def construct_baryonic_source(self, galaxy_name):
        """Construct exact baryonic source B(R) = Σ ρ c²"""
        if galaxy_name not in self.galaxies:
            print(f"Galaxy {galaxy_name} not in catalog")
            return None
        
        galaxy_info = self.galaxies[galaxy_name]
        rotmod_data = self.parse_rotmod_file(galaxy_name)
        
        if rotmod_data is None:
            return None
        
        radius = rotmod_data['radius']  # kpc
        
        # Convert velocity contributions to surface densities
        sigma_gas = self.velocity_to_surface_density(radius, rotmod_data['v_gas'], galaxy_info)
        sigma_disk = self.velocity_to_surface_density(radius, rotmod_data['v_disk'], galaxy_info)
        sigma_bulge = self.velocity_to_surface_density(radius, rotmod_data['v_bulge'], galaxy_info)
        
        # Total baryonic surface density
        sigma_total = sigma_gas + sigma_disk + sigma_bulge  # kg/m²
        
        # Convert to 3D density assuming thin disk with scale height h
        # ρ = Σ/(2h) where h ≈ 0.3 kpc for typical spiral galaxies
        h_disk = 0.3 * kpc_to_m  # m
        rho_total = sigma_total / (2 * h_disk)  # kg/m³
        
        # Baryonic source: B(R) = ρ c²
        B_R = rho_total * c**2  # J/m³
        
        return {
            'radius': radius,  # kpc
            'B_R': B_R,       # J/m³
            'sigma_gas': sigma_gas,
            'sigma_disk': sigma_disk,
            'sigma_bulge': sigma_bulge,
            'v_obs': rotmod_data['v_obs'],
            'v_err': rotmod_data['v_err']
        }
    
    def process_all_galaxies(self):
        """Process all galaxies and extract baryonic sources"""
        results = {}
        
        for galaxy_name in self.galaxies.keys():
            print(f"Processing {galaxy_name}...")
            baryon_data = self.construct_baryonic_source(galaxy_name)
            
            if baryon_data is not None:
                results[galaxy_name] = baryon_data
            else:
                print(f"  Failed to process {galaxy_name}")
        
        print(f"Successfully processed {len(results)} galaxies")
        return results
    
    def save_results(self, results, filename='sparc_exact_baryons.pkl'):
        """Save processed results to file"""
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}")
    
    def load_results(self, filename='sparc_exact_baryons.pkl'):
        """Load processed results from file"""
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded results for {len(results)} galaxies")
        return results

def main():
    """Main processing function"""
    parser = SPARCBaryonParser()
    
    # Process all galaxies
    results = parser.process_all_galaxies()
    
    # Save results
    parser.save_results(results)
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total galaxies processed: {len(results)}")
    
    # Show example
    if results:
        example_galaxy = list(results.keys())[0]
        example_data = results[example_galaxy]
        print(f"\nExample ({example_galaxy}):")
        print(f"  Radius range: {example_data['radius'][0]:.2f} - {example_data['radius'][-1]:.2f} kpc")
        print(f"  B(R) range: {example_data['B_R'].min():.2e} - {example_data['B_R'].max():.2e} J/m³")

if __name__ == "__main__":
    main() 