#!/usr/bin/env python3
"""
LNAL Pure Formula
=================
The immutable gravitational law with zero free parameters.
All galaxy-specific information enters only through Σ(r).
"""

import numpy as np

# Universal constants
G = 6.67430e-11  # m³/kg/s²
G_DAGGER = 1.2e-10  # m/s² (MOND acceleration scale)


def lnal_acceleration(r, surface_density):
    """
    Pure LNAL gravitational acceleration.
    
    Parameters:
    -----------
    r : array_like
        Radius [m]
    surface_density : callable
        Function returning Σ(r) [kg/m²]
    
    Returns:
    --------
    g : array_like
        Total acceleration [m/s²]
    """
    # Step 1: Newtonian surface acceleration
    Sigma = surface_density(r)
    g_N = 2 * np.pi * G * Sigma
    
    # Step 2: Dimensionless ratio
    x = g_N / G_DAGGER
    
    # Step 3: LNAL modifier (pure theory, no parameters)
    mu = x / np.sqrt(1 + x**2)
    
    # Step 4: Total acceleration
    g_LNAL = g_N / mu
    
    return g_LNAL


def lnal_circular_velocity(r, surface_density):
    """
    Circular velocity from pure LNAL theory.
    
    Parameters:
    -----------
    r : array_like
        Radius [m]
    surface_density : callable
        Function returning Σ(r) [kg/m²]
    
    Returns:
    --------
    v : array_like
        Circular velocity [m/s]
    """
    g = lnal_acceleration(r, surface_density)
    return np.sqrt(r * g)


# Example usage
if __name__ == "__main__":
    # Test with exponential disk
    kpc = 3.0856775814913673e19  # m
    M_sun = 1.98847e30  # kg
    
    M_disk = 5e10 * M_sun
    R_d = 3 * kpc
    
    def exponential_disk(r):
        """Example: pure exponential disk"""
        Sigma_0 = M_disk / (2 * np.pi * R_d**2)
        return Sigma_0 * np.exp(-r / R_d)
    
    # Compute velocity curve
    r = np.logspace(np.log10(0.1 * kpc), np.log10(30 * kpc), 100)
    v = lnal_circular_velocity(r, exponential_disk)
    
    print("LNAL Pure Formula Test")
    print(f"Disk mass: {M_disk/M_sun:.2e} M_sun")
    print(f"Scale length: {R_d/kpc:.1f} kpc")
    print(f"V(10 kpc): {v[r > 10*kpc][0]/1000:.1f} km/s")
    print(f"V_asymptotic: {np.mean(v[-10:])/1000:.1f} km/s") 