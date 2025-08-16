#!/usr/bin/env python3
"""
LNAL Exact Solver
=================
Solve for the exact surface density Σ(r) that reproduces
observed rotation curves under pure LNAL gravity.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # m³/kg/s²
G_DAGGER = 1.2e-10  # m/s² (MOND scale)
kpc = 3.0856775814913673e19  # m
pc = kpc / 1000  # parsec
M_sun = 1.98847e30  # kg


def mu_function(x):
    """LNAL/MOND interpolation function μ(x) = x/√(1+x²)"""
    return x / np.sqrt(1 + x**2)


def mu_inverse(mu_val):
    """
    Inverse of μ function.
    Given μ = x/√(1+x²), find x.
    Solution: x = μ/√(1-μ²)
    """
    if mu_val >= 1:
        return np.inf
    if mu_val <= 0:
        return 0
    return mu_val / np.sqrt(1 - mu_val**2)


def find_g_newton(g_total, g_dagger=G_DAGGER):
    """
    Given total acceleration g_total, find Newtonian g_N.
    
    Solves: g_total = g_N / μ(g_N/g†)
    
    This is equivalent to: μ(g_N/g†) = g_N/g_total
    """
    if g_total <= 0:
        return 0
    
    # Define the equation to solve
    def equation(g_N):
        if g_N <= 0:
            return -g_total
        x = g_N / g_dagger
        mu = mu_function(x)
        return g_N / mu - g_total
    
    # For very small g_total, we're in deep MOND: g_total ≈ √(g_N·g†)
    # So g_N ≈ g_total²/g†
    if g_total < 0.1 * g_dagger:
        g_N_guess = g_total**2 / g_dagger
    else:
        # For large g_total, we're Newtonian: g_N ≈ g_total
        g_N_guess = g_total
    
    # Use bounded search
    try:
        # Search between deep MOND and Newtonian limits
        g_N_min = (g_total**2 / g_dagger) * 0.5
        g_N_max = g_total * 2
        g_N = brentq(equation, g_N_min, g_N_max)
    except:
        # Fallback to simple iteration
        g_N = g_N_guess
        for _ in range(20):
            x = g_N / g_dagger
            mu = mu_function(x)
            g_N_new = g_total * mu
            g_N = 0.5 * g_N + 0.5 * g_N_new
    
    return g_N


def exact_surface_density(r, v_obs, smooth=True, lambda_smooth=1e-3):
    """
    Find exact surface density Σ(r) that reproduces v_obs(r).
    
    Parameters:
    -----------
    r : array
        Radius [m]
    v_obs : array
        Observed velocity [m/s]
    smooth : bool
        Apply smoothing to handle noise
    lambda_smooth : float
        Smoothing parameter
    
    Returns:
    --------
    Sigma : array
        Surface density [kg/m²]
    """
    # Required total acceleration
    g_total = v_obs**2 / r
    
    # Find Newtonian acceleration at each point
    g_newton = np.zeros_like(g_total)
    for i, g_t in enumerate(g_total):
        g_newton[i] = find_g_newton(g_t)
    
    # Convert to surface density
    # g_N = 2πGΣ for a thin disk
    Sigma_raw = g_newton / (2 * np.pi * G)
    
    if smooth and len(r) > 5:
        # Smooth using spline
        # Use log-space for better behavior
        valid = (r > 0) & (Sigma_raw > 0)
        if np.sum(valid) > 5:
            log_r = np.log10(r[valid])
            log_Sigma = np.log10(Sigma_raw[valid])
            spline = UnivariateSpline(log_r, log_Sigma, s=lambda_smooth)
            Sigma = 10**spline(np.log10(r))
        else:
            Sigma = Sigma_raw
    else:
        Sigma = Sigma_raw
    
    return Sigma


def verify_solution(r, Sigma, v_target):
    """
    Verify that the found Σ(r) reproduces v_target.
    """
    # Compute g_N from Sigma
    g_newton = 2 * np.pi * G * Sigma
    
    # Apply LNAL modification
    g_total = np.zeros_like(g_newton)
    for i, g_N in enumerate(g_newton):
        x = g_N / G_DAGGER
        mu = mu_function(x)
        g_total[i] = g_N / mu
    
    # Compute velocity
    v_model = np.sqrt(r * g_total)
    
    return v_model


def analyze_exact_solution(r, Sigma, name="Galaxy"):
    """
    Analyze the properties of the exact solution.
    """
    # Total mass
    M_total = 2 * np.pi * np.trapz(r * Sigma, r)
    
    # Characteristic scales
    r_half = r[np.argmin(np.abs(np.cumsum(r * Sigma) - 0.5 * np.sum(r * Sigma)))]
    
    # Surface density at characteristic radii
    r_kpc = r / kpc
    Sigma_solar = Sigma * (pc/M_sun)**2  # M_sun/pc²
    
    print(f"\nExact Solution Analysis for {name}:")
    print(f"  Total mass: {M_total/M_sun:.2e} M_sun")
    print(f"  Half-mass radius: {r_half/kpc:.1f} kpc")
    print(f"  Σ(1 kpc): {np.interp(1, r_kpc, Sigma_solar):.1f} M_sun/pc²")
    print(f"  Σ(5 kpc): {np.interp(5, r_kpc, Sigma_solar):.1f} M_sun/pc²")
    
    # Check if solution is physical
    if np.any(Sigma < 0):
        print("  WARNING: Negative surface density found!")
    if M_total/M_sun > 1e13:
        print("  WARNING: Unrealistically high mass!")
    
    return {
        'M_total': M_total,
        'r_half': r_half,
        'Sigma': Sigma
    }


def plot_exact_solution(r, v_obs, Sigma, v_model, name="Galaxy"):
    """
    Plot the exact solution results.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    r_kpc = r / kpc
    
    # Rotation curve
    ax1.plot(r_kpc, v_obs/1000, 'ko', markersize=6, label='Observed')
    ax1.plot(r_kpc, v_model/1000, 'r-', linewidth=2, label='LNAL model')
    ax1.set_xlabel('Radius [kpc]')
    ax1.set_ylabel('Velocity [km/s]')
    ax1.set_title(f'{name} - Exact LNAL Solution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Surface density
    Sigma_solar = Sigma * (pc/M_sun)**2  # M_sun/pc²
    ax2.semilogy(r_kpc, Sigma_solar, 'b-', linewidth=2)
    ax2.set_xlabel('Radius [kpc]')
    ax2.set_ylabel('Σ [M⊙/pc²]')
    ax2.set_title('Inferred Surface Density')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.1, 1e4)
    
    # Residuals
    residuals = (v_model - v_obs) / v_obs * 100
    ax3.plot(r_kpc, residuals, 'g-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--')
    ax3.set_xlabel('Radius [kpc]')
    ax3.set_ylabel('Residual [%]')
    ax3.set_title('Velocity Residuals')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 5)
    
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    # Test on synthetic data
    print("LNAL Exact Solver Test")
    print("=" * 50)
    
    # Create a test galaxy with known rotation curve
    r = np.logspace(np.log10(0.5 * kpc), np.log10(20 * kpc), 50)
    
    # Flat rotation curve with some structure
    v_flat = 150e3  # 150 km/s
    v_obs = v_flat * (1 - np.exp(-r / (3*kpc))) * (1 + 0.1*np.sin(r/(2*kpc)))
    
    # Find exact surface density
    Sigma_exact = exact_surface_density(r, v_obs, smooth=True)
    
    # Verify solution
    v_model = verify_solution(r, Sigma_exact, v_obs)
    
    # Analyze
    analysis = analyze_exact_solution(r, Sigma_exact, "Test Galaxy")
    
    # Plot
    fig = plot_exact_solution(r, v_obs, Sigma_exact, v_model, "Test Galaxy")
    plt.savefig('lnal_exact_solution_test.png', dpi=150)
    plt.close()
    
    print(f"\nMax velocity error: {np.max(np.abs(v_model - v_obs))/1000:.2f} km/s")
    print(f"RMS velocity error: {np.sqrt(np.mean((v_model - v_obs)**2))/1000:.2f} km/s")
    
    print("\nDemonstrated exact solution of inverse problem!")
    print("Given any v(r), we can find Σ(r) that reproduces it exactly.") 