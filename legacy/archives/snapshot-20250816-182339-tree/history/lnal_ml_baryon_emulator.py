#!/usr/bin/env python3
"""
LNAL ML Baryon Emulator
========================
Use machine learning to emulate realistic baryon distributions
from limited observables.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
import json
import os

# Constants
G = 6.67430e-11  # m³/kg/s²
G_DAGGER = 1.2e-10  # m/s² (MOND scale)
kpc = 3.0856775814913673e19  # m
pc = kpc / 1000
M_sun = 1.98847e30  # kg


def load_sparc_rotmod(galaxy_name, rotmod_dir='Rotmod_LTG'):
    """Load SPARC rotation curve data."""
    filepath = os.path.join(rotmod_dir, f'{galaxy_name}_rotmod.dat')
    
    if not os.path.exists(filepath):
        return None
    
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 8:
                data.append([float(p) for p in parts[:8]])
    
    if not data:
        return None
    
    data = np.array(data)
    return {
        'r': data[:, 0] * kpc,
        'v_obs': data[:, 1] * 1000,
        'v_err': data[:, 2] * 1000,
        'v_gas': data[:, 3] * 1000,
        'v_disk': data[:, 4] * 1000,
        'v_bulge': data[:, 5] * 1000,
    }


def lnal_velocity(r, Sigma_total):
    """Compute LNAL velocity from surface density."""
    # Enclosed mass
    if len(r) > 1:
        M_enc = 2 * np.pi * cumulative_trapezoid(r * Sigma_total, r, initial=0)
    else:
        M_enc = np.zeros_like(r)
    
    # Newtonian acceleration
    g_newton = G * M_enc / r**2
    g_newton[0] = g_newton[1] if len(g_newton) > 1 else 0
    
    # LNAL modification
    x = g_newton / G_DAGGER
    mu = x / np.sqrt(1 + x**2)
    g_total = g_newton / mu
    
    # Velocity
    v = np.sqrt(r * g_total)
    return v


def surface_density_from_velocity(r, v_component):
    """Invert velocity to get surface density (approximate)."""
    # For thin disk: V² = G * M_enc / r
    # where M_enc = 2π ∫ Σ(r') r' dr'
    # This gives: Σ(r) ≈ (1/2πGr) * d(r V²)/dr
    
    v_component = np.maximum(v_component, 0)
    r = np.maximum(r, r[0] * 0.1)
    
    # Compute r * V²
    rv2 = r * v_component**2
    
    # Smooth derivative
    if len(r) > 3:
        drv2_dr = np.gradient(rv2, r)
    else:
        drv2_dr = np.zeros_like(r)
    
    # Surface density
    Sigma = drv2_dr / (2 * np.pi * G * r)
    Sigma = np.maximum(Sigma, 0)
    
    return Sigma


class BaryonEmulator:
    """ML emulator for baryon distributions."""
    
    def __init__(self):
        """Initialize emulator."""
        # Gaussian Process for learning surface density patterns
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.trained = False
        
    def extract_features(self, data):
        """Extract features from galaxy data."""
        r = data['r']
        v_obs = data['v_obs']
        
        # Global features
        v_max = np.max(v_obs)
        r_max = r[np.argmax(v_obs)]
        v_flat = np.median(v_obs[len(v_obs)//2:])
        
        # Shape features
        v_norm = v_obs / v_max
        r_norm = r / r_max
        
        # Gradient features
        if len(r) > 3:
            dv_dr = np.gradient(v_obs, r)
            max_gradient = np.max(np.abs(dv_dr))
        else:
            max_gradient = 0
        
        # Feature vector for each radius
        features = []
        for i in range(len(r)):
            feat = [
                r_norm[i],                    # Normalized radius
                v_norm[i],                    # Normalized velocity
                v_obs[i] / v_flat,           # Velocity relative to flat part
                r[i] / (r_max + 1e-10),     # Radius relative to peak
                np.log10(r[i] / kpc + 0.1), # Log radius
            ]
            features.append(feat)
        
        return np.array(features), {
            'v_max': v_max,
            'r_max': r_max,
            'v_flat': v_flat
        }
    
    def train_on_sparc(self, galaxy_names):
        """Train emulator on SPARC galaxies with known decompositions."""
        print("Training baryon emulator on SPARC data...")
        
        X_train = []
        y_train = []
        
        for name in galaxy_names:
            data = load_sparc_rotmod(name)
            if data is None:
                continue
            
            # Extract features
            features, meta = self.extract_features(data)
            
            # Get "true" surface densities from components
            r = data['r']
            Sigma_gas = surface_density_from_velocity(r, data['v_gas'])
            Sigma_disk = surface_density_from_velocity(r, data['v_disk'])
            Sigma_bulge = surface_density_from_velocity(r, data['v_bulge'])
            Sigma_total = Sigma_gas + Sigma_disk + Sigma_bulge
            
            # Log transform for better learning
            log_Sigma = np.log10(Sigma_total + 1e-10)
            
            X_train.extend(features)
            y_train.extend(log_Sigma)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Normalize features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # Train GP
        self.gp.fit(X_train_scaled, y_train_scaled)
        self.trained = True
        
        print(f"Trained on {len(galaxy_names)} galaxies with {len(X_train)} data points")
        
    def predict_surface_density(self, data):
        """Predict surface density for new galaxy."""
        if not self.trained:
            raise ValueError("Emulator not trained yet!")
        
        # Extract features
        features, meta = self.extract_features(data)
        
        # Scale features
        features_scaled = self.scaler_X.transform(features)
        
        # Predict log surface density
        log_Sigma_scaled, sigma = self.gp.predict(features_scaled, return_std=True)
        
        # Inverse transform
        log_Sigma = self.scaler_y.inverse_transform(log_Sigma_scaled.reshape(-1, 1)).ravel()
        Sigma_pred = 10**log_Sigma
        
        # Uncertainty in linear space (approximate)
        Sigma_upper = 10**(log_Sigma + sigma)
        Sigma_lower = 10**(log_Sigma - sigma)
        
        return Sigma_pred, Sigma_lower, Sigma_upper
    
    def fit_galaxy_ml(self, galaxy_name):
        """Fit galaxy using ML-predicted surface density."""
        # Load data
        data = load_sparc_rotmod(galaxy_name)
        if data is None:
            return None
        
        r = data['r']
        v_obs = data['v_obs']
        v_err = data['v_err']
        
        # Handle missing errors
        v_err[v_err <= 0] = 5000  # 5 km/s default
        
        # Predict surface density
        Sigma_pred, Sigma_lower, Sigma_upper = self.predict_surface_density(data)
        
        # Compute LNAL velocity
        v_lnal = lnal_velocity(r, Sigma_pred)
        v_lnal_lower = lnal_velocity(r, Sigma_lower)
        v_lnal_upper = lnal_velocity(r, Sigma_upper)
        
        # Chi-squared
        residuals = (v_lnal - v_obs) / v_err
        chi2 = np.sum(residuals**2)
        chi2_reduced = chi2 / len(r)
        
        return {
            'galaxy': galaxy_name,
            'r': r,
            'v_obs': v_obs,
            'v_err': v_err,
            'v_lnal': v_lnal,
            'v_lnal_lower': v_lnal_lower,
            'v_lnal_upper': v_lnal_upper,
            'Sigma_pred': Sigma_pred,
            'Sigma_lower': Sigma_lower,
            'Sigma_upper': Sigma_upper,
            'chi2': chi2,
            'chi2_reduced': chi2_reduced
        }


def plot_ml_fit(result, save_path=None):
    """Plot ML emulator fit."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    r_kpc = result['r'] / kpc
    
    # Surface density prediction
    Sigma_scale = (pc/M_sun)**2
    ax1.fill_between(r_kpc, 
                     result['Sigma_lower'] * Sigma_scale,
                     result['Sigma_upper'] * Sigma_scale,
                     alpha=0.3, color='blue', label='ML uncertainty')
    ax1.semilogy(r_kpc, result['Sigma_pred'] * Sigma_scale, 'b-', 
                 linewidth=2, label='ML prediction')
    ax1.set_xlabel('Radius [kpc]')
    ax1.set_ylabel('Σ [M⊙/pc²]')
    ax1.set_title('ML-Predicted Surface Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.1, 1e4)
    
    # Rotation curve
    ax2.errorbar(r_kpc, result['v_obs']/1000, yerr=result['v_err']/1000,
                 fmt='ko', markersize=4, label='Observed', alpha=0.7)
    ax2.fill_between(r_kpc,
                     result['v_lnal_lower']/1000,
                     result['v_lnal_upper']/1000,
                     alpha=0.3, color='red', label='LNAL uncertainty')
    ax2.plot(r_kpc, result['v_lnal']/1000, 'r-', linewidth=2.5,
             label='LNAL + ML')
    ax2.set_xlabel('Radius [kpc]')
    ax2.set_ylabel('Velocity [km/s]')
    ax2.set_title(f"{result['galaxy']} - ML Baryon Emulator")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Residuals
    residuals = (result['v_lnal'] - result['v_obs']) / result['v_err']
    ax3.scatter(r_kpc, residuals, c='purple', s=30)
    ax3.axhline(y=0, color='k', linestyle='--')
    ax3.set_xlabel('Radius [kpc]')
    ax3.set_ylabel('(LNAL - Obs) / Error')
    ax3.set_title('Normalized Residuals')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 5)
    
    # Summary
    ax4.axis('off')
    
    text = "ML Baryon Emulator Results:\n\n"
    text += f"χ²/N = {result['chi2_reduced']:.2f}\n\n"
    text += "Method:\n"
    text += "• Gaussian Process regression\n"
    text += "• Trained on SPARC galaxies\n"
    text += "• Features: radius, velocity shape\n"
    text += "• Predicts log(Σ) with uncertainty\n\n"
    text += "LNAL gravity: zero parameters\n"
    text += "Baryons: ML-predicted from observables"
    
    ax4.text(0.1, 0.9, text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    return fig


def main():
    """Run ML baryon emulator analysis."""
    print("LNAL ML Baryon Emulator")
    print("=" * 60)
    
    # Training galaxies
    train_galaxies = ['NGC3198', 'NGC2403', 'NGC6503', 'NGC0300', 
                      'NGC2841', 'NGC7814', 'UGC2885']
    
    # Test galaxies
    test_galaxies = ['DDO154', 'NGC3109', 'NGC1560', 'IC2574']
    
    # Create and train emulator
    emulator = BaryonEmulator()
    emulator.train_on_sparc(train_galaxies)
    
    # Test on new galaxies
    output_dir = 'lnal_ml_emulator_results'
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for galaxy in test_galaxies:
        print(f"\nPredicting {galaxy} with ML emulator...")
        result = emulator.fit_galaxy_ml(galaxy)
        
        if result is not None:
            results.append(result)
            plot_ml_fit(result, os.path.join(output_dir, f'{galaxy}_ml_emulator.png'))
            print(f"  χ²/N = {result['chi2_reduced']:.2f}")
    
    # Summary
    chi2_values = [r['chi2_reduced'] for r in results]
    print(f"\n{'='*60}")
    print(f"ML Emulator Summary:")
    print(f"  Trained on: {len(train_galaxies)} galaxies")
    print(f"  Tested on: {len(test_galaxies)} galaxies")
    print(f"  Mean χ²/N: {np.mean(chi2_values):.2f}")
    print(f"  Range: {np.min(chi2_values):.2f} - {np.max(chi2_values):.2f}")
    
    # Save summary
    with open(os.path.join(output_dir, 'ml_emulator_summary.json'), 'w') as f:
        json.dump({
            'description': 'ML baryon emulator with LNAL gravity',
            'train_galaxies': train_galaxies,
            'test_galaxies': test_galaxies,
            'results': [{
                'galaxy': r['galaxy'],
                'chi2_reduced': r['chi2_reduced']
            } for r in results]
        }, f, indent=2)


if __name__ == "__main__":
    main() 