#!/usr/bin/env python3
"""
LNAL Pure Pipeline
==================
Fit galaxy rotation curves using the pure LNAL formula
by inferring the baryon distribution, not tuning gravity.
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from lnal_pure_formula import lnal_circular_velocity
from lnal_infer_baryons import (
    GalaxyObservables, BaryonParameters, 
    sample_baryon_parameters, build_surface_density,
    infer_rotation_curve, kpc, M_sun
)

# Set up plotting
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_sparc_observables(galaxy_name: str, 
                          catalog_file: str = 'SPARC_Lelli2016c.mrt') -> Optional[GalaxyObservables]:
    """
    Load galaxy observables from SPARC catalog.
    
    This is a simplified version - in practice would parse the full MRT file.
    """
    # Hardcoded values for test galaxies
    # In production, parse from catalog_file
    test_data = {
        'NGC3198': {
            'L36': 3.8279e10,  # L_sun at 3.6μm
            'q': 0.23,  # axis ratio
            'z': 0.00215,
            'W50': 314,  # HI line width km/s
            'SHI': 286.7,  # HI flux Jy km/s
            'D': 13.8  # Distance Mpc (if known)
        },
        'NGC2403': {
            'L36': 1.0041e10,
            'q': 0.54,
            'z': 0.00043,
            'W50': 262,
            'SHI': 1089.0,
            'D': 3.22
        },
        'DDO154': {
            'L36': 5.3e7,
            'q': 0.66,
            'z': 0.00125,
            'W50': 94,
            'SHI': 53.5,
            'D': 3.7
        },
        'NGC6503': {
            'L36': 1.2845e10,
            'q': 0.26,
            'z': 0.00004,
            'W50': 233,
            'SHI': 495.8,
            'D': 6.3
        },
        'UGC02885': {
            'L36': 4.03525e11,
            'q': 0.55,
            'z': 0.01935,
            'W50': 579,
            'SHI': 39.2,
            'D': 79.7
        }
    }
    
    if galaxy_name not in test_data:
        print(f"Warning: {galaxy_name} not in test catalog")
        return None
    
    data = test_data[galaxy_name]
    
    # Create observables
    obs = GalaxyObservables(
        name=galaxy_name,
        luminosity_3p6=data['L36'],
        axis_ratio=data['q'],
        redshift=data['z'],
        HI_flux=data.get('SHI'),
        distance_prior=(data['D'], data['D'] * 0.1) if 'D' in data else None
    )
    
    return obs


def load_rotation_curve(galaxy_name: str, 
                       rotmod_dir: str = 'Rotmod_LTG') -> Optional[Dict]:
    """Load observed rotation curve from SPARC files"""
    import os
    
    filepath = os.path.join(rotmod_dir, f'{galaxy_name}_rotmod.dat')
    if not os.path.exists(filepath):
        return None
    
    # Parse file
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    if not data:
        return None
    
    data = np.array(data)
    return {
        'r': data[:, 0] * kpc,  # kpc to m
        'v_obs': data[:, 1] * 1000,  # km/s to m/s
        'v_err': data[:, 2] * 1000  # km/s to m/s
    }


class PureLNALFitter:
    """Fit rotation curves by inferring baryons with fixed LNAL law"""
    
    def __init__(self, obs: GalaxyObservables, rotation_data: Dict):
        self.obs = obs
        self.r_data = rotation_data['r']
        self.v_obs = rotation_data['v_obs']
        self.v_err = rotation_data['v_err']
        
        # Parameter names and bounds
        self.param_names = [
            'distance', 'inclination', 'log_M_star', 
            'R_star', 'log_M_HI', 'R_HI'
        ]
        
        # Prior bounds (will be refined based on observables)
        self.bounds = self._get_bounds()
        
    def _get_bounds(self) -> List[Tuple[float, float]]:
        """Get reasonable bounds for baryon parameters"""
        # Distance
        if self.obs.distance_prior:
            d_mean, d_std = self.obs.distance_prior
            d_bounds = (max(0.1, d_mean - 3*d_std), d_mean + 3*d_std)
        else:
            d_bounds = (0.1, 200)  # Mpc
        
        # Other bounds
        bounds = [
            d_bounds,  # distance [Mpc]
            (0, np.pi/2),  # inclination [rad]
            (7, 12),  # log10(M_star/M_sun)
            (0.1, 20),  # R_star [kpc]
            (6, 11),  # log10(M_HI/M_sun)
            (0.5, 50),  # R_HI [kpc]
        ]
        
        return bounds
    
    def log_prior(self, theta: np.ndarray) -> float:
        """Log prior probability"""
        # Check bounds
        for val, (low, high) in zip(theta, self.bounds):
            if val < low or val > high:
                return -np.inf
        
        # Unpack parameters
        distance, incl, log_M_star, R_star, log_M_HI, R_HI = theta
        
        # Physical constraints
        if R_HI < R_star:  # HI should extend beyond stars
            return -np.inf
        
        # Add Gaussian priors based on scaling relations
        log_p = 0
        
        # Distance prior if available
        if self.obs.distance_prior:
            d_mean, d_std = self.obs.distance_prior
            log_p += -0.5 * ((distance - d_mean) / d_std)**2
        
        # M/L prior
        expected_log_M_star = np.log10(0.6 * self.obs.luminosity_3p6)
        log_p += -0.5 * ((log_M_star - expected_log_M_star) / 0.15)**2
        
        # Size prior
        expected_R_star = 3.0 * (10**log_M_star / 1e10)**0.3
        log_p += -0.5 * ((R_star - expected_R_star) / (0.3 * expected_R_star))**2
        
        return log_p
    
    def log_likelihood(self, theta: np.ndarray) -> float:
        """Log likelihood of data given parameters"""
        # Create baryon parameters
        params = BaryonParameters(
            distance=theta[0],
            inclination=theta[1],
            M_star=10**theta[2] * M_sun,
            R_star=theta[3] * kpc,
            M_HI=10**theta[4] * M_sun,
            R_HI=theta[5] * kpc,
            M_H2=0  # Ignore H2 for simplicity
        )
        
        # Build surface density
        Sigma = build_surface_density(params)
        
        # Compute model velocity
        try:
            v_model = lnal_circular_velocity(self.r_data, Sigma)
        except:
            return -np.inf
        
        # Chi-squared
        chi2 = np.sum(((v_model - self.v_obs) / self.v_err)**2)
        
        return -0.5 * chi2
    
    def log_probability(self, theta: np.ndarray) -> float:
        """Log posterior probability"""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)
    
    def run_mcmc(self, nwalkers: int = 32, nsteps: int = 2000) -> Dict:
        """Run MCMC to infer baryon parameters"""
        ndim = len(self.bounds)
        
        # Initialize walkers
        # Start from prior samples
        pos = []
        for _ in range(nwalkers):
            sample = sample_baryon_parameters(self.obs, n_samples=1)[0]
            # Convert to our parameterization
            theta = [
                sample[0],  # distance
                sample[1],  # inclination
                sample[2],  # log_M_star
                sample[3],  # R_star (already in kpc)
                sample[4],  # log_M_HI
                sample[5],  # R_HI (already in kpc)
            ]
            # Ensure within bounds
            for i, (val, (low, high)) in enumerate(zip(theta, self.bounds)):
                theta[i] = np.clip(val, low, high)
            pos.append(theta)
        pos = np.array(pos)
        
        # Run sampler
        print(f"Running MCMC for {self.obs.name}...")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)
        sampler.run_mcmc(pos, nsteps, progress=True)
        
        # Get samples
        samples = sampler.get_chain(discard=500, thin=15, flat=True)
        
        # Best fit (maximum posterior)
        log_prob = sampler.get_log_prob(discard=500, thin=15, flat=True)
        best_idx = np.argmax(log_prob)
        best_params = samples[best_idx]
        
        # Statistics
        params_mean = np.mean(samples, axis=0)
        params_std = np.std(samples, axis=0)
        
        return {
            'samples': samples,
            'best_params': best_params,
            'params_mean': params_mean,
            'params_std': params_std,
            'acceptance_fraction': np.mean(sampler.acceptance_fraction)
        }
    
    def plot_fit(self, mcmc_result: Dict, output_dir: str = 'lnal_pure_results'):
        """Plot rotation curve fit with uncertainties"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get samples
        samples = mcmc_result['samples']
        n_plot = min(100, len(samples))
        
        # Plot rotation curves
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Main plot
        ax1.errorbar(self.r_data/kpc, self.v_obs/1000, yerr=self.v_err/1000,
                    fmt='o', color='black', markersize=4, label='Data', alpha=0.7)
        
        # Plot sample of models
        v_models = []
        for i in np.random.randint(0, len(samples), n_plot):
            params = BaryonParameters(
                distance=samples[i, 0],
                inclination=samples[i, 1],
                M_star=10**samples[i, 2] * M_sun,
                R_star=samples[i, 3] * kpc,
                M_HI=10**samples[i, 4] * M_sun,
                R_HI=samples[i, 5] * kpc,
                M_H2=0
            )
            Sigma = build_surface_density(params)
            v_model = lnal_circular_velocity(self.r_data, Sigma)
            v_models.append(v_model)
            ax1.plot(self.r_data/kpc, v_model/1000, 'b-', alpha=0.05)
        
        # Best fit
        best_params = BaryonParameters(
            distance=mcmc_result['best_params'][0],
            inclination=mcmc_result['best_params'][1],
            M_star=10**mcmc_result['best_params'][2] * M_sun,
            R_star=mcmc_result['best_params'][3] * kpc,
            M_HI=10**mcmc_result['best_params'][4] * M_sun,
            R_HI=mcmc_result['best_params'][5] * kpc,
            M_H2=0
        )
        Sigma_best = build_surface_density(best_params)
        v_best = lnal_circular_velocity(self.r_data, Sigma_best)
        ax1.plot(self.r_data/kpc, v_best/1000, 'r-', linewidth=2, 
                label='Best fit (pure LNAL)')
        
        ax1.set_ylabel('Velocity [km/s]')
        ax1.set_title(f'{self.obs.name} - Pure LNAL Fit (No Free Parameters in Gravity Law)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, self.r_data.max()/kpc * 1.1)
        ax1.set_ylim(0, max(self.v_obs.max()/1000 * 1.2, 50))
        
        # Residuals
        v_models = np.array(v_models)
        v_mean = np.mean(v_models, axis=0)
        v_std = np.std(v_models, axis=0)
        
        residuals = (self.v_obs - v_mean) / self.v_err
        ax2.errorbar(self.r_data/kpc, residuals, yerr=1, 
                    fmt='o', color='black', markersize=4, alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.fill_between(self.r_data/kpc, -v_std/self.v_err, v_std/self.v_err,
                        alpha=0.3, color='blue', label='Model uncertainty')
        ax2.set_xlabel('Radius [kpc]')
        ax2.set_ylabel('(Data - Model) / Error')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(-5, 5)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.obs.name}_fit.png'))
        plt.close()
        
        # Corner plot
        fig = corner.corner(samples, labels=self.param_names,
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 10})
        fig.suptitle(f'{self.obs.name} - Baryon Parameters', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.obs.name}_corner.png'))
        plt.close()
        
        return v_mean, v_std


def run_pure_analysis(galaxy_names: List[str], 
                     output_dir: str = 'lnal_pure_results'):
    """Run pure LNAL analysis on multiple galaxies"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for galaxy_name in galaxy_names:
        print(f"\n{'='*60}")
        print(f"Analyzing {galaxy_name}")
        print('='*60)
        
        # Load observables
        obs = load_sparc_observables(galaxy_name)
        if obs is None:
            print(f"Could not load observables for {galaxy_name}")
            continue
        
        # Load rotation curve
        rot_data = load_rotation_curve(galaxy_name)
        if rot_data is None:
            print(f"Could not load rotation curve for {galaxy_name}")
            continue
        
        # Create fitter
        fitter = PureLNALFitter(obs, rot_data)
        
        # Run MCMC
        mcmc_result = fitter.run_mcmc(nwalkers=32, nsteps=1000)
        
        # Plot results
        v_mean, v_std = fitter.plot_fit(mcmc_result, output_dir)
        
        # Compute chi-squared
        residuals = (rot_data['v_obs'] - v_mean) / rot_data['v_err']
        chi2 = np.sum(residuals**2)
        chi2_reduced = chi2 / len(residuals)
        
        # Store results
        results[galaxy_name] = {
            'chi2': float(chi2),
            'chi2_reduced': float(chi2_reduced),
            'n_data': len(residuals),
            'params_mean': mcmc_result['params_mean'].tolist(),
            'params_std': mcmc_result['params_std'].tolist(),
            'acceptance_fraction': float(mcmc_result['acceptance_fraction'])
        }
        
        # Print summary
        print(f"\nResults for {galaxy_name}:")
        print(f"  χ²/N = {chi2_reduced:.2f}")
        print(f"  Distance: {mcmc_result['params_mean'][0]:.1f} ± {mcmc_result['params_std'][0]:.1f} Mpc")
        print(f"  M_star: {10**mcmc_result['params_mean'][2]:.2e} ± {10**mcmc_result['params_mean'][2] * mcmc_result['params_std'][2] * np.log(10):.2e} M_sun")
        print(f"  M_HI: {10**mcmc_result['params_mean'][4]:.2e} ± {10**mcmc_result['params_mean'][4] * mcmc_result['params_std'][4] * np.log(10):.2e} M_sun")
    
    # Save results
    with open(os.path.join(output_dir, 'pure_lnal_results.json'), 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'theory': 'Pure LNAL (zero free parameters)',
                'method': 'Bayesian inference of baryon distribution'
            },
            'galaxies': results
        }, f, indent=2)
    
    # Summary plot
    if results:
        plt.figure(figsize=(10, 6))
        chi2_values = [r['chi2_reduced'] for r in results.values()]
        plt.hist(chi2_values, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=1, color='red', linestyle='--', label='χ²/N = 1')
        plt.xlabel('χ²/N')
        plt.ylabel('Number of galaxies')
        plt.title('Pure LNAL Performance (No Gravity Parameters Tuned)')
        plt.legend()
        
        # Add text
        mean_chi2 = np.mean(chi2_values)
        plt.text(0.7, 0.9, f'Mean χ²/N = {mean_chi2:.2f}\n{len(results)} galaxies',
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pure_lnal_summary.png'))
        plt.close()
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to {output_dir}/")
    print('='*60)


if __name__ == "__main__":
    # Test galaxies
    test_galaxies = ['NGC3198', 'NGC2403', 'DDO154', 'NGC6503']
    
    # Run analysis
    run_pure_analysis(test_galaxies) 