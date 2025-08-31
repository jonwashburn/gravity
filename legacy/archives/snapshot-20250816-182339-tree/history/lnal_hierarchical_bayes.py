#!/usr/bin/env python3
"""
LNAL Hierarchical Bayesian Model
=================================
Learn population-level parameters across multiple galaxies.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm, lognorm
import emcee
import corner
import json
import os
from multiprocessing import Pool

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


def exponential_disk(r, M_disk, R_d):
    """Exponential disk profile."""
    Sigma_0 = M_disk / (2 * np.pi * R_d**2)
    return Sigma_0 * np.exp(-r / R_d)


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


class HierarchicalModel:
    """Hierarchical Bayesian model for galaxy population."""
    
    def __init__(self, galaxy_names):
        """Initialize with list of galaxies."""
        self.galaxies = []
        
        for name in galaxy_names:
            data = load_sparc_rotmod(name)
            if data is not None:
                # Handle missing errors
                data['v_err'][data['v_err'] <= 0] = 5000  # 5 km/s default
                self.galaxies.append({
                    'name': name,
                    'data': data
                })
        
        print(f"Loaded {len(self.galaxies)} galaxies")
        
        # Parameter structure:
        # Population hyperparameters: mu_log_M, sigma_log_M, mu_log_R, sigma_log_R
        # Individual galaxy parameters: log_M_disk[i], log_R_d[i], log_M_gas[i], log_R_g[i]
        self.n_galaxies = len(self.galaxies)
        self.n_hyper = 8  # 4 means + 4 sigmas
        self.n_params_per_galaxy = 4  # M_disk, R_d, M_gas, R_g
        self.n_params = self.n_hyper + self.n_galaxies * self.n_params_per_galaxy
    
    def log_prior(self, params):
        """Log prior probability."""
        # Unpack hyperparameters
        mu_log_M_disk = params[0]
        sigma_log_M_disk = params[1]
        mu_log_R_disk = params[2]
        sigma_log_R_disk = params[3]
        mu_log_M_gas = params[4]
        sigma_log_M_gas = params[5]
        mu_log_R_gas = params[6]
        sigma_log_R_gas = params[7]
        
        # Hyperprior bounds
        if not (8 < mu_log_M_disk < 12):  # 10^8 to 10^12 M_sun
            return -np.inf
        if not (0.01 < sigma_log_M_disk < 2):
            return -np.inf
        if not (-1 < mu_log_R_disk < 1.5):  # 0.1 to 30 kpc
            return -np.inf
        if not (0.01 < sigma_log_R_disk < 1):
            return -np.inf
        
        if not (7 < mu_log_M_gas < 11):  # 10^7 to 10^11 M_sun
            return -np.inf
        if not (0.01 < sigma_log_M_gas < 2):
            return -np.inf
        if not (-1 < mu_log_R_gas < 2):  # 0.1 to 100 kpc
            return -np.inf
        if not (0.01 < sigma_log_R_gas < 1):
            return -np.inf
        
        log_prior = 0
        
        # Individual galaxy parameters
        for i in range(self.n_galaxies):
            idx = self.n_hyper + i * self.n_params_per_galaxy
            log_M_disk = params[idx]
            log_R_disk = params[idx + 1]
            log_M_gas = params[idx + 2]
            log_R_gas = params[idx + 3]
            
            # Hierarchical priors
            log_prior += norm.logpdf(log_M_disk, mu_log_M_disk, sigma_log_M_disk)
            log_prior += norm.logpdf(log_R_disk, mu_log_R_disk, sigma_log_R_disk)
            log_prior += norm.logpdf(log_M_gas, mu_log_M_gas, sigma_log_M_gas)
            log_prior += norm.logpdf(log_R_gas, mu_log_R_gas, sigma_log_R_gas)
        
        return log_prior
    
    def log_likelihood(self, params):
        """Log likelihood of data given parameters."""
        log_like = 0
        
        # For each galaxy
        for i in range(self.n_galaxies):
            galaxy = self.galaxies[i]
            data = galaxy['data']
            
            # Extract galaxy parameters
            idx = self.n_hyper + i * self.n_params_per_galaxy
            M_disk = 10**params[idx] * M_sun
            R_disk = 10**params[idx + 1] * kpc
            M_gas = 10**params[idx + 2] * M_sun
            R_gas = 10**params[idx + 3] * kpc
            
            # Generate model
            r = data['r']
            Sigma_disk = exponential_disk(r, M_disk, R_disk)
            Sigma_gas = exponential_disk(r, M_gas, R_gas)  # Simple approximation
            Sigma_total = Sigma_disk + Sigma_gas
            
            # LNAL velocity
            v_model = lnal_velocity(r, Sigma_total)
            
            # Chi-squared
            residuals = (v_model - data['v_obs']) / data['v_err']
            chi2 = np.sum(residuals**2)
            
            # Add to log likelihood
            log_like += -0.5 * chi2
        
        return log_like
    
    def log_posterior(self, params):
        """Log posterior probability."""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood(params)
        if not np.isfinite(ll):
            return -np.inf
        
        return lp + ll
    
    def sample(self, n_walkers=32, n_steps=1000, n_burn=200):
        """Run MCMC sampling."""
        # Initial positions
        pos = []
        
        # Hyperparameter initial guesses
        pos.extend([10.5, 0.5, 0.5, 0.3])  # Disk mass and radius
        pos.extend([9.5, 0.5, 1.0, 0.3])   # Gas mass and radius
        
        # Individual galaxy parameters
        for i in range(self.n_galaxies):
            pos.extend([10.5, 0.5, 9.5, 1.0])  # log values
        
        # Add small random perturbations
        pos = np.array(pos)
        pos = pos + 0.1 * np.random.randn(n_walkers, len(pos))
        
        # Set up sampler
        print(f"Running MCMC with {n_walkers} walkers for {n_steps} steps...")
        
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers, self.n_params, self.log_posterior, pool=pool
            )
            
            # Run burn-in
            print("Burn-in phase...")
            pos, _, _ = sampler.run_mcmc(pos, n_burn, progress=True)
            sampler.reset()
            
            # Run production
            print("Production phase...")
            sampler.run_mcmc(pos, n_steps, progress=True)
        
        return sampler
    
    def analyze_results(self, sampler, output_dir='lnal_hierarchical_results'):
        """Analyze and plot MCMC results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get chains
        chain = sampler.get_chain(flat=True)
        
        # Extract hyperparameters
        hyper_chain = chain[:, :self.n_hyper]
        hyper_labels = [
            r'$\mu_{\log M_{\rm disk}}$', r'$\sigma_{\log M_{\rm disk}}$',
            r'$\mu_{\log R_{\rm disk}}$', r'$\sigma_{\log R_{\rm disk}}$',
            r'$\mu_{\log M_{\rm gas}}$', r'$\sigma_{\log M_{\rm gas}}$',
            r'$\mu_{\log R_{\rm gas}}$', r'$\sigma_{\log R_{\rm gas}}$'
        ]
        
        # Corner plot for hyperparameters
        fig = corner.corner(hyper_chain, labels=hyper_labels, 
                           quantiles=[0.16, 0.5, 0.84], show_titles=True)
        plt.savefig(os.path.join(output_dir, 'hyperparameters_corner.png'), dpi=150)
        plt.close()
        
        # Get median hyperparameters
        hyper_median = np.median(hyper_chain, axis=0)
        hyper_std = np.std(hyper_chain, axis=0)
        
        # Plot individual galaxy fits with best parameters
        for i, galaxy in enumerate(self.galaxies):
            idx = self.n_hyper + i * self.n_params_per_galaxy
            galaxy_chain = chain[:, idx:idx+self.n_params_per_galaxy]
            
            # Median parameters
            median_params = np.median(galaxy_chain, axis=0)
            M_disk = 10**median_params[0] * M_sun
            R_disk = 10**median_params[1] * kpc
            M_gas = 10**median_params[2] * M_sun
            R_gas = 10**median_params[3] * kpc
            
            # Generate model
            data = galaxy['data']
            r = data['r']
            Sigma_disk = exponential_disk(r, M_disk, R_disk)
            Sigma_gas = exponential_disk(r, M_gas, R_gas)
            Sigma_total = Sigma_disk + Sigma_gas
            v_model = lnal_velocity(r, Sigma_total)
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            r_kpc = r / kpc
            
            # Rotation curve
            ax1.errorbar(r_kpc, data['v_obs']/1000, yerr=data['v_err']/1000,
                        fmt='ko', markersize=4, label='Observed', alpha=0.7)
            ax1.plot(r_kpc, v_model/1000, 'r-', linewidth=2, label='Hierarchical LNAL')
            ax1.set_xlabel('Radius [kpc]')
            ax1.set_ylabel('Velocity [km/s]')
            ax1.set_title(f"{galaxy['name']} - Hierarchical Fit")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Parameter distributions
            ax2.hist(galaxy_chain[:, 0], bins=30, alpha=0.5, label='log M_disk', density=True)
            ax2.hist(galaxy_chain[:, 1], bins=30, alpha=0.5, label='log R_disk', density=True)
            ax2.set_xlabel('Parameter value')
            ax2.set_ylabel('Probability density')
            ax2.set_title('Parameter posteriors')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{galaxy["name"]}_hierarchical.png'), dpi=150)
            plt.close()
        
        # Save results
        results = {
            'hyperparameters': {
                'mu_log_M_disk': float(hyper_median[0]),
                'sigma_log_M_disk': float(hyper_median[1]),
                'mu_log_R_disk': float(hyper_median[2]),
                'sigma_log_R_disk': float(hyper_median[3]),
                'mu_log_M_gas': float(hyper_median[4]),
                'sigma_log_M_gas': float(hyper_median[5]),
                'mu_log_R_gas': float(hyper_median[6]),
                'sigma_log_R_gas': float(hyper_median[7]),
            },
            'description': 'Hierarchical Bayesian model learning population-level parameters'
        }
        
        with open(os.path.join(output_dir, 'hierarchical_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\nHierarchical Model Results:")
        print("=" * 60)
        print("Population hyperparameters (median ± std):")
        print(f"  log M_disk: {hyper_median[0]:.2f} ± {hyper_std[0]:.2f}")
        print(f"  log R_disk: {hyper_median[2]:.2f} ± {hyper_std[2]:.2f}")
        print(f"  log M_gas:  {hyper_median[4]:.2f} ± {hyper_std[4]:.2f}")
        print(f"  log R_gas:  {hyper_median[6]:.2f} ± {hyper_std[6]:.2f}")
        
        print("\nPopulation scatter:")
        print(f"  σ(log M_disk): {hyper_median[1]:.2f}")
        print(f"  σ(log R_disk): {hyper_median[3]:.2f}")
        print(f"  σ(log M_gas):  {hyper_median[5]:.2f}")
        print(f"  σ(log R_gas):  {hyper_median[7]:.2f}")
        
        return results


def main():
    """Run hierarchical analysis."""
    print("LNAL Hierarchical Bayesian Analysis")
    print("=" * 60)
    
    # Test galaxies
    galaxy_names = ['NGC3198', 'NGC2403', 'NGC6503', 'DDO154']
    
    # Create model
    model = HierarchicalModel(galaxy_names)
    
    # Run MCMC (reduced for testing)
    sampler = model.sample(n_walkers=16, n_steps=500, n_burn=100)
    
    # Analyze results
    results = model.analyze_results(sampler)
    
    print("\nAnalysis complete! Results saved to lnal_hierarchical_results/")


if __name__ == "__main__":
    main() 