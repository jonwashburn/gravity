#!/usr/bin/env python3
"""
LNAL Full Pipeline
==================
Complete pipeline for fitting LNAL gravity theory to SPARC galaxy data.
Includes optimization, visualization, and statistical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import emcee
import corner
import json
import os
from typing import Dict, List, Tuple
from datetime import datetime
import multiprocessing as mp

from lnal_solver_core_v2 import (
    LNALParameters, GalaxyData, v_circ, chi_squared, kpc
)
from lnal_sparc_loader import (
    load_sparc_galaxy, get_galaxy_sample, load_all_sparc_galaxies
)

# Set up matplotlib
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


class LNALFitter:
    """Main class for fitting LNAL model to galaxy data"""
    
    def __init__(self, galaxies: Dict[str, GalaxyData]):
        self.galaxies = galaxies
        self.galaxy_names = list(galaxies.keys())
        self.n_galaxies = len(galaxies)
        
        # Parameter bounds
        self.bounds = [
            (0.5, 2.0),   # hierarchy_strength
            (0.5, 2.0),   # temporal_coupling  
            (0.0, 0.5),   # coherence_fraction
        ]
        
        # Results storage
        self.results = {}
        
    def global_chi2(self, theta: np.ndarray) -> float:
        """Compute total χ² for all galaxies"""
        params = LNALParameters.from_array(theta)
        total_chi2 = 0
        
        for galaxy in self.galaxies.values():
            total_chi2 += chi_squared(params, galaxy)
            
        return total_chi2
    
    def log_likelihood(self, theta: np.ndarray) -> float:
        """Log likelihood for MCMC"""
        # Check bounds
        for i, (val, (low, high)) in enumerate(zip(theta, self.bounds)):
            if val < low or val > high:
                return -np.inf
                
        chi2 = self.global_chi2(theta)
        return -0.5 * chi2
    
    def log_prior(self, theta: np.ndarray) -> float:
        """Log prior for MCMC"""
        # Uniform priors within bounds
        for i, (val, (low, high)) in enumerate(zip(theta, self.bounds)):
            if val < low or val > high:
                return -np.inf
        return 0.0
    
    def log_probability(self, theta: np.ndarray) -> float:
        """Log posterior for MCMC"""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)
    
    def optimize_global(self, method: str = 'differential_evolution') -> Dict:
        """Find global best-fit parameters"""
        print(f"\nOptimizing global parameters using {method}...")
        
        if method == 'differential_evolution':
            result = differential_evolution(
                self.global_chi2,
                self.bounds,
                seed=42,
                workers=-1,  # Use all CPU cores
                maxiter=100,
                popsize=15,
                disp=True
            )
        else:
            # Nelder-Mead from multiple starting points
            best_result = None
            best_chi2 = np.inf
            
            for _ in range(10):
                x0 = [np.random.uniform(low, high) for low, high in self.bounds]
                result = minimize(
                    self.global_chi2,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 1000}
                )
                
                if result.fun < best_chi2:
                    best_chi2 = result.fun
                    best_result = result
                    
            result = best_result
        
        # Store results
        self.best_params = LNALParameters.from_array(result.x)
        self.best_chi2 = result.fun
        
        print(f"\nBest-fit parameters:")
        print(f"  Hierarchy strength: {self.best_params.hierarchy_strength:.3f}")
        print(f"  Temporal coupling: {self.best_params.temporal_coupling:.3f}")
        print(f"  Coherence fraction: {self.best_params.coherence_fraction:.3f}")
        print(f"  Total χ²: {self.best_chi2:.1f}")
        print(f"  Reduced χ²/N: {self.best_chi2 / self.count_data_points():.3f}")
        
        return {
            'best_params': self.best_params.to_array().tolist(),
            'best_chi2': float(self.best_chi2),
            'chi2_per_galaxy': self.compute_individual_chi2(self.best_params)
        }
    
    def run_mcmc(self, nwalkers: int = 32, nsteps: int = 2000) -> Dict:
        """Run MCMC to explore parameter space"""
        print(f"\nRunning MCMC with {nwalkers} walkers for {nsteps} steps...")
        
        ndim = len(self.bounds)
        
        # Initialize walkers around best-fit
        if hasattr(self, 'best_params'):
            center = self.best_params.to_array()
        else:
            center = np.array([1.0, 1.0, 0.15])
            
        pos = center + 0.1 * np.random.randn(nwalkers, ndim)
        
        # Ensure within bounds
        for i in range(ndim):
            pos[:, i] = np.clip(pos[:, i], self.bounds[i][0], self.bounds[i][1])
        
        # Run sampler
        with mp.Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.log_probability, pool=pool
            )
            sampler.run_mcmc(pos, nsteps, progress=True)
        
        # Get samples
        samples = sampler.get_chain(discard=500, thin=15, flat=True)
        
        # Compute statistics
        params_median = np.median(samples, axis=0)
        params_std = np.std(samples, axis=0)
        
        print(f"\nMCMC Results (median ± std):")
        param_names = ['hierarchy_strength', 'temporal_coupling', 'coherence_fraction']
        for name, med, std in zip(param_names, params_median, params_std):
            print(f"  {name}: {med:.3f} ± {std:.3f}")
        
        # Store results
        self.mcmc_samples = samples
        self.mcmc_params = LNALParameters.from_array(params_median)
        
        return {
            'params_median': params_median.tolist(),
            'params_std': params_std.tolist(),
            'acceptance_fraction': float(np.mean(sampler.acceptance_fraction))
        }
    
    def compute_individual_chi2(self, params: LNALParameters) -> Dict[str, float]:
        """Compute χ² for each galaxy"""
        chi2_dict = {}
        for name, galaxy in self.galaxies.items():
            chi2_dict[name] = float(chi_squared(params, galaxy))
        return chi2_dict
    
    def count_data_points(self) -> int:
        """Count total number of data points"""
        return sum(len(galaxy.r) for galaxy in self.galaxies.values())
    
    def plot_best_fits(self, output_dir: str = 'lnal_fits', n_examples: int = 6):
        """Plot rotation curves for best and worst fits"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get individual χ² values
        chi2_dict = self.compute_individual_chi2(self.best_params)
        
        # Sort by χ²/N
        sorted_galaxies = sorted(
            chi2_dict.items(), 
            key=lambda x: x[1] / len(self.galaxies[x[0]].r)
        )
        
        # Select examples
        n_best = n_examples // 2
        n_worst = n_examples - n_best
        examples = sorted_galaxies[:n_best] + sorted_galaxies[-n_worst:]
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (name, chi2) in enumerate(examples):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            galaxy = self.galaxies[name]
            
            # Model curve
            r_model = np.logspace(
                np.log10(galaxy.r.min()), 
                np.log10(galaxy.r.max()), 
                200
            )
            v_model = v_circ(r_model, self.best_params, galaxy)
            
            # Plot
            ax.errorbar(
                galaxy.r / kpc, 
                galaxy.v_obs / 1000,
                yerr=galaxy.v_err / 1000,
                fmt='o', 
                color='black',
                markersize=4,
                label='Data',
                alpha=0.7
            )
            
            ax.plot(
                r_model / kpc,
                v_model / 1000,
                'b-',
                linewidth=2,
                label='LNAL model'
            )
            
            # Labels
            chi2_n = chi2 / len(galaxy.r)
            ax.set_title(f'{name} (χ²/N = {chi2_n:.2f})')
            ax.set_xlabel('Radius [kpc]')
            ax.set_ylabel('Velocity [km/s]')
            ax.legend(loc='best')
            ax.set_xlim(0, galaxy.r.max() / kpc * 1.1)
            ax.set_ylim(0, galaxy.v_obs.max() / 1000 * 1.2)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'best_worst_fits.png'))
        plt.close()
        
        print(f"\nSaved example fits to {output_dir}/best_worst_fits.png")
    
    def plot_chi2_distribution(self, output_dir: str = 'lnal_fits'):
        """Plot χ²/N distribution"""
        os.makedirs(output_dir, exist_ok=True)
        
        chi2_dict = self.compute_individual_chi2(self.best_params)
        chi2_per_n = [
            chi2 / len(self.galaxies[name].r) 
            for name, chi2 in chi2_dict.items()
        ]
        
        plt.figure(figsize=(10, 6))
        plt.hist(chi2_per_n, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=1, color='red', linestyle='--', label='χ²/N = 1')
        plt.xlabel('χ²/N')
        plt.ylabel('Number of galaxies')
        plt.title(f'LNAL Model Performance ({self.n_galaxies} galaxies)')
        plt.legend()
        
        # Add statistics
        mean_chi2n = np.mean(chi2_per_n)
        median_chi2n = np.median(chi2_per_n)
        plt.text(0.7, 0.9, f'Mean: {mean_chi2n:.2f}\nMedian: {median_chi2n:.2f}',
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'chi2_distribution.png'))
        plt.close()
        
        print(f"Saved χ² distribution to {output_dir}/chi2_distribution.png")
    
    def save_results(self, output_dir: str = 'lnal_fits'):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_galaxies': self.n_galaxies,
            'galaxy_names': self.galaxy_names,
            'best_fit': {
                'parameters': self.best_params.to_array().tolist(),
                'total_chi2': float(self.best_chi2),
                'reduced_chi2': float(self.best_chi2 / self.count_data_points()),
                'chi2_per_galaxy': self.compute_individual_chi2(self.best_params)
            }
        }
        
        # Add MCMC results if available
        if hasattr(self, 'mcmc_samples'):
            results['mcmc'] = {
                'median_params': np.median(self.mcmc_samples, axis=0).tolist(),
                'std_params': np.std(self.mcmc_samples, axis=0).tolist(),
                'n_samples': len(self.mcmc_samples)
            }
        
        # Save JSON
        with open(os.path.join(output_dir, 'lnal_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nSaved results to {output_dir}/lnal_results.json")


def main():
    """Run full LNAL analysis pipeline"""
    print("=" * 70)
    print("LNAL Full Analysis Pipeline")
    print("=" * 70)
    
    # Load galaxies
    print("\nLoading SPARC galaxies...")
    galaxy_sample = get_galaxy_sample('high_quality')
    galaxies = {}
    
    for name in galaxy_sample:
        galaxy = load_sparc_galaxy(name)
        if galaxy is not None:
            galaxies[name] = galaxy
            
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Create fitter
    fitter = LNALFitter(galaxies)
    
    # Optimize
    opt_results = fitter.optimize_global()
    
    # Run MCMC (optional - comment out for quick tests)
    # mcmc_results = fitter.run_mcmc(nwalkers=32, nsteps=1000)
    
    # Generate plots
    fitter.plot_best_fits()
    fitter.plot_chi2_distribution()
    
    # Save results
    fitter.save_results()
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main() 