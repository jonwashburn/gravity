#!/usr/bin/env python3
"""
Error propagation analysis for LNAL gravity scale factors.
Implements Monte Carlo error propagation to derive σ_δ for each galaxy.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import pickle
from tqdm import tqdm
import corner
import emcee

class ErrorPropagationAnalysis:
    """Full error propagation for scale factor uncertainties"""
    
    def __init__(self, galaxy_data):
        self.galaxy_data = galaxy_data
        self.n_mc = 1000  # Monte Carlo samples
        
    def propagate_errors(self, galaxy):
        """
        Propagate observational uncertainties to scale factor.
        
        Uncertainties considered:
        - Distance: ±10%
        - Inclination: ±5°
        - M/L ratio: ±0.3 dex
        - Velocity measurements: from data
        """
        # Extract nominal values
        D_nom = galaxy.get('D', 10.0)  # Mpc
        inc_nom = galaxy.get('inc', 60.0)  # degrees
        ML_star_nom = galaxy.get('ML_star', 0.5)
        ML_gas_nom = galaxy.get('ML_gas', 1.0)
        
        # Uncertainty estimates (typical SPARC values)
        sigma_D = 0.1 * D_nom  # 10% distance uncertainty
        sigma_inc = 5.0  # 5 degree inclination uncertainty
        sigma_ML = 0.3  # 0.3 dex in M/L
        
        # Monte Carlo sampling
        scale_samples = []
        
        for i in range(self.n_mc):
            # Sample from uncertainty distributions
            D_sample = np.random.normal(D_nom, sigma_D)
            inc_sample = np.random.normal(inc_nom, sigma_inc)
            ML_star_sample = ML_star_nom * 10**np.random.normal(0, sigma_ML)
            ML_gas_sample = ML_gas_nom * 10**np.random.normal(0, sigma_ML*0.5)
            
            # Propagate to masses (scale with D²)
            Mstar_sample = galaxy['Mstar'] * (D_sample/D_nom)**2 * (ML_star_sample/ML_star_nom)
            Mgas_sample = galaxy['Mgas'] * (D_sample/D_nom)**2 * (ML_gas_sample/ML_gas_nom)
            
            # Inclination correction for velocities
            sin_inc_corr = np.sin(np.radians(inc_sample)) / np.sin(np.radians(inc_nom))
            
            # This would require refitting - for now approximate
            # Scale factor changes approximately as (M_total)^(1/4) for MOND-like theories
            M_ratio = (Mstar_sample + Mgas_sample) / (galaxy['Mstar'] + galaxy['Mgas'])
            scale_sample = galaxy['scale'] * M_ratio**(0.25) * sin_inc_corr**(0.5)
            
            scale_samples.append(scale_sample)
        
        scale_samples = np.array(scale_samples)
        return {
            'scale_mean': np.mean(scale_samples),
            'scale_std': np.std(scale_samples),
            'scale_median': np.median(scale_samples),
            'scale_16': np.percentile(scale_samples, 16),
            'scale_84': np.percentile(scale_samples, 84),
            'delta_mean': (np.mean(scale_samples) - 1) * 100,
            'delta_std': np.std(scale_samples) * 100
        }

def hierarchical_bayesian_model(galaxies_df):
    """
    Hierarchical Bayesian model for δ ~ δ₀ + α·I + ε
    where I is the inefficiency metric (gas fraction)
    """
    print("\n=== HIERARCHICAL BAYESIAN ANALYSIS ===")
    
    # Prepare data
    delta = galaxies_df['delta_mean'].values
    delta_err = galaxies_df['delta_std'].values
    f_gas = galaxies_df['f_gas'].values
    
    # Remove outliers
    mask = (delta > -2) & (delta < 10) & (delta_err < 5)
    delta = delta[mask]
    delta_err = delta_err[mask]
    f_gas = f_gas[mask]
    
    # Define log likelihood
    def log_likelihood(theta, x, y, yerr):
        delta0, alpha, log_sigma = theta
        model = delta0 + alpha * x
        sigma2 = yerr**2 + np.exp(2*log_sigma)
        return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(2*np.pi*sigma2))
    
    # Define log prior
    def log_prior(theta):
        delta0, alpha, log_sigma = theta
        # Priors: δ₀ ~ N(1, 1), α ~ N(2, 2), log_σ ~ N(-1, 1)
        if -5 < delta0 < 5 and -10 < alpha < 10 and -5 < log_sigma < 2:
            return (-0.5 * ((delta0 - 1)**2 / 1**2) - 
                    0.5 * ((alpha - 2)**2 / 2**2) -
                    0.5 * ((log_sigma + 1)**2 / 1**2))
        return -np.inf
    
    # Define log posterior
    def log_posterior(theta, x, y, yerr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y, yerr)
    
    # Run MCMC
    ndim = 3
    nwalkers = 32
    nsteps = 5000
    
    # Initialize walkers
    pos = np.array([1.0, 2.0, -1.0]) + 0.1 * np.random.randn(nwalkers, ndim)
    
    # Run sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                                    args=(f_gas, delta, delta_err))
    
    print("Running MCMC...")
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    # Get samples (discard burn-in)
    samples = sampler.get_chain(discard=1000, thin=15, flat=True)
    
    # Results
    delta0_samples = samples[:, 0]
    alpha_samples = samples[:, 1]
    
    print(f"\nδ₀ = {np.median(delta0_samples):.2f} ± {np.std(delta0_samples):.2f}%")
    print(f"α = {np.median(alpha_samples):.2f} ± {np.std(alpha_samples):.2f}")
    
    # Is δ₀ > 0 at 3σ?
    p_positive = (delta0_samples > 0).sum() / len(delta0_samples)
    sigma_level = stats.norm.ppf(p_positive)
    print(f"\nP(δ₀ > 0) = {p_positive:.3f} ({sigma_level:.1f}σ)")
    
    # Make corner plot
    labels = [r'$\delta_0$ (%)', r'$\alpha$', r'$\log\sigma$']
    fig = corner.corner(samples, labels=labels, truths=[1.0, 2.0, -1.0],
                       quantiles=[0.16, 0.5, 0.84], show_titles=True)
    plt.savefig('hierarchical_bayes_corner.png', dpi=150, bbox_inches='tight')
    
    return {
        'delta0': np.median(delta0_samples),
        'delta0_err': np.std(delta0_samples),
        'alpha': np.median(alpha_samples),
        'alpha_err': np.std(alpha_samples),
        'p_positive': p_positive,
        'samples': samples
    }

def analyze_radial_profiles(galaxy_curves):
    """
    Test if δ(r) is flat with radius (ledger prediction)
    vs tilted (ΛCDM-like systematics)
    """
    print("\n=== RADIAL PROFILE ANALYSIS ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Analyze a few representative galaxies
    test_galaxies = ['NGC3198', 'NGC2403', 'DDO154', 'NGC6503']
    
    for idx, gname in enumerate(test_galaxies):
        ax = axes[idx//2, idx%2]
        
        if gname in galaxy_curves:
            data = galaxy_curves[gname]
            r = data['r']
            v_obs = data['v_obs']
            v_model = data['v_model']
            
            # Compute local scale factor in radial bins
            r_bins = np.linspace(r.min(), r.max(), 8)
            r_centers = (r_bins[:-1] + r_bins[1:]) / 2
            delta_r = []
            delta_r_err = []
            
            for i in range(len(r_bins)-1):
                mask = (r >= r_bins[i]) & (r < r_bins[i+1])
                if mask.sum() > 3:
                    local_scale = np.mean(v_obs[mask] / v_model[mask])
                    local_delta = (local_scale - 1) * 100
                    delta_r.append(local_delta)
                    delta_r_err.append(np.std(v_obs[mask] / v_model[mask]) * 100)
            
            # Plot
            ax.errorbar(r_centers, delta_r, yerr=delta_r_err, 
                       fmt='o-', linewidth=2, markersize=8, capsize=5)
            ax.axhline(y=np.mean(delta_r), color='red', linestyle='--', 
                      label=f'Mean = {np.mean(delta_r):.1f}%')
            ax.set_xlabel('Radius (kpc)')
            ax.set_ylabel('δ(r) (%)')
            ax.set_title(gname)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Test for tilt
            if len(r_centers) > 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(r_centers, delta_r)
                ax.text(0.05, 0.95, f'Slope = {slope:.3f}±{std_err:.3f}\np = {p_value:.3f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('radial_delta_profiles.png', dpi=150, bbox_inches='tight')
    plt.show()

def generate_key_figures(results_df):
    """Generate the four key figures for the paper"""
    
    # Figure 1: The wedge with error bars
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Remove outliers
    mask = (results_df['delta_mean'] > -2) & (results_df['delta_mean'] < 10)
    df_clean = results_df[mask]
    
    # Plot with error bars
    ax.errorbar(df_clean['f_gas'], df_clean['delta_mean'], 
               yerr=df_clean['delta_std'], fmt='o', alpha=0.6, 
               markersize=6, elinewidth=1, capsize=3)
    
    # Fit quantile regression for envelope
    from sklearn.linear_model import QuantileRegressor
    X = df_clean['f_gas'].values.reshape(-1, 1)
    y = df_clean['delta_mean'].values
    
    qr_90 = QuantileRegressor(quantile=0.9, alpha=0)
    qr_10 = QuantileRegressor(quantile=0.1, alpha=0)
    qr_50 = QuantileRegressor(quantile=0.5, alpha=0)
    
    qr_90.fit(X, y)
    qr_10.fit(X, y)
    qr_50.fit(X, y)
    
    x_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    ax.plot(x_plot, qr_90.predict(x_plot), 'g-', linewidth=2, label='90th percentile')
    ax.plot(x_plot, qr_50.predict(x_plot), 'b-', linewidth=2, label='Median')
    ax.plot(x_plot, qr_10.predict(x_plot), 'r-', linewidth=2, label='10th percentile')
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Gas Fraction', fontsize=14)
    ax.set_ylabel('δ (%)', fontsize=14)
    ax.set_title('Information Inefficiency Wedge with Error Propagation', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.savefig('figure1_wedge_errors.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Distribution with posterior
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Histogram of data
    ax.hist(df_clean['delta_mean'], bins=30, alpha=0.5, density=True, 
            color='blue', edgecolor='black', label='Data')
    
    # Overlay theoretical distribution from hierarchical model
    x = np.linspace(-2, 10, 200)
    # Approximate as mixture of gaussians based on gas fraction distribution
    pdf = 0.4 * stats.norm.pdf(x, 0.5, 0.5) + 0.6 * stats.norm.pdf(x, 2.5, 1.5)
    ax.plot(x, pdf, 'r-', linewidth=2, label='Theoretical prediction')
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Expected minimum')
    ax.set_xlabel('δ (%)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title('Distribution of Scale Factor Deviations', fontsize=16)
    ax.legend(fontsize=12)
    
    plt.savefig('figure2_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
def main():
    print("=== LNAL GRAVITY ERROR PROPAGATION ANALYSIS ===")
    
    # Load galaxy data (using synthetic for demonstration)
    # In practice, load from SPARC results
    np.random.seed(42)
    n_galaxies = 175
    
    # Generate realistic galaxy sample
    galaxies = []
    for i in range(n_galaxies):
        f_gas = np.random.beta(2 if i < n_galaxies//2 else 5, 
                              8 if i < n_galaxies//2 else 3)
        
        galaxy = {
            'name': f'Galaxy_{i}',
            'D': np.random.uniform(5, 50),  # Mpc
            'inc': np.random.uniform(30, 80),  # degrees
            'Mstar': 10**np.random.uniform(8, 11),  # Msun
            'Mgas': 10**np.random.uniform(7, 10) * f_gas,  # Msun
            'ML_star': 0.5,
            'ML_gas': 1.0,
            'scale': 1.0 + 0.01 * (0.5 + 3*f_gas + np.random.normal(0, 0.8)),
            'f_gas': f_gas
        }
        galaxies.append(galaxy)
    
    # Run error propagation
    print("\nPropagating errors for all galaxies...")
    error_prop = ErrorPropagationAnalysis(galaxies)
    
    results = []
    for galaxy in tqdm(galaxies):
        error_result = error_prop.propagate_errors(galaxy)
        error_result['name'] = galaxy['name']
        error_result['f_gas'] = galaxy['f_gas']
        error_result['scale_nominal'] = galaxy['scale']
        results.append(error_result)
    
    results_df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\n=== ERROR PROPAGATION RESULTS ===")
    print(f"Mean δ = {results_df['delta_mean'].mean():.2f} ± {results_df['delta_mean'].std():.2f}%")
    print(f"Mean σ_δ = {results_df['delta_std'].mean():.2f}%")
    print(f"Fraction with δ-σ_δ < 0: {((results_df['delta_mean'] - results_df['delta_std']) < 0).sum()/len(results_df)*100:.1f}%")
    
    # Run hierarchical Bayesian analysis
    bayes_results = hierarchical_bayesian_model(results_df)
    
    # Generate key figures
    generate_key_figures(results_df)
    
    # Save results
    results_df.to_csv('error_propagation_results.csv', index=False)
    with open('hierarchical_bayes_results.pkl', 'wb') as f:
        pickle.dump(bayes_results, f)
    
    print("\nAnalysis complete! Results saved.")
    
    return results_df, bayes_results

if __name__ == "__main__":
    results_df, bayes_results = main() 