#!/usr/bin/env python3
"""
Two-Level Bayesian Optimization for Recognition Science Gravity
===============================================================
Global parameters via Optuna, per-galaxy via Nelder-Mead
"""

import numpy as np
import optuna
from scipy.optimize import minimize
import pandas as pd
import json
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from rs_gravity_tunable_enhanced import EnhancedGravitySolver, GalaxyData, GalaxyParameters
import joblib
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "Rotmod_LTG"
RESULTS_DIR = "bayesian_optimization_results"
N_TRIALS = 100  # Optuna trials
MAX_GALAXIES = None  # Use all galaxies (set to number to limit)
N_JOBS = -1  # Parallel jobs (-1 = all cores)

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global parameter bounds
GLOBAL_BOUNDS = {
    'lambda_eff': (20e-6, 100e-6),      # 20-100 μm
    'beta_scale': (0.5, 1.5),            # 50-150% of theoretical
    'mu_scale': (0.2, 2.0),              # 20-200% of theoretical
    'coupling_scale': (0.2, 2.0),        # 20-200% of theoretical
}

# Per-galaxy parameter bounds
GALAXY_BOUNDS = {
    'ML_disk': (0.3, 1.0),               # Disk M/L ratio
    'ML_bulge': (0.3, 0.9),              # Bulge M/L ratio  
    'gas_factor': (1.25, 1.40),          # He correction
    'h_scale': (100, 600),               # Scale height in pc
}


def load_galaxy_from_file(filepath):
    """Load galaxy data from rotmod file"""
    name = os.path.basename(filepath).replace('_rotmod.dat', '')
    
    try:
        data = np.loadtxt(filepath, skiprows=1)
        
        R_kpc = data[:, 0]
        v_obs = data[:, 1]
        
        if len(R_kpc) < 5:
            return None
            
        v_err = np.maximum(0.03 * v_obs, 2.0)
        
        if data.shape[1] >= 7:
            sigma_gas = data[:, 5]  # Raw values (scaling in solver)
            sigma_disk = data[:, 6]
        else:
            sigma_gas = 10 * np.exp(-R_kpc / 2)
            sigma_disk = 100 * np.exp(-R_kpc / 3)
        
        sigma_bulge = None
        if data.shape[1] >= 8 and np.any(data[:, 7] > 0):
            sigma_bulge = data[:, 7]
        
        return GalaxyData(
            name=name,
            R_kpc=R_kpc,
            v_obs=v_obs,
            v_err=v_err,
            sigma_gas=sigma_gas,
            sigma_disk=sigma_disk,
            sigma_bulge=sigma_bulge
        )
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def optimize_galaxy_params(galaxy: GalaxyData, solver: EnhancedGravitySolver, 
                         max_iter: int = 20) -> Tuple[GalaxyParameters, float]:
    """Optimize per-galaxy parameters using Nelder-Mead"""
    
    # Define objective function
    def objective(x):
        params = GalaxyParameters(
            ML_disk=x[0],
            ML_bulge=x[1] if galaxy.sigma_bulge is not None else 0.7,
            gas_factor=x[2],
            h_scale=x[3]
        )
        
        try:
            result = solver.solve_galaxy(galaxy, params)
            return result['chi2_reduced']
        except:
            return 1e6
    
    # Initial guess
    if galaxy.sigma_bulge is not None:
        x0 = [0.5, 0.7, 1.33, 300]
        bounds = [GALAXY_BOUNDS['ML_disk'], GALAXY_BOUNDS['ML_bulge'], 
                 GALAXY_BOUNDS['gas_factor'], GALAXY_BOUNDS['h_scale']]
    else:
        x0 = [0.5, 1.33, 300]
        bounds = [GALAXY_BOUNDS['ML_disk'], GALAXY_BOUNDS['gas_factor'], 
                 GALAXY_BOUNDS['h_scale']]
        objective_no_bulge = lambda x: objective([x[0], 0.7, x[1], x[2]])
        objective = objective_no_bulge
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', 
                     bounds=bounds, options={'maxiter': max_iter})
    
    # Extract optimized parameters
    if galaxy.sigma_bulge is not None:
        opt_params = GalaxyParameters(
            ML_disk=result.x[0],
            ML_bulge=result.x[1],
            gas_factor=result.x[2],
            h_scale=result.x[3]
        )
    else:
        opt_params = GalaxyParameters(
            ML_disk=result.x[0],
            ML_bulge=0.7,
            gas_factor=result.x[1],
            h_scale=result.x[2]
        )
    
    return opt_params, result.fun


def evaluate_global_params(params: Dict, galaxies: List[GalaxyData], 
                          quick_mode: bool = False) -> float:
    """Evaluate global parameters on all galaxies"""
    
    # Create solver
    solver = EnhancedGravitySolver(
        lambda_eff=params['lambda_eff'],
        beta_scale=params['beta_scale'],
        mu_scale=params['mu_scale'],
        coupling_scale=params['coupling_scale']
    )
    
    # Evaluate on each galaxy
    chi2_values = []
    
    if quick_mode:
        # Quick mode: no per-galaxy optimization
        for galaxy in galaxies:
            try:
                result = solver.solve_galaxy(galaxy)
                chi2_values.append(result['chi2_reduced'])
            except:
                chi2_values.append(1000)
    else:
        # Full mode: optimize per-galaxy parameters
        def process_galaxy(galaxy):
            try:
                opt_params, chi2 = optimize_galaxy_params(galaxy, solver)
                return chi2
            except:
                return 1000
        
        # Parallel processing
        chi2_values = joblib.Parallel(n_jobs=N_JOBS)(
            joblib.delayed(process_galaxy)(galaxy) for galaxy in galaxies
        )
    
    # Return median chi2
    return np.median(chi2_values)


class OptimizationCallback:
    """Callback to track optimization progress"""
    def __init__(self):
        self.best_value = float('inf')
        self.best_params = None
        self.history = []
    
    def __call__(self, study, trial):
        if trial.value < self.best_value:
            self.best_value = trial.value
            self.best_params = trial.params
            print(f"  New best: median χ²/N = {trial.value:.3f}")
        
        self.history.append({
            'trial': trial.number,
            'value': trial.value,
            'params': trial.params
        })


def run_bayesian_optimization(galaxies: List[GalaxyData], n_trials: int = 100):
    """Run full Bayesian optimization"""
    
    print(f"Starting Bayesian optimization with {len(galaxies)} galaxies...")
    print(f"Running {n_trials} trials on {joblib.cpu_count()} cores")
    
    # Define objective function for Optuna
    def objective(trial):
        # Sample global parameters
        params = {
            'lambda_eff': trial.suggest_float('lambda_eff', *GLOBAL_BOUNDS['lambda_eff']),
            'beta_scale': trial.suggest_float('beta_scale', *GLOBAL_BOUNDS['beta_scale']),
            'mu_scale': trial.suggest_float('mu_scale', *GLOBAL_BOUNDS['mu_scale']),
            'coupling_scale': trial.suggest_float('coupling_scale', *GLOBAL_BOUNDS['coupling_scale'])
        }
        
        # Quick evaluation for first 20 trials, then full
        quick_mode = trial.number < 20
        return evaluate_global_params(params, galaxies, quick_mode)
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Add callback
    callback = OptimizationCallback()
    
    # Run optimization
    start_time = datetime.now()
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    end_time = datetime.now()
    
    print(f"\nOptimization completed in {(end_time - start_time).total_seconds():.1f} seconds")
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\nBest median χ²/N: {best_value:.3f}")
    print("Best parameters:")
    for key, value in best_params.items():
        if key == 'lambda_eff':
            print(f"  {key}: {value*1e6:.1f} μm")
        else:
            print(f"  {key}: {value:.3f}")
    
    return study, callback.history


def full_analysis_with_best_params(galaxies: List[GalaxyData], 
                                  global_params: Dict) -> pd.DataFrame:
    """Run full analysis with best parameters and per-galaxy optimization"""
    
    print("\nRunning full analysis with optimized parameters...")
    
    # Create solver
    solver = EnhancedGravitySolver(**global_params)
    
    results = []
    
    for i, galaxy in enumerate(galaxies):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(galaxies)} galaxies...")
        
        try:
            # Optimize per-galaxy parameters
            opt_params, chi2_opt = optimize_galaxy_params(galaxy, solver, max_iter=50)
            
            # Get full results
            result = solver.solve_galaxy(galaxy, opt_params)
            
            results.append({
                'name': galaxy.name,
                'chi2_reduced': result['chi2_reduced'],
                'chi2_reduced_default': solver.solve_galaxy(galaxy)['chi2_reduced'],
                'improvement': solver.solve_galaxy(galaxy)['chi2_reduced'] / result['chi2_reduced'],
                'ML_disk': opt_params.ML_disk,
                'ML_bulge': opt_params.ML_bulge,
                'gas_factor': opt_params.gas_factor,
                'h_scale': opt_params.h_scale,
                'n_points': len(galaxy.v_obs),
                'max_radius': max(galaxy.R_kpc),
                'has_bulge': galaxy.sigma_bulge is not None
            })
        except Exception as e:
            print(f"    Error with {galaxy.name}: {e}")
            results.append({
                'name': galaxy.name,
                'chi2_reduced': np.nan,
                'chi2_reduced_default': np.nan,
                'improvement': np.nan
            })
    
    return pd.DataFrame(results)


def create_optimization_plots(history: List[Dict], results_df: pd.DataFrame):
    """Create visualization of optimization results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Optimization history
    trials = [h['trial'] for h in history]
    values = [h['value'] for h in history]
    ax1.plot(trials, values, 'b-', alpha=0.7)
    ax1.scatter(trials, values, c=values, cmap='viridis', s=30)
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Median χ²/N')
    ax1.set_title('Bayesian Optimization Progress')
    ax1.grid(True, alpha=0.3)
    
    # 2. Parameter evolution
    param_names = list(GLOBAL_BOUNDS.keys())
    for i, param in enumerate(param_names):
        param_values = [h['params'][param] for h in history]
        if param == 'lambda_eff':
            param_values = [v*1e6 for v in param_values]  # Convert to μm
        ax2.scatter(trials, param_values, label=param, alpha=0.6, s=20)
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Parameter Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Final chi2 distribution
    valid_results = results_df[~results_df['chi2_reduced'].isna()]
    ax3.hist(valid_results['chi2_reduced'], bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(valid_results['chi2_reduced'].median(), color='r', linestyle='--',
               label=f"Median = {valid_results['chi2_reduced'].median():.2f}")
    ax3.set_xlabel('χ²/N')
    ax3.set_ylabel('Number of Galaxies')
    ax3.set_title('Final χ² Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, min(100, valid_results['chi2_reduced'].quantile(0.95)))
    
    # 4. Improvement histogram
    improvements = valid_results['improvement']
    improvements_clipped = improvements[improvements < 10]  # Clip extreme values
    ax4.hist(improvements_clipped, bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(improvements_clipped.median(), color='r', linestyle='--',
               label=f"Median = {improvements_clipped.median():.2f}x")
    ax4.set_xlabel('Improvement Factor')
    ax4.set_ylabel('Number of Galaxies')
    ax4.set_title('Per-Galaxy Parameter Optimization Impact')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'optimization_summary.png'), dpi=300)
    plt.show()


def main():
    """Main optimization routine"""
    
    print("Recognition Science Bayesian Optimization")
    print("="*60)
    
    # Load galaxies
    print(f"\nLoading galaxies from {DATA_DIR}...")
    galaxy_files = glob.glob(os.path.join(DATA_DIR, "*_rotmod.dat"))
    if MAX_GALAXIES:
        galaxy_files = galaxy_files[:MAX_GALAXIES]
    
    galaxies = []
    for filepath in galaxy_files:
        galaxy = load_galaxy_from_file(filepath)
        if galaxy is not None:
            galaxies.append(galaxy)
    
    print(f"Loaded {len(galaxies)} galaxies successfully")
    
    if len(galaxies) == 0:
        print("No galaxies loaded! Check data directory.")
        return
    
    # Run Bayesian optimization
    study, history = run_bayesian_optimization(galaxies, N_TRIALS)
    
    # Get best parameters
    best_global_params = study.best_params
    
    # Run full analysis with best parameters
    results_df = full_analysis_with_best_params(galaxies, best_global_params)
    
    # Save results
    optimization_results = {
        'best_global_parameters': best_global_params,
        'best_median_chi2': study.best_value,
        'n_galaxies': len(galaxies),
        'n_trials': N_TRIALS,
        'timestamp': datetime.now().isoformat(),
        'optimization_history': history,
        'final_statistics': {
            'mean_chi2': results_df['chi2_reduced'].mean(),
            'median_chi2': results_df['chi2_reduced'].median(),
            'std_chi2': results_df['chi2_reduced'].std(),
            'frac_below_5': (results_df['chi2_reduced'] < 5).sum() / len(results_df),
            'frac_below_2': (results_df['chi2_reduced'] < 2).sum() / len(results_df),
            'median_improvement': results_df['improvement'].median()
        }
    }
    
    # Save to JSON
    with open(os.path.join(RESULTS_DIR, 'bayesian_optimization_results.json'), 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    # Save detailed results
    results_df.to_csv(os.path.join(RESULTS_DIR, 'galaxy_parameters_optimized.csv'), index=False)
    
    # Create plots
    create_optimization_plots(history, results_df)
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY:")
    print("="*60)
    print(f"Best median χ²/N: {study.best_value:.3f}")
    print(f"Final mean χ²/N: {results_df['chi2_reduced'].mean():.3f}")
    print(f"Final median χ²/N: {results_df['chi2_reduced'].median():.3f}")
    print(f"Galaxies with χ²/N < 5: {(results_df['chi2_reduced'] < 5).sum()}/{len(results_df)} ({100*(results_df['chi2_reduced'] < 5).sum()/len(results_df):.1f}%)")
    print(f"Galaxies with χ²/N < 2: {(results_df['chi2_reduced'] < 2).sum()}/{len(results_df)} ({100*(results_df['chi2_reduced'] < 2).sum()/len(results_df):.1f}%)")
    print(f"Median improvement from per-galaxy optimization: {results_df['improvement'].median():.2f}x")
    
    print(f"\nResults saved to {RESULTS_DIR}/")
    
    # Save best parameter file for future use
    best_params_file = {
        'global_parameters': best_global_params,
        'description': 'Optimized parameters from Bayesian search',
        'performance': {
            'median_chi2': results_df['chi2_reduced'].median(),
            'mean_chi2': results_df['chi2_reduced'].mean()
        }
    }
    
    with open('best_parameters.json', 'w') as f:
        json.dump(best_params_file, f, indent=2)
    
    print("\nBest parameters saved to best_parameters.json")


if __name__ == "__main__":
    main() 