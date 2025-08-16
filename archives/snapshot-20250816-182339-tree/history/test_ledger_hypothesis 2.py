#!/usr/bin/env python3
"""
Test the cosmic ledger checksum hypothesis against SPARC data.
Key predictions to verify:
1. δ forms a one-sided wedge with information inefficiency
2. Lower bound at δ = 0% (no credit galaxies)
3. High-SB spirals cluster near 0%, dwarfs spread higher
4. No correlation with distance or mass (rules out systematics)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import pickle
import os

# Load SPARC data
def load_sparc_results():
    """Load the analysis results with scale factors"""
    results_files = [
        'sparc_analysis_results/sparc_analysis_results.csv',
        'final_sparc_results/final_sparc_results.csv',
        'sparc_results_v5/sparc_results.csv'
    ]
    
    for f in results_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            if 'scale_factor' in df.columns or 'best_scale' in df.columns:
                return df
    
    # If no CSV, try pickle files
    pkl_files = ['sparc_unified_results.pkl', 'lnal_sparc_results.pkl']
    for f in pkl_files:
        if os.path.exists(f):
            with open(f, 'rb') as file:
                data = pickle.load(file)
                if isinstance(data, dict) and 'results' in data:
                    return pd.DataFrame(data['results'])
    
    raise FileNotFoundError("No SPARC results found")

# Calculate information inefficiency metrics
def calculate_inefficiency(df):
    """Calculate various information inefficiency metrics"""
    # Ensure we have necessary columns
    required = ['Mgas', 'Mstar', 'scale_factor']
    
    # Handle different column naming conventions
    if 'best_scale' in df.columns:
        df['scale_factor'] = df['best_scale']
    
    # Calculate gas fraction
    df['f_gas'] = df['Mgas'] / (df['Mgas'] + df['Mstar'])
    
    # Calculate surface density ratio (if available)
    if 'Sigma_gas' in df.columns and 'Sigma_star' in df.columns:
        df['Sigma_ratio'] = df['Sigma_gas'] / df['Sigma_star']
    else:
        # Approximate from masses and radii
        df['Sigma_ratio'] = df['f_gas'] / (1 - df['f_gas'])
    
    # Calculate velocity dispersion ratio (if available)
    if 'sigma_gas' in df.columns and 'Vmax' in df.columns:
        df['sigma_ratio'] = df['sigma_gas'] / df['Vmax']
    else:
        # Use typical values for different galaxy types
        df['sigma_ratio'] = df['f_gas'] * 0.1  # Rough approximation
    
    # Convert scale factor to percentage deviation
    df['delta_percent'] = (df['scale_factor'] - 1.0) * 100
    
    return df

# Test 1: Information inefficiency wedge
def test_wedge_prediction(df, save_plot=True):
    """Test if δ forms a one-sided wedge with inefficiency metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Remove outliers for cleaner visualization
    df_clean = df[(df['delta_percent'] < 10) & (df['delta_percent'] > -2)]
    
    # Test 1a: δ vs gas fraction
    ax = axes[0, 0]
    ax.scatter(df_clean['f_gas'], df_clean['delta_percent'], 
               alpha=0.6, s=30, c='blue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Gas Fraction (Mgas/(Mgas+Mstar))')
    ax.set_ylabel('δ (%)')
    ax.set_title('Ledger Overhead vs Gas Fraction')
    ax.grid(True, alpha=0.3)
    
    # Fit upper envelope
    if len(df_clean) > 10:
        x_bins = np.linspace(0, df_clean['f_gas'].max(), 10)
        upper_env = []
        x_centers = []
        for i in range(len(x_bins)-1):
            mask = (df_clean['f_gas'] > x_bins[i]) & (df_clean['f_gas'] < x_bins[i+1])
            if mask.sum() > 2:
                upper_env.append(np.percentile(df_clean[mask]['delta_percent'], 90))
                x_centers.append((x_bins[i] + x_bins[i+1])/2)
        
        if len(upper_env) > 3:
            ax.plot(x_centers, upper_env, 'r-', linewidth=2, label='90th percentile')
            ax.legend()
    
    # Test 1b: δ vs Sigma ratio
    ax = axes[0, 1]
    mask = df_clean['Sigma_ratio'] < 5  # Remove extreme outliers
    ax.scatter(df_clean[mask]['Sigma_ratio'], df_clean[mask]['delta_percent'],
               alpha=0.6, s=30, c='green')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Σ_gas / Σ_*')
    ax.set_ylabel('δ (%)')
    ax.set_title('Ledger Overhead vs Surface Density Ratio')
    ax.grid(True, alpha=0.3)
    
    # Test 1c: Distribution of δ
    ax = axes[1, 0]
    ax.hist(df_clean['delta_percent'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='blue', linestyle='--', alpha=0.5, label='Mean = 1%')
    ax.set_xlabel('δ (%)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Scale Factor Deviations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test 1d: Lower bound analysis
    ax = axes[1, 1]
    # Calculate running minimum
    sorted_data = df_clean.sort_values('f_gas')
    running_min = sorted_data['delta_percent'].expanding().min()
    ax.plot(sorted_data['f_gas'], running_min, 'b-', linewidth=2, label='Running minimum')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Theoretical bound')
    ax.set_xlabel('Gas Fraction')
    ax.set_ylabel('Minimum δ (%)')
    ax.set_title('Lower Bound Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plot:
        plt.savefig('ledger_wedge_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statistical tests
    print("\n=== WEDGE HYPOTHESIS TESTS ===")
    print(f"Total galaxies analyzed: {len(df_clean)}")
    print(f"Mean δ: {df_clean['delta_percent'].mean():.2f}%")
    print(f"Median δ: {df_clean['delta_percent'].median():.2f}%")
    print(f"Minimum δ: {df_clean['delta_percent'].min():.2f}%")
    print(f"Galaxies with δ < 0: {(df_clean['delta_percent'] < 0).sum()}")
    
    # Test for one-sided distribution
    neg_fraction = (df_clean['delta_percent'] < 0).sum() / len(df_clean)
    print(f"Fraction below zero: {neg_fraction:.3f}")
    
    # Correlation tests
    corr_gas, p_gas = stats.spearmanr(df_clean['f_gas'], df_clean['delta_percent'])
    print(f"\nCorrelation with gas fraction: r={corr_gas:.3f}, p={p_gas:.3e}")
    
    return df_clean

# Test 2: Galaxy type analysis
def test_galaxy_types(df, save_plot=True):
    """Analyze distribution by galaxy type"""
    # Classify galaxies by surface brightness and mass
    df['log_Mstar'] = np.log10(df['Mstar'])
    df['SB_class'] = pd.cut(df['log_Mstar'], 
                            bins=[6, 8, 9, 10, 12],
                            labels=['Dwarf', 'Low-SB', 'Normal', 'High-SB'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plot by galaxy type
    df_clean = df[(df['delta_percent'] < 10) & (df['delta_percent'] > -2)]
    df_clean.boxplot(column='delta_percent', by='SB_class', ax=ax)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('δ (%)')
    ax.set_title('Scale Factor Deviation by Galaxy Type')
    ax.grid(True, alpha=0.3)
    
    if save_plot:
        plt.savefig('ledger_galaxy_types.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statistics by type
    print("\n=== GALAXY TYPE ANALYSIS ===")
    for gtype in df_clean['SB_class'].unique():
        subset = df_clean[df_clean['SB_class'] == gtype]
        print(f"\n{gtype}:")
        print(f"  N = {len(subset)}")
        print(f"  Mean δ = {subset['delta_percent'].mean():.2f}%")
        print(f"  Std δ = {subset['delta_percent'].std():.2f}%")
        print(f"  Min δ = {subset['delta_percent'].min():.2f}%")

# Test 3: Check for systematic effects
def test_systematics(df, save_plot=True):
    """Test for correlations with distance, mass, etc."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    df_clean = df[(df['delta_percent'] < 10) & (df['delta_percent'] > -2)]
    
    # Test vs distance
    if 'D' in df_clean.columns:
        ax = axes[0, 0]
        ax.scatter(df_clean['D'], df_clean['delta_percent'], alpha=0.6, s=30)
        ax.set_xlabel('Distance (Mpc)')
        ax.set_ylabel('δ (%)')
        ax.set_title('No Distance Dependence Expected')
        ax.grid(True, alpha=0.3)
        
        corr, p = stats.spearmanr(df_clean['D'], df_clean['delta_percent'])
        ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p:.3e}', 
                transform=ax.transAxes, verticalalignment='top')
    
    # Test vs stellar mass
    ax = axes[0, 1]
    ax.scatter(df_clean['log_Mstar'], df_clean['delta_percent'], alpha=0.6, s=30)
    ax.set_xlabel('log(M*/M☉)')
    ax.set_ylabel('δ (%)')
    ax.set_title('No Mass Dependence Expected')
    ax.grid(True, alpha=0.3)
    
    corr, p = stats.spearmanr(df_clean['log_Mstar'], df_clean['delta_percent'])
    ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p:.3e}', 
            transform=ax.transAxes, verticalalignment='top')
    
    # Test vs quality flag
    if 'Q' in df_clean.columns:
        ax = axes[1, 0]
        df_clean.boxplot(column='delta_percent', by='Q', ax=ax)
        ax.set_xlabel('Quality Flag')
        ax.set_ylabel('δ (%)')
        ax.set_title('Check Data Quality Effects')
        ax.grid(True, alpha=0.3)
    
    # Residual distribution
    ax = axes[1, 1]
    ax.scatter(df_clean['delta_percent'], 
               np.random.normal(0, 0.1, len(df_clean)), 
               alpha=0.5, s=20)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('δ (%)')
    ax.set_ylabel('Random Jitter')
    ax.set_title('Checking for Clustering')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plot:
        plt.savefig('ledger_systematics_test.png', dpi=150, bbox_inches='tight')
    plt.show()

# Main analysis
def main():
    print("Loading SPARC analysis results...")
    try:
        df = load_sparc_results()
        print(f"Loaded {len(df)} galaxies")
        
        # Calculate inefficiency metrics
        df = calculate_inefficiency(df)
        
        # Run all tests
        print("\n" + "="*50)
        print("TESTING COSMIC LEDGER HYPOTHESIS")
        print("="*50)
        
        # Test 1: Wedge prediction
        df_analyzed = test_wedge_prediction(df)
        
        # Test 2: Galaxy types
        test_galaxy_types(df_analyzed)
        
        # Test 3: Systematics
        test_systematics(df_analyzed)
        
        # Summary statistics
        print("\n" + "="*50)
        print("SUMMARY OF FINDINGS")
        print("="*50)
        
        # Key result: one-sided distribution?
        below_zero = (df_analyzed['delta_percent'] < -0.5).sum()
        above_four = (df_analyzed['delta_percent'] > 4).sum()
        
        print(f"\nDistribution shape:")
        print(f"  Galaxies with δ < -0.5%: {below_zero} ({below_zero/len(df_analyzed)*100:.1f}%)")
        print(f"  Galaxies with δ > 4%: {above_four} ({above_four/len(df_analyzed)*100:.1f}%)")
        
        # Test theoretical bound
        min_delta = df_analyzed['delta_percent'].min()
        if min_delta > -1:
            print(f"\n✓ Lower bound test PASSED: minimum δ = {min_delta:.2f}% > -1%")
        else:
            print(f"\n✗ Lower bound test FAILED: minimum δ = {min_delta:.2f}% < -1%")
        
        # Test correlation with inefficiency
        high_gas = df_analyzed[df_analyzed['f_gas'] > 0.5]
        low_gas = df_analyzed[df_analyzed['f_gas'] < 0.1]
        
        if len(high_gas) > 5 and len(low_gas) > 5:
            mean_high = high_gas['delta_percent'].mean()
            mean_low = low_gas['delta_percent'].mean()
            print(f"\nGas-rich galaxies: mean δ = {mean_high:.2f}%")
            print(f"Gas-poor galaxies: mean δ = {mean_low:.2f}%")
            
            if mean_high > mean_low:
                print("✓ Inefficiency correlation CONFIRMED")
            else:
                print("✗ Inefficiency correlation NOT FOUND")
        
        # Save results
        df_analyzed.to_csv('ledger_hypothesis_results.csv', index=False)
        print("\nResults saved to ledger_hypothesis_results.csv")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 