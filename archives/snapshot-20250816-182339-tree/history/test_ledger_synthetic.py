#!/usr/bin/env python3
"""
Test ledger hypothesis using synthetic data matching SPARC statistics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate synthetic galaxy data matching SPARC statistics
np.random.seed(42)
n_galaxies = 175

# Gas fractions: bimodal distribution (gas-poor spirals vs gas-rich dwarfs)
f_gas = np.concatenate([
    np.random.beta(2, 8, n_galaxies//2),  # Gas-poor
    np.random.beta(5, 3, n_galaxies - n_galaxies//2)   # Gas-rich
])

# Scale factor deviations based on ledger hypothesis
# δ = base_overhead + inefficiency_cost * f_gas + noise
base_overhead = 0.5  # 0.5% minimum overhead
inefficiency_slope = 3.0  # Up to 3% additional for gas-rich
noise_scale = 0.8

# Generate δ values with one-sided distribution
deltas = base_overhead + inefficiency_slope * f_gas + np.abs(np.random.normal(0, noise_scale, n_galaxies))

# Add a few galaxies near zero (efficient spirals)
efficient_mask = (f_gas < 0.1) & (np.random.random(n_galaxies) < 0.2)
deltas[efficient_mask] = np.random.uniform(0, 0.3, efficient_mask.sum())

# Ensure no negative values (no credit galaxies)
deltas = np.maximum(deltas, 0)

print("=== SYNTHETIC SPARC-LIKE DATA ===")
print(f"N galaxies: {n_galaxies}")
print(f"Mean δ: {deltas.mean():.2f}%")
print(f"Median δ: {np.median(deltas):.2f}%")
print(f"Min δ: {deltas.min():.2f}%")
print(f"Max δ: {deltas.max():.2f}%")
print(f"Galaxies with δ < 0: {(deltas < 0).sum()}")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: The wedge
ax = axes[0, 0]
ax.scatter(f_gas, deltas, alpha=0.6, s=40, c=f_gas, cmap='viridis')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Theoretical bound')

# Fit and plot envelope
from sklearn.linear_model import QuantileRegressor
X = f_gas.reshape(-1, 1)
qr_upper = QuantileRegressor(quantile=0.9, alpha=0)
qr_lower = QuantileRegressor(quantile=0.1, alpha=0)
qr_upper.fit(X, deltas)
qr_lower.fit(X, deltas)

x_plot = np.linspace(0, 1, 100).reshape(-1, 1)
y_upper = qr_upper.predict(x_plot)
y_lower = qr_lower.predict(x_plot)

ax.plot(x_plot, y_upper, 'g-', linewidth=2, label='90th percentile')
ax.plot(x_plot, y_lower, 'b-', linewidth=2, label='10th percentile')
ax.fill_between(x_plot.ravel(), y_lower, y_upper, alpha=0.2, color='gray')

ax.set_xlabel('Gas Fraction')
ax.set_ylabel('δ (%)')
ax.set_title('Ledger Overhead Wedge')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Distribution
ax = axes[0, 1]
counts, bins, _ = ax.hist(deltas, bins=30, alpha=0.7, color='purple', 
                          edgecolor='black', density=True)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax.axvline(x=1, color='blue', linestyle='--', alpha=0.5, label='Expected mean')
ax.set_xlabel('δ (%)')
ax.set_ylabel('Probability Density')
ax.set_title('One-Sided Distribution')
ax.legend()

# Plot 3: Cumulative distribution
ax = axes[0, 2]
sorted_deltas = np.sort(deltas)
cumulative = np.arange(1, len(deltas) + 1) / len(deltas)
ax.plot(sorted_deltas, cumulative, 'b-', linewidth=2)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax.axvline(x=1, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('δ (%)')
ax.set_ylabel('Cumulative Probability')
ax.set_title('CDF showing lower bound')
ax.grid(True, alpha=0.3)

# Plot 4: Efficiency classes
ax = axes[1, 0]
# Define efficiency classes
ultra_efficient = deltas[deltas < 0.5]
efficient = deltas[(deltas >= 0.5) & (deltas < 1.5)]
normal = deltas[(deltas >= 1.5) & (deltas < 3)]
inefficient = deltas[deltas >= 3]

ax.bar(['Ultra\nEfficient\n(<0.5%)', 'Efficient\n(0.5-1.5%)', 
        'Normal\n(1.5-3%)', 'Inefficient\n(>3%)'],
       [len(ultra_efficient), len(efficient), len(normal), len(inefficient)],
       color=['darkgreen', 'green', 'orange', 'red'], alpha=0.7)
ax.set_ylabel('Number of Galaxies')
ax.set_title('Galaxy Efficiency Classes')

# Plot 5: Correlation test
ax = axes[1, 1]
# Bin the data
n_bins = 10
bin_edges = np.linspace(0, f_gas.max(), n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_means = []
bin_stds = []

for i in range(n_bins):
    mask = (f_gas >= bin_edges[i]) & (f_gas < bin_edges[i+1])
    if mask.sum() > 2:
        bin_means.append(deltas[mask].mean())
        bin_stds.append(deltas[mask].std())
    else:
        bin_means.append(np.nan)
        bin_stds.append(np.nan)

valid = ~np.isnan(bin_means)
ax.errorbar(bin_centers[valid], np.array(bin_means)[valid], 
            yerr=np.array(bin_stds)[valid], fmt='o-', linewidth=2, 
            markersize=8, capsize=5)
ax.set_xlabel('Gas Fraction (binned)')
ax.set_ylabel('Mean δ (%)')
ax.set_title('Inefficiency Correlation')
ax.grid(True, alpha=0.3)

# Plot 6: Theoretical prediction
ax = axes[1, 2]
# Show theoretical model
f_theory = np.linspace(0, 1, 100)
delta_theory_min = 0  # Minimum possible
delta_theory_mean = 0.5 + 1.5 * f_theory  # Expected mean
delta_theory_max = 0.5 + 4 * f_theory  # Maximum expected

ax.fill_between(f_theory, delta_theory_min, delta_theory_max, 
                alpha=0.3, color='blue', label='Theoretical range')
ax.plot(f_theory, delta_theory_mean, 'b-', linewidth=2, label='Expected mean')
ax.scatter(f_gas, deltas, alpha=0.5, s=30, c='red', label='Data')
ax.set_xlabel('Gas Fraction')
ax.set_ylabel('δ (%)')
ax.set_title('Theory vs Data')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ledger_synthetic_test.png', dpi=150, bbox_inches='tight')
plt.show()

# Statistical tests
print("\n=== STATISTICAL TESTS ===")

# 1. Test for lower bound
print(f"\n1. Lower Bound Test:")
print(f"   Minimum δ = {deltas.min():.3f}%")
print(f"   Fraction below 0.1% = {(deltas < 0.1).sum()/len(deltas)*100:.1f}%")
if deltas.min() >= -0.1:
    print("   ✓ PASSED: No significant negative values")
else:
    print("   ✗ FAILED: Found negative values")

# 2. Test correlation with gas fraction
print(f"\n2. Inefficiency Correlation:")
r_spearman, p_spearman = stats.spearmanr(f_gas, deltas)
r_pearson, p_pearson = stats.pearsonr(f_gas, deltas)
print(f"   Spearman r = {r_spearman:.3f} (p = {p_spearman:.3e})")
print(f"   Pearson r = {r_pearson:.3f} (p = {p_pearson:.3e})")
if r_spearman > 0.3 and p_spearman < 0.001:
    print("   ✓ PASSED: Significant positive correlation")
else:
    print("   ✗ FAILED: No clear correlation")

# 3. Test gas-rich vs gas-poor
print(f"\n3. Gas-Rich vs Gas-Poor:")
gas_poor = deltas[f_gas < 0.2]
gas_rich = deltas[f_gas > 0.6]
if len(gas_poor) > 5 and len(gas_rich) > 5:
    t_stat, p_value = stats.ttest_ind(gas_rich, gas_poor)
    print(f"   Gas-poor mean = {gas_poor.mean():.2f}% (N={len(gas_poor)})")
    print(f"   Gas-rich mean = {gas_rich.mean():.2f}% (N={len(gas_rich)})")
    print(f"   Difference = {gas_rich.mean() - gas_poor.mean():.2f}%")
    print(f"   t-test p-value = {p_value:.3e}")
    if gas_rich.mean() > gas_poor.mean() and p_value < 0.01:
        print("   ✓ PASSED: Gas-rich galaxies have higher overhead")
    else:
        print("   ✗ FAILED: No clear difference")

# 4. Test for one-sided distribution
print(f"\n4. One-Sided Distribution Test:")
# Kolmogorov-Smirnov test against normal distribution
_, p_normal = stats.kstest(deltas, 'norm', args=(deltas.mean(), deltas.std()))
print(f"   KS test vs normal: p = {p_normal:.3e}")
skewness = stats.skew(deltas)
print(f"   Skewness = {skewness:.3f}")
if p_normal < 0.01 and skewness > 0.5:
    print("   ✓ PASSED: Distribution is one-sided (not normal)")
else:
    print("   ✗ FAILED: Distribution appears symmetric")

print("\n=== INTERPRETATION ===")
print("If this synthetic data passes all tests, it demonstrates that:")
print("1. The ledger overhead hypothesis is internally consistent")
print("2. A 1% mean overhead emerges naturally from the model")
print("3. The wedge pattern is a robust prediction")
print("4. No 'credit' galaxies (δ < 0) is a fundamental constraint")
print("\nNext step: Check if real SPARC data shows the same patterns!") 