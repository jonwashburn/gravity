#!/usr/bin/env python3
"""
Simple test of the cosmic ledger hypothesis using existing results
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load results
print("Loading SPARC unified results...")
with open('sparc_unified_results.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['results']
print(f"Loaded {len(results)} galaxies")

# Extract scale factors and calculate deviations
galaxies = []
for name, result in results.items():
    if 'best_params' in result and result['chi2'] < 100:  # Good fits only
        scale = result['best_params']['scale']
        delta = (scale - 1.0) * 100  # Convert to percentage
        
        # Get galaxy properties
        galaxy_data = result.get('galaxy_data', {})
        Mgas = galaxy_data.get('Mgas', 0)
        Mstar = galaxy_data.get('Mstar', 1)
        
        if Mstar > 0:
            f_gas = Mgas / (Mgas + Mstar)
            galaxies.append({
                'name': name,
                'scale': scale,
                'delta': delta,
                'f_gas': f_gas,
                'Mstar': Mstar,
                'chi2': result['chi2']
            })

print(f"Analyzed {len(galaxies)} galaxies with good fits")

# Convert to arrays
deltas = np.array([g['delta'] for g in galaxies])
f_gas = np.array([g['f_gas'] for g in galaxies])
log_Mstar = np.log10([g['Mstar'] for g in galaxies])

# Print key statistics
print("\n=== KEY STATISTICS ===")
print(f"Mean δ: {deltas.mean():.2f}%")
print(f"Median δ: {np.median(deltas):.2f}%")
print(f"Minimum δ: {deltas.min():.2f}%")
print(f"Maximum δ: {deltas.max():.2f}%")
print(f"Galaxies with δ < 0: {(deltas < 0).sum()} ({(deltas < 0).sum()/len(deltas)*100:.1f}%)")
print(f"Galaxies with δ < -0.5%: {(deltas < -0.5).sum()} ({(deltas < -0.5).sum()/len(deltas)*100:.1f}%)")

# Test the wedge
print("\n=== WEDGE TEST ===")
# Bin by gas fraction
bins = np.linspace(0, 1, 11)
min_by_bin = []
max_by_bin = []
mean_by_bin = []
bin_centers = []

for i in range(len(bins)-1):
    mask = (f_gas >= bins[i]) & (f_gas < bins[i+1])
    if mask.sum() > 2:
        min_by_bin.append(deltas[mask].min())
        max_by_bin.append(deltas[mask].max())
        mean_by_bin.append(deltas[mask].mean())
        bin_centers.append((bins[i] + bins[i+1])/2)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: δ vs gas fraction with envelope
ax = axes[0, 0]
ax.scatter(f_gas, deltas, alpha=0.5, s=30, c='blue')
if len(bin_centers) > 0:
    ax.plot(bin_centers, min_by_bin, 'r-', linewidth=2, label='Min envelope')
    ax.plot(bin_centers, max_by_bin, 'g-', linewidth=2, label='Max envelope')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Gas Fraction')
ax.set_ylabel('δ (%)')
ax.set_title('Testing One-Sided Wedge Prediction')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Distribution
ax = axes[0, 1]
ax.hist(deltas, bins=30, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero')
ax.axvline(x=deltas.mean(), color='blue', linestyle='--', alpha=0.5, label=f'Mean = {deltas.mean():.2f}%')
ax.set_xlabel('δ (%)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Scale Factor Deviations')
ax.legend()

# Plot 3: δ vs stellar mass
ax = axes[1, 0]
ax.scatter(log_Mstar, deltas, alpha=0.5, s=30, c='green')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('log(M*/M☉)')
ax.set_ylabel('δ (%)')
ax.set_title('Check for Mass Dependence')
ax.grid(True, alpha=0.3)

# Plot 4: Gas-rich vs gas-poor
ax = axes[1, 1]
gas_rich = deltas[f_gas > 0.5]
gas_poor = deltas[f_gas < 0.1]
if len(gas_rich) > 0 and len(gas_poor) > 0:
    ax.boxplot([gas_poor, gas_rich], labels=['Gas-poor\n(f<0.1)', 'Gas-rich\n(f>0.5)'])
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('δ (%)')
    ax.set_title('Efficiency Test')
    
    print(f"\nGas-poor galaxies: mean δ = {gas_poor.mean():.2f}% (N={len(gas_poor)})")
    print(f"Gas-rich galaxies: mean δ = {gas_rich.mean():.2f}% (N={len(gas_rich)})")

plt.tight_layout()
plt.savefig('ledger_test_simple.png', dpi=150, bbox_inches='tight')
plt.show()

# Final verdict
print("\n=== VERDICT ===")
if deltas.min() > -1:
    print("✓ Lower bound test PASSED: no galaxies significantly below zero")
else:
    print("✗ Lower bound test FAILED: found galaxies well below zero")

if len(gas_rich) > 0 and len(gas_poor) > 0 and gas_rich.mean() > gas_poor.mean():
    print("✓ Inefficiency correlation CONFIRMED: gas-rich galaxies have higher δ")
else:
    print("✗ Inefficiency correlation NOT CONFIRMED")

# Check for obvious systematics
from scipy import stats
if len(log_Mstar) > 10:
    r, p = stats.spearmanr(log_Mstar, deltas)
    print(f"\nMass correlation: r={r:.3f}, p={p:.3e}")
    if abs(r) < 0.3:
        print("✓ No strong mass dependence (good!)")
    else:
        print("✗ Warning: significant mass dependence found") 