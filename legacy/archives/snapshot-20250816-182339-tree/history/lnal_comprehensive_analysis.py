#!/usr/bin/env python3
"""
Comprehensive analysis of LNAL parameter-free model
Shows progress and identifies remaining issues
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Run analysis with the best parameter-free model
exec(open('lnal_diskmodel_v2.py').read())

# Load results
with open('lnal_diskmodel_v2_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Also load original catalog for correlations
with open('sparc_real_data.pkl', 'rb') as f:
    sparc_data = pickle.load(f)

print("\n=== COMPREHENSIVE LNAL ANALYSIS ===")
print("All components parameter-free from Recognition Science:")
print("- Ξ: Baryon completeness from φ")
print("- Ψ: Information debt with Schwarzschild cutoff")
print("- Λ: MOND interpolation with recognition modulation")
print("- P: Prime sieve factor φ^(-1/2) × 8/π²")
print("- H₂: Metallicity-based with φ^(1/2)/2 exponent")

# Extract properties for correlation analysis
properties = []
for res in results:
    name = res['name']
    if name in sparc_data:
        cat = sparc_data[name]['catalog']
        properties.append({
            'name': name,
            'mean_ratio': res['mean_ratio'],
            'median_ratio': res['median_ratio'],
            'M_star': cat['M_star'],
            'M_HI': cat['M_HI'],
            'f_gas': cat['M_HI'] / (cat['M_star'] + cat['M_HI']) * 1e9,
            'V_flat': cat['V_flat'],
            'quality': cat['quality'],
            'type': cat['type']
        })

# Overall statistics
all_ratios = np.concatenate([r['ratios'] for r in results])
mean_ratios = [p['mean_ratio'] for p in properties]
median_ratios = [p['median_ratio'] for p in properties]

print(f"\nOVERALL PERFORMANCE:")
print(f"Points analyzed: {len(all_ratios)}")
print(f"Galaxies: {len(results)}")
print(f"Mean ratio: {np.mean(mean_ratios):.3f} ± {np.std(mean_ratios):.3f}")
print(f"Median ratio: {np.median(median_ratios):.3f}")
print(f"Success rate (0.8-1.2): {100*np.sum((np.array(mean_ratios) > 0.8) & (np.array(mean_ratios) < 1.2))/len(mean_ratios):.1f}%")

# Correlation analysis
f_gas = [p['f_gas'] for p in properties]
log_M_star = [np.log10(p['M_star']) for p in properties]
types = [p['type'] for p in properties]

r_gas, p_gas = pearsonr(f_gas, mean_ratios)
r_mass, p_mass = pearsonr(log_M_star, mean_ratios)

print(f"\nCORRELATIONS:")
print(f"Gas fraction: r={r_gas:.3f} (p={p_gas:.3e})")
print(f"Stellar mass: r={r_mass:.3f} (p={p_mass:.3e})")

# By galaxy type
print(f"\nBY MORPHOLOGY:")
for t_min, t_max, label in [(0, 3, "Early (S0-Sb)"), (4, 7, "Intermediate (Sbc-Sd)"), (8, 11, "Late (Sdm-Im)")]:
    mask = [(t >= t_min) and (t <= t_max) for t in types]
    if sum(mask) > 0:
        subset = np.array(mean_ratios)[mask]
        print(f"{label}: {np.mean(subset):.3f} ± {np.std(subset):.3f} (n={sum(mask)})")

# Create diagnostic plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Ratio histogram
ax1.hist(mean_ratios, bins=30, alpha=0.7, edgecolor='black')
ax1.axvline(1.0, color='red', linestyle='--', label='Perfect')
ax1.axvline(np.median(median_ratios), color='blue', linestyle='-', 
           label=f'Median={np.median(median_ratios):.3f}')
ax1.set_xlabel('Mean V_model/V_obs per galaxy')
ax1.set_ylabel('Count')
ax1.set_title('Parameter-Free LNAL Performance')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gas fraction correlation
colors = ['red', 'orange', 'green']
for q in [1, 2, 3]:
    mask = [p['quality'] == q for p in properties]
    if sum(mask) > 0:
        ax2.scatter(np.array(f_gas)[mask], np.array(mean_ratios)[mask],
                   c=colors[q-1], label=f'Q={q}', alpha=0.6, s=30)
ax2.set_xlabel('Gas Fraction')
ax2.set_ylabel('V_model/V_obs')
ax2.set_title(f'Gas Correlation (r={r_gas:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Mass correlation  
for q in [1, 2, 3]:
    mask = [p['quality'] == q for p in properties]
    if sum(mask) > 0:
        ax3.scatter(np.array(log_M_star)[mask], np.array(mean_ratios)[mask],
                   c=colors[q-1], label=f'Q={q}', alpha=0.6, s=30)
ax3.set_xlabel('log(M*/M_sun)')
ax3.set_ylabel('V_model/V_obs')
ax3.set_title(f'Mass Correlation (r={r_mass:.3f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Morphology analysis
type_bins = np.arange(-0.5, 12.5, 1)
type_centers = 0.5 * (type_bins[:-1] + type_bins[1:])
mean_by_type = []
std_by_type = []
for i in range(12):
    mask = [t == i for t in types]
    if sum(mask) > 3:
        subset = np.array(mean_ratios)[mask]
        mean_by_type.append(np.mean(subset))
        std_by_type.append(np.std(subset)/np.sqrt(len(subset)))
    else:
        mean_by_type.append(np.nan)
        std_by_type.append(np.nan)

ax4.errorbar(type_centers[:len(mean_by_type)], mean_by_type, yerr=std_by_type, 
             marker='o', capsize=5)
ax4.axhline(1.0, color='red', linestyle='--')
ax4.set_xlabel('Hubble Type')
ax4.set_ylabel('Mean V_model/V_obs')
ax4.set_title('Morphology Dependence')
ax4.set_xticks(range(12))
ax4.set_xticklabels(['S0','Sa','Sab','Sb','Sbc','Sc','Scd','Sd','Sdm','Sm','Im','BCD'])
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lnal_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAnalysis plots saved to:")
print("- lnal_comprehensive_analysis.png")
print("- lnal_diskmodel_v2_examples.png") 