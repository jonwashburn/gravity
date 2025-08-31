#!/usr/bin/env python3
import pickle
import numpy as np

# Load results from our best solver
with open('lnal_advanced_v2_results.pkl', 'rb') as f:
    results = pickle.load(f)

print('Analyzing percentage errors from lnal_advanced_solver_v2...')
print()

all_percent_errors = []
galaxy_errors = []

for result in results['individual']:
    v_obs = result['v_obs']
    v_model = result['v_model']
    
    # Calculate percentage error for each data point
    percent_errors = 100 * np.abs(v_obs - v_model) / v_obs
    
    # Store individual galaxy average error
    avg_error = np.mean(percent_errors)
    galaxy_errors.append(avg_error)
    
    # Store all point errors
    all_percent_errors.extend(percent_errors)

all_percent_errors = np.array(all_percent_errors)
galaxy_errors = np.array(galaxy_errors)

print(f'PER-POINT ANALYSIS:')
print(f'  Total data points: {len(all_percent_errors)}')
print(f'  Mean error: {np.mean(all_percent_errors):.1f}%')
print(f'  Median error: {np.median(all_percent_errors):.1f}%')
print(f'  RMS error: {np.sqrt(np.mean(all_percent_errors**2)):.1f}%')
print(f'  Range: {np.min(all_percent_errors):.1f}% - {np.max(all_percent_errors):.1f}%')
print()

print(f'PER-GALAXY ANALYSIS:')
print(f'  Galaxies analyzed: {len(galaxy_errors)}')
print(f'  Mean galaxy error: {np.mean(galaxy_errors):.1f}%')
print(f'  Median galaxy error: {np.median(galaxy_errors):.1f}%')
print(f'  Best galaxy: {np.min(galaxy_errors):.1f}%')
print(f'  Worst galaxy: {np.max(galaxy_errors):.1f}%')
print()

print(f'ERROR DISTRIBUTION:')
print(f'  < 10% error: {np.mean(all_percent_errors < 10):.1%}')
print(f'  < 20% error: {np.mean(all_percent_errors < 20):.1%}')
print(f'  < 50% error: {np.mean(all_percent_errors < 50):.1%}')
print(f'  > 100% error: {np.mean(all_percent_errors > 100):.1%}')

print()
print(f'CHI-SQUARED CONTEXT:')
print(f'  Mean χ²/N: {results["chi2_mean"]:.2f}')
print(f'  Median χ²/N: {results["chi2_median"]:.2f}')
print(f'  Best χ²/N: {min(results["chi2_values"]):.2f}') 