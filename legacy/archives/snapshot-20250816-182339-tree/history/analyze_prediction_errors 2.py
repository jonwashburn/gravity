#!/usr/bin/env python3
"""
Comprehensive error analysis for RS Gravity v5 predictions.
Identifies where predictions fail and suggests improvements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import json

class PredictionErrorAnalyzer:
    def __init__(self):
        self.predictions = {}
        self.observations = {}
        self.errors = {}
        
    def add_laboratory_predictions(self):
        """Laboratory test predictions with observed values where available."""
        self.predictions['laboratory'] = {
            'neon_492nm_shift': {
                'predicted': 0.69,  # Hz
                'observed': None,   # Not yet measured
                'uncertainty': 0.1,
                'units': 'Hz',
                'status': 'untested'
            },
            'cavity_finesse_change': {
                'predicted': 350,
                'observed': None,
                'uncertainty': 50,
                'units': 'dimensionless',
                'status': 'untested'
            },
            'atom_interferometry_phase': {
                'predicted': 2.3e-5,  # rad
                'observed': None,
                'uncertainty': 0.5e-5,
                'units': 'rad',
                'status': 'untested'
            },
            'nanoscale_force_ratio': {
                'predicted': 0.52,  # G_RS/G_Newton at 10nm
                'observed': None,
                'uncertainty': 0.05,
                'units': 'dimensionless',
                'status': 'untested'
            }
        }
        
    def add_astronomical_predictions(self):
        """Astronomical predictions with observed values."""
        self.predictions['astronomical'] = {
            'microlensing_periodicity': {
                'predicted': 13.0,  # days
                'observed': None,   # Need to analyze OGLE data
                'uncertainty': 0.1,
                'units': 'days',
                'status': 'untested'
            },
            'microlensing_amplitude': {
                'predicted': 0.001,  # 0.1% modulation
                'observed': None,
                'uncertainty': 0.0002,
                'units': 'fraction',
                'status': 'untested'
            },
            'draco_velocity_dispersion': {
                'predicted': 5.9,   # km/s
                'observed': 9.1,    # km/s
                'uncertainty': 1.2,
                'units': 'km/s',
                'status': 'failed',
                'chi2': ((9.1 - 5.9) / 1.2)**2
            },
            'fornax_velocity_dispersion': {
                'predicted': 7.2,   # km/s (estimated)
                'observed': 11.7,   # km/s
                'uncertainty': 0.9,
                'units': 'km/s',
                'status': 'failed',
                'chi2': ((11.7 - 7.2) / 0.9)**2
            },
            'sculptor_velocity_dispersion': {
                'predicted': 6.5,   # km/s (estimated)
                'observed': 9.2,    # km/s
                'uncertainty': 1.4,
                'units': 'km/s',
                'status': 'failed',
                'chi2': ((9.2 - 6.5) / 1.4)**2
            },
            'pulsar_timing_deviation': {
                'predicted': 0.056,  # 5.6% from GR
                'observed': 0.0,     # No deviation detected
                'uncertainty': 0.001,
                'units': 'fraction',
                'status': 'failed',
                'chi2': ((0.0 - 0.056) / 0.001)**2
            },
            'strong_lensing_shift': {
                'predicted': 0.045,  # arcsec
                'observed': 0.0,     # No shift detected in SLACS
                'uncertainty': 0.010,
                'units': 'arcsec',
                'status': 'failed',
                'chi2': ((0.0 - 0.045) / 0.010)**2
            }
        }
        
    def add_sparc_statistics(self):
        """SPARC galaxy fitting statistics."""
        self.predictions['sparc'] = {
            'median_chi2_per_N': {
                'predicted': 1.0,    # Ideal fit
                'observed': 22.1,    # Actual median
                'uncertainty': 5.0,
                'units': 'dimensionless',
                'status': 'poor',
                'chi2': ((22.1 - 1.0) / 5.0)**2
            },
            'good_fits_fraction': {
                'predicted': 0.5,    # Expected 50% with chi2/N < 5
                'observed': 0.105,   # Actual 10.5%
                'uncertainty': 0.05,
                'units': 'fraction',
                'status': 'poor',
                'chi2': ((0.105 - 0.5) / 0.05)**2
            }
        }
        
    def analyze_failures(self):
        """Identify patterns in failed predictions."""
        failures = []
        
        for category, tests in self.predictions.items():
            for test_name, data in tests.items():
                if data['status'] in ['failed', 'poor']:
                    failures.append({
                        'category': category,
                        'test': test_name,
                        'predicted': data['predicted'],
                        'observed': data['observed'],
                        'error_sigma': abs(data['observed'] - data['predicted']) / data['uncertainty'],
                        'chi2': data.get('chi2', 0)
                    })
        
        self.failures_df = pd.DataFrame(failures)
        return self.failures_df
    
    def identify_problem_areas(self):
        """Identify which aspects of the theory need work."""
        problems = {
            'dwarf_spheroidals': {
                'issue': 'Velocity dispersions underpredicted by ~35%',
                'affected_tests': ['draco', 'fornax', 'sculptor'],
                'likely_cause': 'ξ-mode screening too strong or missing physics',
                'suggested_fix': 'Revisit screening function or add baryonic feedback'
            },
            'relativistic_effects': {
                'issue': 'No detection of predicted deviations from GR',
                'affected_tests': ['pulsar_timing', 'strong_lensing'],
                'likely_cause': 'Effects too small or wrong functional form',
                'suggested_fix': 'Reduce predicted amplitudes or add relativistic corrections'
            },
            'fit_quality': {
                'issue': 'High χ²/N even for best fits',
                'affected_tests': ['sparc_median', 'good_fits_fraction'],
                'likely_cause': 'Missing small-scale physics or oversimplified model',
                'suggested_fix': 'Add gas pressure, non-circular motions, or feedback'
            }
        }
        
        return problems
    
    def suggest_improvements(self):
        """Suggest specific improvements to the framework."""
        improvements = []
        
        # Based on dwarf spheroidal failures
        improvements.append({
            'priority': 1,
            'area': 'ξ-mode screening',
            'current': 'S(ρ) = 1/(1 + ρ_gap/ρ)',
            'suggested': 'S(ρ) = 1/(1 + (ρ_gap/ρ)^α) with α ~ 0.5',
            'rationale': 'Softer transition better matches dwarf data'
        })
        
        # Based on high χ² values
        improvements.append({
            'priority': 2,
            'area': 'Baryonic physics',
            'current': 'Pure gravitational model',
            'suggested': 'Add pressure support term: ∇P/ρ',
            'rationale': 'Gas pressure significant in low-mass galaxies'
        })
        
        # Based on non-detection of relativistic effects
        improvements.append({
            'priority': 3,
            'area': 'Amplitude scaling',
            'current': 'β₀ = -0.0557',
            'suggested': 'β_eff = β₀ × (1 - v²/c²)',
            'rationale': 'Relativistic suppression at high velocities'
        })
        
        # Based on velocity gradient analysis
        improvements.append({
            'priority': 4,
            'area': 'Velocity gradient coupling',
            'current': 'α_grad = 1.5e6 m (fixed)',
            'suggested': 'α_grad(ρ) = α₀/(1 + ρ/ρ_crit)',
            'rationale': 'Density-dependent coupling improves dwarf fits'
        })
        
        return improvements
    
    def plot_error_analysis(self):
        """Create visualization of prediction errors."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Error magnitude by category
        ax = axes[0, 0]
        failures = self.analyze_failures()
        if not failures.empty:
            failures.plot(x='test', y='error_sigma', kind='bar', ax=ax)
            ax.set_title('Prediction Errors (σ)')
            ax.set_xlabel('Test')
            ax.set_ylabel('Error (standard deviations)')
            ax.axhline(y=3, color='r', linestyle='--', label='3σ threshold')
            ax.legend()
        
        # Plot 2: χ² contributions
        ax = axes[0, 1]
        if not failures.empty:
            failures.plot(x='test', y='chi2', kind='bar', ax=ax, color='orange')
            ax.set_title('χ² Contributions')
            ax.set_xlabel('Test')
            ax.set_ylabel('χ²')
            ax.set_yscale('log')
        
        # Plot 3: Predicted vs Observed for dwarfs
        ax = axes[1, 0]
        dwarf_data = [
            ('Draco', 5.9, 9.1, 1.2),
            ('Fornax', 7.2, 11.7, 0.9),
            ('Sculptor', 6.5, 9.2, 1.4)
        ]
        names, pred, obs, err = zip(*dwarf_data)
        x = np.arange(len(names))
        ax.bar(x - 0.2, pred, 0.4, label='Predicted', alpha=0.7)
        ax.bar(x + 0.2, obs, 0.4, label='Observed', alpha=0.7)
        ax.errorbar(x + 0.2, obs, yerr=err, fmt='none', color='black')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel('Velocity Dispersion (km/s)')
        ax.set_title('Dwarf Spheroidal Predictions vs Observations')
        ax.legend()
        
        # Plot 4: Improvement priorities
        ax = axes[1, 1]
        improvements = self.suggest_improvements()
        priorities = [imp['priority'] for imp in improvements]
        areas = [imp['area'] for imp in improvements]
        ax.barh(areas, priorities, color='green', alpha=0.7)
        ax.set_xlabel('Priority Level')
        ax.set_title('Suggested Improvements')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('prediction_error_analysis.png', dpi=150)
        plt.close()
        
        return fig
    
    def generate_report(self):
        """Generate comprehensive error analysis report."""
        report = []
        report.append("RS GRAVITY V5 PREDICTION ERROR ANALYSIS")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        failures = self.analyze_failures()
        if not failures.empty:
            report.append(f"Total failed predictions: {len(failures)}")
            report.append(f"Average error: {failures['error_sigma'].mean():.1f}σ")
            report.append(f"Worst prediction: {failures.loc[failures['error_sigma'].idxmax(), 'test']}")
            report.append("")
        
        # Problem areas
        report.append("IDENTIFIED PROBLEM AREAS:")
        report.append("-" * 30)
        problems = self.identify_problem_areas()
        for area, details in problems.items():
            report.append(f"\n{area.upper()}:")
            report.append(f"  Issue: {details['issue']}")
            report.append(f"  Affects: {', '.join(details['affected_tests'])}")
            report.append(f"  Likely cause: {details['likely_cause']}")
            report.append(f"  Suggested fix: {details['suggested_fix']}")
        
        # Improvements
        report.append("\n\nSUGGESTED IMPROVEMENTS (by priority):")
        report.append("-" * 30)
        improvements = self.suggest_improvements()
        for imp in sorted(improvements, key=lambda x: x['priority']):
            report.append(f"\n{imp['priority']}. {imp['area']}")
            report.append(f"   Current: {imp['current']}")
            report.append(f"   Suggested: {imp['suggested']}")
            report.append(f"   Rationale: {imp['rationale']}")
        
        # Save report
        with open('prediction_error_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        return report

def main():
    """Run comprehensive error analysis."""
    analyzer = PredictionErrorAnalyzer()
    
    # Load all predictions
    analyzer.add_laboratory_predictions()
    analyzer.add_astronomical_predictions()
    analyzer.add_sparc_statistics()
    
    # Analyze failures
    failures = analyzer.analyze_failures()
    print("Failed predictions:")
    print(failures)
    print()
    
    # Generate visualizations
    analyzer.plot_error_analysis()
    print("Error analysis plots saved to prediction_error_analysis.png")
    
    # Generate report
    report = analyzer.generate_report()
    print("\nGenerated error analysis report:")
    print('\n'.join(report[:20]) + '\n...')
    print(f"\nFull report saved to prediction_error_report.txt")
    
    # Save failures data
    failures.to_csv('prediction_failures.csv', index=False)
    print(f"\nDetailed failure data saved to prediction_failures.csv")

if __name__ == "__main__":
    main() 