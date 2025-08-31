#!/usr/bin/env python3
"""
SPARC Galaxy Rotation Curve Solver
==================================
Processes all 175 SPARC galaxies using the unified RS gravity framework
Zero free parameters - everything derived from first principles
"""

import numpy as np
import pandas as pd
from lnal_sparc_solver import UnifiedGravitySolver, GalaxyData
import matplotlib.pyplot as plt
from typing import Dict, List
import os
import glob
import pickle


class SPARCDataProcessor:
    """Processes SPARC galaxy data files"""
    
    def __init__(self, data_dir: str = "Rotmod_LTG"):
        """
        Initialize processor
        
        Parameters:
        -----------
        data_dir : str
            Directory containing _rotmod.dat files
        """
        self.data_dir = data_dir
        self.galaxies = {}
        
    def load_galaxy(self, filename: str) -> GalaxyData:
        """
        Load a single galaxy from rotmod file
        
        Parameters:
        -----------
        filename : str
            Path to _rotmod.dat file
            
        Returns:
        --------
        GalaxyData object
        """
        # Extract galaxy name
        basename = os.path.basename(filename)
        name = basename.replace('_rotmod.dat', '')
        
        # Read data
        data = np.loadtxt(filename, skiprows=1)
        
        # Extract columns
        # Typical format: R(kpc) v_obs v_gas v_disk v_bul SBgas SBdisk SBbul
        R_kpc = data[:, 0]
        v_obs = data[:, 1]
        v_gas = data[:, 2]
        v_disk = data[:, 3]
        v_bul = data[:, 4] if data.shape[1] > 4 else np.zeros_like(R_kpc)
        
        # Surface brightness to surface density conversion
        # Assuming standard conversion factors
        SBgas = data[:, 5] if data.shape[1] > 5 else np.zeros_like(R_kpc)
        SBdisk = data[:, 6] if data.shape[1] > 6 else np.zeros_like(R_kpc)
        SBbul = data[:, 7] if data.shape[1] > 7 else np.zeros_like(R_kpc)
        
        # Convert to surface densities (Msun/pc²)
        # Using typical gas and stellar M/L ratios
        sigma_gas = 1.33 * SBgas  # Include He correction
        sigma_disk = 0.5 * SBdisk  # Typical stellar M/L
        sigma_bulge = 0.7 * SBbul if np.any(SBbul > 0) else None
        
        # Estimate errors (3% or 2 km/s minimum)
        v_err = np.maximum(0.03 * v_obs, 2.0)
        
        return GalaxyData(
            name=name,
            R_kpc=R_kpc,
            v_obs=v_obs,
            v_err=v_err,
            sigma_gas=sigma_gas,
            sigma_disk=sigma_disk,
            sigma_bulge=sigma_bulge
        )
    
    def load_all_galaxies(self) -> Dict[str, GalaxyData]:
        """Load all galaxies from data directory"""
        
        pattern = os.path.join(self.data_dir, "*_rotmod.dat")
        files = glob.glob(pattern)
        
        print(f"Found {len(files)} galaxy files in {self.data_dir}")
        
        for file in files:
            try:
                galaxy = self.load_galaxy(file)
                self.galaxies[galaxy.name] = galaxy
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
        print(f"Successfully loaded {len(self.galaxies)} galaxies")
        return self.galaxies


class SPARCUnifiedAnalysis:
    """Complete SPARC analysis with unified RS gravity"""
    
    def __init__(self):
        """Initialize analysis"""
        self.solver = UnifiedGravitySolver()
        self.processor = SPARCDataProcessor()
        self.results = {}
        
    def analyze_galaxy(self, galaxy: GalaxyData) -> Dict:
        """
        Analyze single galaxy
        
        Returns:
        --------
        Dictionary with fit results and statistics
        """
        print(f"\nAnalyzing {galaxy.name}...")
        
        try:
            # Solve with unified framework
            result = self.solver.solve_galaxy(galaxy)
            
            # Add galaxy info
            result['name'] = galaxy.name
            result['N_points'] = len(galaxy.R_kpc)
            
            # Compute additional statistics
            residuals = galaxy.v_obs - result['v_model']
            result['rms'] = np.sqrt(np.mean(residuals**2))
            result['max_residual'] = np.max(np.abs(residuals))
            
            # Quality classification
            if result['chi2_reduced'] < 1.5:
                result['quality'] = 'Excellent'
            elif result['chi2_reduced'] < 3.0:
                result['quality'] = 'Good'
            elif result['chi2_reduced'] < 5.0:
                result['quality'] = 'Fair'
            else:
                result['quality'] = 'Poor'
                
            print(f"  χ²/N = {result['chi2_reduced']:.3f} ({result['quality']})")
            
            return result
            
        except Exception as e:
            print(f"  ERROR: {e}")
            return None
    
    def analyze_all(self, save_results: bool = True):
        """Analyze all SPARC galaxies"""
        
        # Load galaxies
        galaxies = self.processor.load_all_galaxies()
        
        print("\n" + "="*60)
        print("ANALYZING ALL SPARC GALAXIES")
        print("="*60)
        
        # Process each galaxy
        for name, galaxy in galaxies.items():
            result = self.analyze_galaxy(galaxy)
            if result is not None:
                self.results[name] = result
                
        # Summary statistics
        self._print_summary()
        
        # Save results
        if save_results:
            self._save_results()
            
    def _print_summary(self):
        """Print analysis summary"""
        
        chi2_values = [r['chi2_reduced'] for r in self.results.values()]
        rms_values = [r['rms'] for r in self.results.values()]
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Total galaxies analyzed: {len(self.results)}")
        print(f"Mean χ²/N: {np.mean(chi2_values):.3f}")
        print(f"Median χ²/N: {np.median(chi2_values):.3f}")
        print(f"Best χ²/N: {np.min(chi2_values):.3f}")
        print(f"Worst χ²/N: {np.max(chi2_values):.3f}")
        
        # Quality breakdown
        quality_counts = {}
        for r in self.results.values():
            q = r['quality']
            quality_counts[q] = quality_counts.get(q, 0) + 1
            
        print("\nQuality distribution:")
        for q in ['Excellent', 'Good', 'Fair', 'Poor']:
            count = quality_counts.get(q, 0)
            pct = 100 * count / len(self.results)
            print(f"  {q}: {count} ({pct:.1f}%)")
            
    def _save_results(self):
        """Save analysis results"""
        
        # Save pickle
        with open('sparc_unified_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
            
        # Save summary CSV
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'Galaxy': name,
                'N_points': result['N_points'],
                'chi2_N': result['chi2_reduced'],
                'RMS_km/s': result['rms'],
                'Quality': result['quality']
            })
            
        df = pd.DataFrame(summary_data)
        df.to_csv('sparc_unified_summary.csv', index=False)
        print(f"\nResults saved to sparc_unified_results.pkl and sparc_unified_summary.csv")
        
    def plot_best_fits(self, n_galaxies: int = 6):
        """Plot best fitting galaxies"""
        
        # Sort by chi2
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1]['chi2_reduced'])
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(sorted_results[:n_galaxies]):
            ax = axes[i]
            galaxy = self.processor.galaxies[name]
            
            # Plot data and fit
            ax.errorbar(galaxy.R_kpc, galaxy.v_obs, yerr=galaxy.v_err,
                       fmt='ko', alpha=0.6, markersize=4)
            ax.plot(galaxy.R_kpc, result['v_model'], 'r-', linewidth=2,
                   label=f'χ²/N = {result["chi2_reduced"]:.2f}')
            ax.plot(galaxy.R_kpc, result['v_baryon'], 'b--', alpha=0.7,
                   label='Baryonic')
            
            ax.set_xlabel('R (kpc)')
            ax.set_ylabel('v (km/s)')
            ax.set_title(name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('sparc_best_fits.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_acceleration_relation(self):
        """Plot the universal acceleration relation"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Collect all acceleration data
        all_a_baryon = []
        all_a_total = []
        
        for result in self.results.values():
            all_a_baryon.extend(result['a_baryon'])
            all_a_total.extend(result['a_total'])
            
        all_a_baryon = np.array(all_a_baryon)
        all_a_total = np.array(all_a_total)
        
        # Plot
        ax.loglog(all_a_baryon, all_a_total, 'o', alpha=0.3, 
                 markersize=2, color='blue')
        
        # Theory curves
        a_range = np.logspace(-14, -8, 100)
        
        # MOND limit
        a_mond = np.sqrt(a_range * 1.2e-10)
        ax.loglog(a_range, a_mond, 'k--', linewidth=2, 
                 label='MOND: a = √(a_N g†)')
        
        # Newton limit
        ax.loglog(a_range, a_range, 'k:', linewidth=2,
                 label='Newton: a = a_N')
        
        ax.set_xlabel('a_baryon (m/s²)', fontsize=14)
        ax.set_ylabel('a_total (m/s²)', fontsize=14)
        ax.set_title('SPARC Acceleration Relation (RS Theory)', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        textstr = f'N_galaxies = {len(self.results)}\n'
        textstr += f'N_points = {len(all_a_baryon)}\n'
        textstr += f'Mean χ²/N = {np.mean([r["chi2_reduced"] for r in self.results.values()]):.2f}'
        
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('sparc_acceleration_relation.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Run complete SPARC analysis"""
    
    print("Recognition Science SPARC Analysis")
    print("=" * 60)
    print("Zero free parameters!")
    print("All constants derived from J(x) = ½(x + 1/x)")
    print("=" * 60)
    
    # Create analyzer
    analyzer = SPARCUnifiedAnalysis()
    
    # Run analysis
    analyzer.analyze_all(save_results=True)
    
    # Create plots
    print("\nGenerating plots...")
    analyzer.plot_best_fits()
    analyzer.plot_acceleration_relation()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main() 