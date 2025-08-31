#!/usr/bin/env python3
"""
LNAL Recognition Science Gravity Solver
Complete implementation based on Recognition Science manuscript

Key principles:
1. Running Newton coupling: G(r) = G_∞ (λ_rec/r)^β
2. Eight-tick cycle completeness with cosmological clock lag
3. Dual-column ledger accounting (flow and stock)
4. Information field dynamics
5. Golden ratio cascade
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import interp1d
import pickle
import os
from typing import Dict, Tuple, Optional

# Physical constants (SI units)
c = 2.998e8  # m/s
G_inf = 6.674e-11  # m³/kg/s² (cosmic scale Newton constant)
hbar = 1.055e-34  # J⋅s
m_p = 1.673e-27  # kg (proton mass)
k_B = 1.381e-23  # J/K
kpc_to_m = 3.086e19  # m/kpc
km_to_m = 1000  # m/km
eV_to_J = 1.602e-19  # J/eV

# Recognition Science constants
phi = (1 + np.sqrt(5)) / 2  # Golden ratio = 1.618034
beta = -(phi - 1) / phi**5  # ≈ -0.0557 (running G exponent)
lambda_rec = 42.9e-9  # m (recognition recurrence length)
E_coh = 0.090 * eV_to_J  # J (coherence quantum)

# Derived constants
tau_0 = hbar / E_coh  # ≈ 7.33 fs (fundamental tick)
Gamma = 8 * tau_0  # Macro-chronon (8-tick cycle)
CLOCK_LAG = 45 / 960  # 4.69% cosmological clock lag

# Voxel parameters
L_0 = 0.335e-9  # m (voxel size)
V_voxel = L_0**3  # m³
I_star = m_p * c**2 / V_voxel  # J/m³ (information capacity scale)
mu = hbar / (c * lambda_rec)  # m⁻² (field mass parameter)
g_dagger = 1.2e-10  # m/s² (MOND acceleration scale)

class RecognitionGravitySolver:
    """Complete Recognition Science gravity solver"""
    
    def __init__(self, baryon_data_file='sparc_exact_baryons.pkl'):
        self.baryon_data = self.load_baryon_data(baryon_data_file)
        self.results = {}
        
        print("Recognition Science Gravity Solver initialized")
        print(f"  β = {beta:.6f} (running G exponent)")
        print(f"  λ_rec = {lambda_rec*1e9:.1f} nm")
        print(f"  Clock lag = {CLOCK_LAG*100:.2f}%")
        print(f"  I* = {I_star:.2e} J/m³")
        
    def load_baryon_data(self, filename):
        """Load exact baryonic source data"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded baryon data for {len(data)} galaxies")
            return data
        else:
            print(f"Warning: {filename} not found")
            return {}
    
    def G_running(self, r):
        """
        Running Newton coupling from Recognition Science
        G(r) = G_∞ (λ_rec/r)^β
        """
        return G_inf * (lambda_rec / r)**beta
    
    def solve_information_field(self, R_kpc, B_R):
        """
        Solve the information field equation:
        ∇·[μ(u)∇ρ_I] - μ²ρ_I = -λB
        
        where u = |∇ρ_I|/(I*μ) is the MOND parameter
        """
        R = R_kpc * kpc_to_m
        
        def field_equation(rho_I, r):
            """ODEs for ρ_I and its derivative"""
            rho, drho_dr = rho_I
            
            # Avoid singularity at r=0
            if r < 1e-10:
                return [drho_dr, 0]
            
            # MOND parameter
            u = abs(drho_dr) / (I_star * mu)
            mu_u = self.mond_interpolation(u)
            
            # Source term (interpolate B)
            r_kpc = r / kpc_to_m
            B_interp = np.interp(r_kpc, R_kpc, B_R)
            
            # Coupling constant
            lambda_coupling = np.sqrt(g_dagger * c**2 / I_star)
            
            # Second derivative from field equation
            d2rho_dr2 = (mu**2 * rho - lambda_coupling * B_interp) / mu_u - (2/r) * drho_dr
            
            return [drho_dr, d2rho_dr2]
        
        # Initial conditions
        rho_I_0 = B_R[0] * np.sqrt(g_dagger * c**2 / I_star) / mu**2
        initial_conditions = [rho_I_0, 0]
        
        # Solve ODE
        solution = odeint(field_equation, initial_conditions, R)
        rho_I = solution[:, 0]
        drho_I_dr = solution[:, 1]
        
        # Ensure positivity
        rho_I = np.maximum(rho_I, 0)
        
        return rho_I, drho_I_dr
    
    def mond_interpolation(self, u):
        """MOND interpolation function μ(u) = u/√(1+u²)"""
        return u / np.sqrt(1 + u**2)
    
    def compute_acceleration(self, r, sigma_total, rho_I, drho_I_dr):
        """
        Compute total acceleration including:
        1. Newtonian with running G(r)
        2. Information field contribution
        3. Clock lag correction
        """
        # Running Newton's constant
        G_r = self.G_running(r)
        
        # Newtonian acceleration (thin disk)
        a_N = 2 * np.pi * G_r * sigma_total
        
        # Information field acceleration
        lambda_coupling = np.sqrt(g_dagger * c**2 / I_star)
        a_info = (lambda_coupling / c**2) * drho_I_dr
        
        # MOND parameter
        x = a_N / g_dagger
        
        # Deep MOND regime
        deep_mond_mask = x < 0.1
        
        # Total acceleration with regime-dependent scaling
        a_total = np.zeros_like(a_N)
        
        # Deep MOND: a = √(a_N * g_dagger) with Recognition corrections
        a_total[deep_mond_mask] = np.sqrt(a_N[deep_mond_mask] * g_dagger) * (1 + CLOCK_LAG)
        
        # Transition/Newtonian regime
        transition_mask = ~deep_mond_mask
        u = abs(drho_I_dr) / (I_star * mu)
        mu_u = self.mond_interpolation(u)
        a_total[transition_mask] = a_N[transition_mask] + a_info[transition_mask] * mu_u[transition_mask]
        
        return a_total, a_N
    
    def solve_galaxy(self, galaxy_name):
        """Solve for a single galaxy using Recognition Science principles"""
        if galaxy_name not in self.baryon_data:
            return None
        
        data = self.baryon_data[galaxy_name]
        R_kpc = data['radius']  # kpc
        v_obs = data['v_obs']  # km/s
        v_err = data['v_err']  # km/s
        
        # Total surface density
        sigma_total = data['sigma_gas'] + data['sigma_disk'] + data['sigma_bulge']  # kg/m²
        
        # Baryonic energy density for information field
        R = R_kpc * kpc_to_m
        B_R = sigma_total * c**2 / (2 * np.pi * R)  # J/m³
        
        # Solve information field
        rho_I, drho_I_dr = self.solve_information_field(R_kpc, B_R)
        
        # Compute accelerations
        a_total, a_N = self.compute_acceleration(R, sigma_total, rho_I, drho_I_dr)
        
        # Convert to velocity
        v_model_squared = a_total * R
        v_model = np.sqrt(np.maximum(v_model_squared, 0)) / km_to_m  # km/s
        
        # Baryonic velocity
        v_baryon = np.sqrt(a_N * R) / km_to_m
        
        # Apply minimum velocity constraint
        v_model = np.maximum(v_model, v_baryon)
        
        # Compute χ²
        chi2 = np.sum(((v_obs - v_model) / v_err)**2)
        chi2_reduced = chi2 / len(v_obs)
        
        return {
            'galaxy': galaxy_name,
            'R_kpc': R_kpc,
            'v_obs': v_obs,
            'v_err': v_err,
            'v_model': v_model,
            'v_baryon': v_baryon,
            'a_total': a_total,
            'a_N': a_N,
            'rho_I': rho_I,
            'chi2': chi2,
            'chi2_reduced': chi2_reduced,
            'N_points': len(v_obs)
        }
    
    def solve_all_galaxies(self, max_galaxies=None):
        """Solve for all galaxies"""
        galaxy_names = list(self.baryon_data.keys())
        if max_galaxies:
            galaxy_names = galaxy_names[:max_galaxies]
        
        print(f"\nSolving Recognition field equations for {len(galaxy_names)} galaxies...")
        print("="*60)
        
        results = []
        chi2_values = []
        
        for i, galaxy_name in enumerate(galaxy_names):
            print(f"[{i+1:3d}/{len(galaxy_names)}] {galaxy_name:15s}", end=' ... ')
            
            try:
                result = self.solve_galaxy(galaxy_name)
                if result and np.isfinite(result['chi2_reduced']) and result['chi2_reduced'] > 0:
                    results.append(result)
                    chi2_values.append(result['chi2_reduced'])
                    print(f"χ²/N = {result['chi2_reduced']:7.3f}")
                else:
                    print("Failed: invalid result")
            except Exception as e:
                print(f"Error: {str(e)[:40]}...")
                continue
        
        if not chi2_values:
            print("No valid results obtained")
            return None
        
        # Compute statistics
        chi2_values = np.array(chi2_values)
        chi2_mean = np.mean(chi2_values)
        chi2_std = np.std(chi2_values)
        chi2_median = np.median(chi2_values)
        
        print("\n" + "="*60)
        print("RECOGNITION SCIENCE GRAVITY RESULTS:")
        print(f"  Galaxies processed: {len(results)}")
        print(f"  Mean χ²/N: {chi2_mean:.3f} ± {chi2_std:.3f}")
        print(f"  Median χ²/N: {chi2_median:.3f}")
        print(f"  Best fit: χ²/N = {np.min(chi2_values):.3f}")
        print(f"  Worst fit: χ²/N = {np.max(chi2_values):.3f}")
        print(f"  Fraction with χ²/N < 2: {np.mean(chi2_values < 2):.1%}")
        print(f"  Fraction with χ²/N < 5: {np.mean(chi2_values < 5):.1%}")
        print(f"  Fraction with χ²/N < 10: {np.mean(chi2_values < 10):.1%}")
        
        self.results = {
            'individual': results,
            'chi2_mean': chi2_mean,
            'chi2_std': chi2_std,
            'chi2_median': chi2_median,
            'chi2_values': chi2_values,
            'n_galaxies': len(results)
        }
        
        # Save results
        with open('lnal_recognition_gravity_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        print("\nResults saved to lnal_recognition_gravity_results.pkl")
        
        return self.results
    
    def plot_best_examples(self, n_examples=6):
        """Plot best fitting galaxies"""
        if not self.results or not self.results['individual']:
            print("No results to plot")
            return
        
        # Sort by χ²
        sorted_results = sorted(self.results['individual'], key=lambda x: x['chi2_reduced'])
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(min(n_examples, len(sorted_results))):
            result = sorted_results[i]
            ax = axes[i]
            
            # Plot rotation curve
            ax.errorbar(result['R_kpc'], result['v_obs'], yerr=result['v_err'], 
                       fmt='ko', alpha=0.6, markersize=4, label='Observed')
            ax.plot(result['R_kpc'], result['v_model'], 'r-', linewidth=2, 
                   label=f'Recognition Model')
            ax.plot(result['R_kpc'], result['v_baryon'], 'b--', linewidth=1.5, 
                   alpha=0.7, label='Baryonic')
            
            ax.set_xlabel('Radius (kpc)')
            ax.set_ylabel('Velocity (km/s)')
            ax.set_title(f'{result["galaxy"]}\nχ²/N = {result["chi2_reduced"]:.2f}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lnal_recognition_gravity_best_fits.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Best fits saved to lnal_recognition_gravity_best_fits.png")
    
    def plot_acceleration_relation(self):
        """Plot the radial acceleration relation"""
        if not self.results or not self.results['individual']:
            return
        
        # Collect all accelerations
        all_a_N = []
        all_a_total = []
        
        for result in self.results['individual']:
            all_a_N.extend(result['a_N'])
            all_a_total.extend(result['a_total'])
        
        all_a_N = np.array(all_a_N)
        all_a_total = np.array(all_a_total)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Data points
        ax.loglog(all_a_N, all_a_total, 'k.', alpha=0.1, markersize=1)
        
        # Theory curves
        a_N_theory = np.logspace(-13, -8, 100)
        
        # Newtonian
        ax.loglog(a_N_theory, a_N_theory, 'b:', linewidth=2, label='Newtonian')
        
        # MOND
        a_MOND = np.sqrt(a_N_theory * g_dagger)
        ax.loglog(a_N_theory, a_MOND, 'g--', linewidth=2, label='MOND')
        
        # Recognition Science (with running G)
        r_test = 10 * kpc_to_m  # typical radius
        G_ratio = self.G_running(r_test) / G_inf
        a_RS = a_MOND * np.sqrt(G_ratio) * (1 + CLOCK_LAG)
        ax.loglog(a_N_theory, a_RS, 'r-', linewidth=2, label='Recognition Science')
        
        ax.set_xlabel('$a_N$ (m/s²)')
        ax.set_ylabel('$a_{total}$ (m/s²)')
        ax.set_title('Radial Acceleration Relation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lnal_recognition_acceleration_relation.png', dpi=150)
        plt.show()

def main():
    """Execute Recognition Science gravity analysis"""
    print("="*60)
    print("LNAL RECOGNITION SCIENCE GRAVITY SOLVER")
    print("Based on Recognition Science manuscript principles")
    print("="*60)
    
    solver = RecognitionGravitySolver()
    
    # Test on subset first
    print("\nTesting on 10 galaxies...")
    test_results = solver.solve_all_galaxies(max_galaxies=10)
    
    if test_results and test_results['chi2_mean'] < 100:
        # Run full analysis
        print("\nRunning full SPARC analysis...")
        full_results = solver.solve_all_galaxies()
        
        if full_results:
            solver.plot_best_examples()
            solver.plot_acceleration_relation()
            
            # Assessment
            chi2_mean = full_results['chi2_mean']
            if chi2_mean < 2.0:
                print(f"\n✅ EXCELLENT! χ²/N = {chi2_mean:.3f}")
                print("Recognition Science gravity validated!")
            elif chi2_mean < 5.0:
                print(f"\n⚠️  GOOD: χ²/N = {chi2_mean:.3f}")
                print("Minor refinements may improve fit")
            else:
                print(f"\n❌ χ²/N = {chi2_mean:.3f}")
                print("Further investigation needed")

if __name__ == "__main__":
    main() 