
import numpy as np
import pickle
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Constants
PHI = (1 + np.sqrt(5)) / 2
ALPHA = 0.5 * (1 - 1/PHI)  # ~0.191
C_LAG = PHI**(-5)          # ~0.090
SIGMA_0_FLOOR = 10.0 

# Analytic n(r) parameters
A_N = 7.0
R0_N = 8.0
P_N = 1.6

def get_n_r(r):
    return 1 + A_N * (1 - np.exp(-(r/R0_N)**P_N))

def get_xi(f_gas, thresholds):
    bin_idx = np.searchsorted(thresholds, f_gas)
    bin_idx = np.clip(bin_idx, 0, len(thresholds))
    n_bins = len(thresholds) + 1
    u_b = (bin_idx + 0.5) / n_bins
    return 1 + C_LAG * np.sqrt(u_b)

def eval_global_chi2(master_table, thresholds, ml_val, t_ref_val, lambda_val):
    total_chi2 = 0
    total_n = 0
    
    for name, g in master_table.items():
        r = np.array(g['r'])
        v_obs = np.array(g['v_obs'])
        
        if 'data' in g and 'verr' in g['data']:
            v_err = np.array(g['data']['verr'])
        else:
            v_err = np.ones_like(v_obs) * 5.0
            
        if 'data' in g:
            v_gas = np.array(g['data']['vgas'])
            v_disk = np.array(g['data']['vdisk'])
            v_bul = np.array(g['data']['vbul'])
        else:
            continue
            
        # Fixed M/L
        v_baryon = np.sqrt(
            v_gas**2 + 
            (v_disk * np.sqrt(ml_val))**2 + 
            (v_bul * np.sqrt(ml_val))**2
        )
        
        v_safe = np.maximum(v_baryon, 5.0)
        T_dyn_s = 2 * np.pi * (r * 3.086e19) / (v_safe * 1000)
        T_dyn_yr = T_dyn_s / (365.25 * 24 * 3600)
        
        # ILG Weight
        term_time = 1 + C_LAG * ((T_dyn_yr / t_ref_val)**ALPHA - 1)
        term_n = get_n_r(r)
        f_gas = g.get('f_gas_true', 0.5)
        term_xi = get_xi(f_gas, thresholds)
        
        R_d = g.get('R_d', 2.0)
        h_z = 0.25 * R_d
        x_z = r / R_d
        f_profile = np.exp(-x_z/2) * (1 + x_z/3)
        term_zeta = 1 + 0.5 * (h_z / (r + 0.1 * R_d)) * f_profile
        
        w = lambda_val * term_time * term_n * term_zeta * term_xi
        v_model = np.sqrt(w * v_baryon**2)
        
        # Errors
        dist = g.get('distance', 10.0)
        beam_kpc = 15.0 * dist / 206265.0
        sigma_beam = 0.3 * beam_kpc * np.abs(v_obs) / (r + beam_kpc)
        
        v_max = np.max(np.abs(v_obs))
        if v_max < 80:
            sigma_asym = 0.10 * np.abs(v_obs)
        else:
            sigma_asym = 0.05 * np.abs(v_obs)
            
        sigma_total = np.sqrt(v_err**2 + sigma_beam**2 + sigma_asym**2 + SIGMA_0_FLOOR**2)
        
        mask = r > beam_kpc
        if np.sum(mask) < 5:
            continue
            
        res = (v_obs - v_model)[mask]
        err = sigma_total[mask]
        
        total_chi2 += np.sum((res/err)**2)
        total_n += len(res)
        
    return total_chi2 / total_n if total_n > 0 else 1e9

def calibrate():
    print("Calibrating Global Parameters (M/L, T_REF, LAMBDA)...")
    
    data_path = Path('active/scripts/sparc_master_real.pkl')
    if not data_path.exists():
        data_path = Path('sparc_master_real.pkl')
    
    if not data_path.exists():
        # Try looking in parent directories or explicit path
        data_path = Path('/Users/jonathanwashburn/.cursor/worktrees/gravity/j8Zyv/active/scripts/sparc_master_real.pkl')
        
    if not data_path.exists():
        print(f"Error: sparc_master_real.pkl not found at {data_path}")
        return
        
    with open(data_path, 'rb') as f:
        master_table = pickle.load(f)
        
    valid_f_gas = [g['f_gas_true'] for g in master_table.values() if not np.isnan(g['f_gas_true'])]
    thresholds = np.quantile(valid_f_gas, [0.2, 0.4, 0.6, 0.8])
    
    # Grid Search
    ml_grid = np.linspace(0.4, 0.8, 5)
    # Log space for T_ref
    t_ref_grid = np.logspace(3, 9, 7) # 1e3 to 1e9
    # Lambda grid (around 1/mean(n) ~ 0.2)
    lambda_grid = np.linspace(0.1, 0.5, 5)
    
    best_chi2 = 1e9
    best_params = None
    
    print(f"Scanning {len(ml_grid)*len(t_ref_grid)*len(lambda_grid)} combinations...")
    
    for ml in ml_grid:
        for t_ref in t_ref_grid:
            for lam in lambda_grid:
                chi2 = eval_global_chi2(master_table, thresholds, ml, t_ref, lam)
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_params = (ml, t_ref, lam)
                    print(f"New Best: Chi2={chi2:.4f} | M/L={ml:.2f}, T={t_ref:.1e}, L={lam:.2f}")
                
    print("\nCalibration Complete.")
    print(f"Best Global Chi2/N: {best_chi2:.4f}")
    print(f"Best M/L: {best_params[0]:.3f}")
    print(f"Best T_REF: {best_params[1]:.3e}")
    print(f"Best LAMBDA: {best_params[2]:.3f}")

if __name__ == "__main__":
    calibrate()

