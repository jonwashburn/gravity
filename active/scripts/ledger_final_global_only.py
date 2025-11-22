
import numpy as np
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Constants from Source.txt / Paper
PHI = (1 + np.sqrt(5)) / 2
ALPHA = 0.5 * (1 - 1/PHI)  # ~0.191
C_LAG = PHI**(-5)          # ~0.090
SIGMA_0_FLOOR = 10.0 # km/s error floor

# Calibrated Global Parameters
ML_GLOBAL = 0.40
T_REF_GLOBAL = 1.0e6
LAMBDA_GLOBAL = 0.40

# Analytic n(r) parameters (Optimized Global)
A_N = 9.5
R0_N = 3.2
P_N = 0.5

def get_n_r(r):
    return 1 + A_N * (1 - np.exp(-(r/R0_N)**P_N))

def get_xi(f_gas, thresholds):
    bin_idx = np.searchsorted(thresholds, f_gas)
    bin_idx = np.clip(bin_idx, 0, len(thresholds))
    n_bins = len(thresholds) + 1
    u_b = (bin_idx + 0.5) / n_bins
    return 1 + C_LAG * np.sqrt(u_b)

def run_pass(master_table, thresholds, ml_val):
    chi2_list = []
    
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
        
        T_REF = T_REF_GLOBAL
        term_time = 1 + C_LAG * ((T_dyn_yr / T_REF)**ALPHA - 1)
        term_n = get_n_r(r)
        f_gas = g.get('f_gas_true', 0.5)
        term_xi = get_xi(f_gas, thresholds)
        
        R_d = g.get('R_d', 2.0)
        h_z = 0.25 * R_d
        x_z = r / R_d
        f_profile = np.exp(-x_z/2) * (1 + x_z/3)
        term_zeta = 1 + 0.5 * (h_z / (r + 0.1 * R_d)) * f_profile
        
        w = LAMBDA_GLOBAL * term_time * term_n * term_zeta * term_xi
        v_model = np.sqrt(w * v_baryon**2)
        
        dist = g.get('distance', 10.0)
        beam_kpc = 15.0 * dist / 206265.0
        sigma_beam = 0.44 * beam_kpc * np.abs(v_obs) / (r + beam_kpc)
        
        v_max = np.max(np.abs(v_obs))
        if v_max < 80:
            sigma_asym = 0.08 * np.abs(v_obs)
        else:
            sigma_asym = 0.06 * np.abs(v_obs)
            
        sigma_total = np.sqrt(v_err**2 + sigma_beam**2 + sigma_asym**2 + SIGMA_0_FLOOR**2)
        
        mask = r > beam_kpc
        if np.sum(mask) < 5:
            continue
            
        res = (v_obs - v_model)[mask]
        err = sigma_total[mask]
        chi2 = np.sum((res/err)**2)
        dof = len(res)
        chi2_list.append(chi2 / dof)
        
    return np.array(chi2_list)

def solve():
    print("Running Global-Only ILG Solver...")
    
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
    
    print(f"Processing {len(master_table)} galaxies...")
    
    # Run M/L=0.4 (Calibrated)
    res_10 = run_pass(master_table, thresholds, ML_GLOBAL)
    
    print("\n" + "="*40)
    print(f"GLOBAL-ONLY RESULTS (N={len(res_10)})")
    print("="*40)
    print(f"Fixed M/L: {ML_GLOBAL}")
    print(f"Fixed T_REF: {T_REF_GLOBAL}")
    print(f"Fixed LAMBDA: {LAMBDA_GLOBAL}")
    print(f"Median Chi2/N: {np.median(res_10):.3f}")
    print(f"Mean Chi2/N:   {np.mean(res_10):.3f}")
    print("="*40)

if __name__ == "__main__":
    solve()
