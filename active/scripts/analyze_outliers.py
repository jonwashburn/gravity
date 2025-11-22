
import numpy as np
import pickle
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Constants
PHI = (1 + np.sqrt(5)) / 2
ALPHA = 0.5 * (1 - 1/PHI)
C_LAG = PHI**(-5)
SIGMA_0_FLOOR = 10.0 

# Optimized Parameters
A_N = 9.5
R0_N = 3.2
P_N = 0.5
ML_GLOBAL = 0.40
T_REF_GLOBAL = 1.0e6
LAMBDA_GLOBAL = 0.40

def get_n_r(r):
    return 1 + A_N * (1 - np.exp(-(r/R0_N)**P_N))

def get_xi(f_gas, thresholds):
    bin_idx = np.searchsorted(thresholds, f_gas)
    bin_idx = np.clip(bin_idx, 0, len(thresholds))
    n_bins = len(thresholds) + 1
    u_b = (bin_idx + 0.5) / n_bins
    return 1 + C_LAG * np.sqrt(u_b)

def analyze_outliers():
    data_path = Path('/Users/jonathanwashburn/.cursor/worktrees/gravity/j8Zyv/active/scripts/sparc_master_real.pkl')
    if not data_path.exists():
        data_path = Path('sparc_master_real.pkl')
        
    with open(data_path, 'rb') as f:
        master_table = pickle.load(f)
        
    valid_f_gas = [g['f_gas_true'] for g in master_table.values() if not np.isnan(g['f_gas_true'])]
    thresholds = np.quantile(valid_f_gas, [0.2, 0.4, 0.6, 0.8])
    
    results = []
    
    for name, g in master_table.items():
        r = np.array(g['r'])
        v_obs = np.array(g['v_obs'])
        
        if 'data' in g and 'verr' in g['data']:
            v_err = np.array(g['data']['verr'])
        else:
            v_err = np.ones_like(v_obs) * 5.0
            
        if 'data' not in g: continue
            
        v_gas = np.array(g['data']['vgas'])
        v_disk = np.array(g['data']['vdisk'])
        v_bul = np.array(g['data']['vbul'])
        
        # Model
        v_baryon = np.sqrt(v_gas**2 + ML_GLOBAL*(v_disk**2 + v_bul**2))
        v_safe = np.maximum(v_baryon, 5.0)
        T_dyn_yr = (2 * np.pi * (r * 3.086e19) / (v_safe * 1000)) / (365.25 * 24 * 3600)
        
        term_time = 1 + C_LAG * ((T_dyn_yr / T_REF_GLOBAL)**ALPHA - 1)
        term_n = get_n_r(r)
        term_xi = get_xi(g.get('f_gas_true', 0.5), thresholds)
        
        R_d = g.get('R_d', 2.0)
        h_z = 0.25 * R_d
        f_profile = np.exp(-r/(2*R_d)) * (1 + (r/R_d)/3)
        term_zeta = 1 + 0.5 * (h_z / (r + 0.1 * R_d)) * f_profile
        
        w = LAMBDA_GLOBAL * term_time * term_n * term_zeta * term_xi
        v_model = np.sqrt(w * v_baryon**2)
        
        # Errors
        dist = g.get('distance', 10.0)
        beam_kpc = 15.0 * dist / 206265.0
        sigma_beam = 0.44 * beam_kpc * np.abs(v_obs) / (r + beam_kpc)
        
        v_max = np.max(np.abs(v_obs))
        sigma_asym = (0.08 if v_max < 80 else 0.06) * np.abs(v_obs)
            
        sigma_total = np.sqrt(v_err**2 + sigma_beam**2 + sigma_asym**2 + SIGMA_0_FLOOR**2)
        
        mask = r > beam_kpc
        if np.sum(mask) < 5: continue
        
        res = (v_obs - v_model)[mask]
        chi2 = np.sum((res/sigma_total[mask])**2)
        dof = len(res)
        
        results.append({'name': name, 'chi2_nu': chi2/dof})

    # Sort and Print
    results.sort(key=lambda x: x['chi2_nu'], reverse=True)
    
    vals = [x['chi2_nu'] for x in results]
    median = np.median(vals)
    mean = np.mean(vals)
    
    print(f"Median: {median:.3f}")
    print(f"Mean:   {mean:.3f}")
    print(f"\nTop 10 Worst Offenders:")
    for i in range(10):
        print(f"{i+1}. {results[i]['name']}: {results[i]['chi2_nu']:.2f}")
        
    # Impact of removing outliers
    print(f"\nImpact of removing worst 5% ({int(len(vals)*0.05)} galaxies):")
    vals_clean = sorted(vals)[:-int(len(vals)*0.05)]
    print(f"New Mean: {np.mean(vals_clean):.3f}")
    print(f"New Median: {np.median(vals_clean):.3f}")

if __name__ == "__main__":
    analyze_outliers()

