
import numpy as np
import pickle
from pathlib import Path
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# Load Data
data_path = Path('/Users/jonathanwashburn/.cursor/worktrees/gravity/j8Zyv/active/scripts/sparc_master_real.pkl')
if not data_path.exists():
    data_path = Path('sparc_master_real.pkl')
    
with open(data_path, 'rb') as f:
    master_table = pickle.load(f)

# Pre-compute thresholds
valid_f_gas = [g['f_gas_true'] for g in master_table.values() if not np.isnan(g['f_gas_true'])]
THRESHOLDS = np.quantile(valid_f_gas, [0.2, 0.4, 0.6, 0.8])

# Fixed Physics Constants (from previous calibration)
PHI = (1 + np.sqrt(5)) / 2
ALPHA_PHYS = 0.5 * (1 - 1/PHI)
C_LAG = PHI**(-5)
ML_GLOBAL = 0.40
T_REF = 1.0e6
LAMBDA_GLOBAL = 0.40

def get_xi(f_gas):
    bin_idx = np.searchsorted(THRESHOLDS, f_gas)
    bin_idx = np.clip(bin_idx, 0, len(THRESHOLDS))
    n_bins = len(THRESHOLDS) + 1
    u_b = (bin_idx + 0.5) / n_bins
    return 1 + C_LAG * np.sqrt(u_b)

def global_loss(params):
    # Unpack parameters
    # n(r) params
    A_n, R0_n, P_n = params[0], params[1], params[2]
    # Error params
    alpha_beam, beta_dwarf, beta_spiral, sigma_0 = params[3], params[4], params[5], params[6]
    
    chi2_list = []
    
    for name, g in master_table.items():
        r = np.array(g['r'])
        v_obs = np.array(g['v_obs'])
        
        # Data filtering
        if len(r) < 5: continue
        
        if 'data' in g and 'verr' in g['data']:
            v_err = np.array(g['data']['verr'])
        else:
            v_err = np.ones_like(v_obs) * 5.0
            
        v_gas = np.array(g['data']['vgas'])
        v_disk = np.array(g['data']['vdisk'])
        v_bul = np.array(g['data']['vbul'])
        
        # --- Physics Model ---
        v_baryon = np.sqrt(v_gas**2 + ML_GLOBAL*(v_disk**2 + v_bul**2))
        v_safe = np.maximum(v_baryon, 1.0)
        
        # T_dyn (years)
        T_dyn_s = 2 * np.pi * (r * 3.086e19) / (v_safe * 1000)
        T_dyn_yr = T_dyn_s / (365.25 * 86400)
        
        # Weights
        w_time = 1 + C_LAG * ((T_dyn_yr / T_REF)**ALPHA_PHYS - 1)
        
        # n(r)
        w_n = 1 + A_n * (1 - np.exp(-(r/R0_n)**P_n))
        
        # xi
        f_gas = g.get('f_gas_true', 0.5)
        w_xi = get_xi(f_gas)
        
        # zeta (simplified)
        R_d = g.get('R_d', 2.0)
        h_z = 0.25 * R_d
        w_zeta = 1 + 0.5 * (h_z / (r + 0.1*R_d)) * np.exp(-r/(2*R_d))
        
        w_total = LAMBDA_GLOBAL * w_time * w_n * w_xi * w_zeta
        v_model = np.sqrt(w_total * v_baryon**2)
        
        # --- Error Model ---
        dist = g.get('distance', 10.0)
        beam_kpc = 15.0 * dist / 206265.0
        
        # Beam error
        sigma_beam = alpha_beam * beam_kpc * np.abs(v_obs) / (r + beam_kpc)
        
        # Asym error
        v_max = np.max(np.abs(v_obs))
        if v_max < 80:
            sigma_asym = beta_dwarf * np.abs(v_obs)
        else:
            sigma_asym = beta_spiral * np.abs(v_obs)
            
        sigma_tot = np.sqrt(v_err**2 + sigma_beam**2 + sigma_asym**2 + sigma_0**2)
        
        # Mask
        mask = r > beam_kpc
        if np.sum(mask) < 5: continue
        
        res = (v_obs - v_model)[mask]
        err = sigma_tot[mask]
        
        chi2_gal = np.sum((res/err)**2)
        ndof = len(res)
        chi2_list.append(chi2_gal/ndof)
        
    return np.median(chi2_list)

# Initial guess (Current defaults)
x0 = [7.0, 8.0, 1.6,   # n(r)
      0.3, 0.10, 0.05, 10.0] # Errors

print(f"Initial Median Chi2/N: {global_loss(x0):.4f}")

# Bounds (Constrained sigma_0)
bounds = [
    (1.0, 20.0), (1.0, 20.0), (0.5, 3.0), # n(r)
    (0.0, 1.0), (0.0, 0.3), (0.0, 0.3), (5.0, 15.0) # Errors (sigma_0 limited)
]

print("Optimizing global profile and error parameters...")
res = minimize(global_loss, x0, bounds=bounds, method='Nelder-Mead', options={'maxiter': 100})

print("\nOptimized Parameters:")
print(f"n(r): A={res.x[0]:.2f}, R0={res.x[1]:.2f}, p={res.x[2]:.2f}")
print(f"Errors: alpha_beam={res.x[3]:.2f}, beta_dwarf={res.x[4]:.3f}, beta_spiral={res.x[5]:.3f}, sigma_0={res.x[6]:.2f}")
print(f"Final Median Chi2/N: {res.fun:.4f}")

