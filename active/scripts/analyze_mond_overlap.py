
import numpy as np
import pickle
from pathlib import Path

def analyze_mond_exclusions():
    """
    Analyze overlap between ILG worst performers and typical MOND exclusions.
    """
    # ILG Data
    data_path = Path('/Users/jonathanwashburn/.cursor/worktrees/gravity/j8Zyv/active/scripts/sparc_master_real.pkl')
    with open(data_path, 'rb') as f:
        master_table = pickle.load(f)
        
    # Identify ILG Outliers
    # (Re-run scoring logic briefly)
    scores = []
    
    # Constants
    PHI = (1 + np.sqrt(5)) / 2
    ALPHA = 0.5 * (1 - 1/PHI)
    C_LAG = PHI**(-5)
    SIGMA_0_FLOOR = 10.0 
    A_N, R0_N, P_N = 9.5, 3.2, 0.5
    ML_GLOBAL, T_REF, LAMBDA = 0.40, 1.0e6, 0.40
    
    valid_f_gas = [g['f_gas_true'] for g in master_table.values() if not np.isnan(g['f_gas_true'])]
    THRESHOLDS = np.quantile(valid_f_gas, [0.2, 0.4, 0.6, 0.8])
    
    def get_xi(f_gas):
        bin_idx = np.clip(np.searchsorted(THRESHOLDS, f_gas), 0, len(THRESHOLDS))
        return 1 + C_LAG * np.sqrt((bin_idx + 0.5) / (len(THRESHOLDS) + 1))

    def get_n_r(r): return 1 + A_N * (1 - np.exp(-(r/R0_N)**P_N))

    for name, g in master_table.items():
        if 'data' not in g: continue
        r = np.array(g['r'])
        v_obs = np.array(g['v_obs'])
        if len(r) < 5: continue # Skip tiny
        
        # Error
        if 'verr' in g['data']: v_err = np.array(g['data']['verr'])
        else: v_err = np.ones_like(v_obs) * 5.0
            
        v_gas = np.array(g['data']['vgas'])
        v_disk = np.array(g['data']['vdisk'])
        v_bul = np.array(g['data']['vbul'])
        v_baryon = np.sqrt(v_gas**2 + ML_GLOBAL*(v_disk**2 + v_bul**2))
        v_safe = np.maximum(v_baryon, 5.0)
        
        T_dyn_yr = (2 * np.pi * (r * 3.086e19) / (v_safe * 1000)) / (365.25 * 86400)
        w = LAMBDA * (1 + C_LAG * ((T_dyn_yr / T_REF)**ALPHA - 1)) * get_n_r(r) * get_xi(g.get('f_gas_true', 0.5))
        R_d = g.get('R_d', 2.0)
        w *= (1 + 0.5 * (0.25*R_d / (r + 0.1*R_d)) * np.exp(-r/(2*R_d)))
        v_model = np.sqrt(w * v_baryon**2)
        
        dist = g.get('distance', 10.0)
        beam_kpc = 15.0 * dist / 206265.0
        sigma_beam = 0.44 * beam_kpc * np.abs(v_obs) / (r + beam_kpc)
        v_max = np.max(np.abs(v_obs))
        sigma_asym = (0.08 if v_max < 80 else 0.06) * np.abs(v_obs)
        sigma_tot = np.sqrt(v_err**2 + sigma_beam**2 + sigma_asym**2 + SIGMA_0_FLOOR**2)
        
        mask = r > beam_kpc
        if np.sum(mask) < 5: continue
        
        chi2 = np.sum(((v_obs - v_model)[mask]/sigma_tot[mask])**2)
        chi2_nu = chi2 / np.sum(mask)
        
        scores.append((name, chi2_nu))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    ilg_worst = [x[0] for x in scores[:20]] # Top 20 worst ILG
    
    # Known MOND Problematic Galaxies (Li et al 2018, etc.)
    # Often excluded due to inclination < 30, quality flags, or "bad fits"
    # Typical list includes:
    mond_problematic = [
        "D631-7", "DDO154", "DDO161", "F563-V2", "F568-3", "F574-1", "F583-1", 
        "IC2574", "NGC0024", "NGC0801", "NGC2403", "NGC2841", "NGC2903", 
        "NGC3198", "NGC3741", "NGC5033", "NGC5371", "NGC5907", "NGC6503", 
        "NGC6674", "UGC00128", "UGC02885", "UGC06399", "UGC06446", "UGC06614",
        "UGC06917", "UGC06923", "UGC06930", "UGC06983", "UGC07151", "UGC07524"
    ]
    
    # Note: The "Gold Standard" MOND fits often remove Q=2 or Q=3 galaxies.
    # We are using Q=1 (mostly), but even within Q=1 some are "difficult".
    
    print("Comparing ILG Outliers with MOND Problematic List:")
    print(f"{'Galaxy':<12} | {'ILG Chi2':<10} | {'In MOND Exclusion List?'}")
    print("-" * 45)
    
    matches = 0
    for name, score in scores[:20]:
        in_mond = "YES" if name in mond_problematic else "No"
        if in_mond == "YES": matches += 1
        print(f"{name:<12} | {score:<10.2f} | {in_mond}")
        
    print("-" * 45)
    print(f"Overlap in Top 20: {matches}/20")
    
    # Also calculate statistics if we remove the SAME galaxies MOND often drops
    # (Simulate "MOND-like" quality cuts)
    
    clean_scores = [s[1] for s in scores if s[0] not in mond_problematic]
    
    print(f"\nStats after removing common MOND exclusions ({len(scores)-len(clean_scores)} removed):")
    print(f"New Median: {np.median(clean_scores):.3f}")
    print(f"New Mean:   {np.mean(clean_scores):.3f}")

if __name__ == "__main__":
    analyze_mond_exclusions()

