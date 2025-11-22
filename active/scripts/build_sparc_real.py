
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

def parse_distance(filepath):
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "Distance" in line:
                    # Format: # Distance = 39.7 Mpc
                    parts = line.split('=')
                    val = parts[1].strip().split()[0]
                    return float(val)
    except:
        pass
    return None

def estimate_H2_mass(M_HI, M_star):
    if M_star <= 0 or M_HI <= 0: return 0
    Z_ratio = (M_star / 1e10) ** 0.3
    return 0.4 * Z_ratio * M_HI

def build():
    print("Building real SPARC master table...")
    # Adjust path to be absolute or relative to where we run
    files = glob.glob('/Users/jonathanwashburn/.cursor/worktrees/gravity/j8Zyv/active/data/Rotmod_LTG/*.dat')
    files = [f for f in files if ' 2' not in f]
    
    print(f"Found {len(files)} files.")
    
    master = {}
    G_kpc = 4.302e-6
    
    for fpath in files:
        name = Path(fpath).stem.replace('_rotmod', '')
        
        try:
            df = pd.read_csv(fpath, sep=r'\s+', comment='#', 
                             names=["rad", "vobs", "verr", "vgas", "vdisk", "vbul", "sbdisk", "sbbul"])
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue
            
        dist = parse_distance(fpath)
        if dist is None:
            dist = 10.0 # Fallback Mpc
            
        # Masses for f_gas
        R_last = df['rad'].iloc[-1]
        v_gas_last = df['vgas'].iloc[-1]
        M_HI = (v_gas_last**2 * R_last / G_kpc) if v_gas_last > 0 else 1e8
        
        v_disk_last = df['vdisk'].iloc[-1]
        v_bul_last = df['vbul'].iloc[-1]
        v_star_sq = v_disk_last**2 + v_bul_last**2
        M_star = (v_star_sq * R_last / G_kpc) if v_star_sq > 0 else 1e9
        
        M_H2 = estimate_H2_mass(M_HI, M_star)
        f_gas = (M_HI + M_H2) / (M_HI + M_H2 + M_star)
        
        # R_d
        if np.any(df['vdisk'] > 0):
            idx_peak = np.argmax(df['vdisk'])
            R_peak = df['rad'].iloc[idx_peak]
            R_d = R_peak / 2.2
        else:
            R_d = 2.0
            
        master[name] = {
            'name': name,
            'r': df['rad'].values,
            'v_obs': df['vobs'].values,
            'data': df, # Store full DF for compatibility
            'distance': dist, # Mpc
            'f_gas_true': f_gas,
            'R_d': R_d
        }
        
    print(f"Built {len(master)} galaxies.")
    
    # Save to active/scripts directory
    out_path = '/Users/jonathanwashburn/.cursor/worktrees/gravity/j8Zyv/active/scripts/sparc_master_real.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(master, f)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    build()

