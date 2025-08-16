#!/usr/bin/env python3
"""
Parameter Tuner for Recognition Science Gravity Framework
=========================================================
Performs a simple grid search over key parameters to minimize average
χ²/N across a subset of SPARC galaxies.
"""

import numpy as np
import itertools
import json
from typing import Dict
from rs_gravity_final_working import FinalGravitySolver, GalaxyData
import glob, os

# Paths
DATA_DIR = "Rotmod_LTG"
RESULTS_FILE = "tuning_results.json"

# Parameter grids
lambda_eff_grid = [50e-6, 63e-6, 80e-6]  # meters
h_scale_grid = [200, 300, 400]  # pc
beta_scale_grid = [0.9, 1.0, 1.1]  # scale factor on β

# Load small subset of SPARC galaxies for fast tuning (e.g., 10 galaxies)
all_files = glob.glob(os.path.join(DATA_DIR, "*_rotmod.dat"))[:10]

def load_galaxy(file_path: str) -> GalaxyData:
    name = os.path.basename(file_path).replace("_rotmod.dat", "")
    data = np.loadtxt(file_path, skiprows=1)
    R_kpc = data[:, 0]
    v_obs = data[:, 1]
    v_err = np.maximum(0.03 * v_obs, 2.0)
    sigma_gas = data[:, 5] * 1.33  # He correction
    sigma_disk = data[:, 6] * 0.5
    return GalaxyData(name, R_kpc, v_obs, v_err, sigma_gas, sigma_disk)

# Prepare galaxies
GALAXIES = [load_galaxy(fp) for fp in all_files]

best_config: Dict = {}
best_chi2 = np.inf

results = []

for lambda_eff in lambda_eff_grid:
    for h_pc in h_scale_grid:
        for beta_scale in beta_scale_grid:
            # Instantiate solver with modified parameters
            solver = FinalGravitySolver()
            solver.lambda_eff = lambda_eff  # override
            solver.h_scale = h_pc * 3.086e16  # convert pc to m
            solver.beta = solver.beta * beta_scale  # scale β
            
            chi2_list = []
            for g in GALAXIES:
                res = solver.solve_galaxy(g)
                chi2_list.append(res['chi2_reduced'])
            mean_chi2 = float(np.mean(chi2_list))
            config = {
                "lambda_eff": lambda_eff,
                "h_pc": h_pc,
                "beta_scale": beta_scale,
                "mean_chi2": mean_chi2
            }
            results.append(config)
            print(f"Tested {config}")
            if mean_chi2 < best_chi2:
                best_chi2 = mean_chi2
                best_config = config

# Save results
with open(RESULTS_FILE, "w") as f:
    json.dump({"best_config": best_config, "all_results": results}, f, indent=2)

print("\nBest configuration:")
print(best_config) 