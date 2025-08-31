#!/usr/bin/env python3
"""
Quick Test of Bayesian Optimization
===================================
Test with 10 trials on 30 galaxies
"""

import sys
sys.path.append('.')

# Modify configuration for quick test
import bayes_global_optimization as bgo

# Override settings for quick test
bgo.N_TRIALS = 10
bgo.MAX_GALAXIES = 30

if __name__ == "__main__":
    print("Running quick test with 10 trials on 30 galaxies...")
    bgo.main() 