#!/usr/bin/env python3
"""
Run the solver with delays to avoid rate limiting
"""

import subprocess
import sys
import os
import time

def main():
    print("Starting Recognition Science Solver with rate limit protection...")
    print("This will add a 30-second delay between iterations to avoid API rate limits")
    print("=" * 80)
    
    # Set environment variable to add delay
    os.environ['SOLVER_DELAY'] = '30'
    
    # Change to the formal directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Run the solver
        subprocess.run([sys.executable, "ultimate_autonomous_solver.py"], check=True)
    except KeyboardInterrupt:
        print("\n\nSolver interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n\nError running solver: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 