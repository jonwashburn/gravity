#!/usr/bin/env python3
"""
Launch script for the Ultimate Recognition Science Solver
=========================================================

This script launches the solver with optimal settings for autonomous proof generation.
Uses Claude 3.5 Sonnet initially, escalating to Opus when needed.
"""

import subprocess
import sys
import os
from datetime import datetime

def main():
    print("=" * 80)
    print("RECOGNITION SCIENCE AUTOMATED PROOF SYSTEM")
    print("=" * 80)
    print(f"Launch time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Model hierarchy: Claude 3.5 Sonnet → Claude 3 Opus")
    print("Token usage: MAXIMUM (autonomous operation prioritized)")
    print("=" * 80)
    
    # Change to the formal directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("\nStarting Ultimate Autonomous Solver...")
    print("Press Ctrl+C to interrupt if needed\n")
    
    try:
        # Run the solver
        subprocess.run([sys.executable, "ultimate_autonomous_solver.py"], check=True)
    except KeyboardInterrupt:
        print("\n\nSolver interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n\nError running solver: {e}")
        return 1
    
    print("\n" + "=" * 80)
    print("Solver session complete. Check recognition_progress.json for results.")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 