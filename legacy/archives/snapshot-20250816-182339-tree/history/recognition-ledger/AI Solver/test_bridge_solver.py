#!/usr/bin/env python3
"""
Test the bridge solver on a single file
"""

from pathlib import Path
from navier_stokes_classical_bridge_solver import ClassicalBridgeSolver

def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return
    
    solver = ClassicalBridgeSolver(api_key)
    
    # Test on just VorticityBound.lean which should use our bridge lemmas
    target_files = [
        Path("../NavierStokesLedger/VorticityBound.lean"),
    ]
    
    solver.solve_files(target_files)

if __name__ == "__main__":
    main() 