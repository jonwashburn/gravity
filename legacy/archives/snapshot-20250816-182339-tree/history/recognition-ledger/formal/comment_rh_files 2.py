#!/usr/bin/env python3
"""
Script to comment out RH/pattern-layer proof files in Recognition Science project.
This preserves the files for future work while removing them from compilation.
"""

import os
import sys

# Files to comment out
RH_FILES = [
    "DetailedProofs.lean",
    "DetailedProofs_completed.lean", 
    "DetailedProofs_COMPLETE.lean",
    "ExampleCompleteProof.lean",
    "ExampleCompleteProof_COMPLETE.lean"
]

def comment_out_file(filepath):
    """Comment out an entire Lean file by wrapping in block comment."""
    print(f"Commenting out {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already commented
    if content.strip().startswith('/-') and content.strip().endswith('-/'):
        print(f"  Already commented out, skipping.")
        return
    
    # Wrap entire file in block comment
    commented = f"/-\n  TEMPORARILY COMMENTED OUT: RH/Pattern-Layer Proof\n  This file contains work on the Riemann Hypothesis proof from pattern layer.\n  Commented out to focus on core Recognition Science framework.\n  To restore: remove the outer /- and -/ markers.\n\n{content}\n-/"
    
    with open(filepath, 'w') as f:
        f.write(commented)
    
    print(f"  Done.")

def main():
    # Change to formal directory
    if os.path.exists('formal'):
        os.chdir('formal')
    elif not os.path.exists('DetailedProofs.lean'):
        print("Error: Must run from project root or formal directory")
        sys.exit(1)
    
    # Comment out each RH file
    for filename in RH_FILES:
        if os.path.exists(filename):
            comment_out_file(filename)
        else:
            print(f"Warning: {filename} not found, skipping.")
    
    print("\nAll RH/pattern-layer files have been commented out.")
    print("You can now rebuild the project without these files.")

if __name__ == "__main__":
    main() 