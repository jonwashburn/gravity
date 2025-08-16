#!/usr/bin/env python3
"""
Apply simple fixes to sorry statements that are very likely to work
Only applies the safest, most straightforward resolutions
"""

import re
from pathlib import Path

class SimpleFixer:
    def __init__(self):
        self.fixes_applied = 0
        self.files_modified = set()
        
    def apply_simple_numerical_fix(self, file_path, line_num, declaration):
        """Apply norm_num to simple numerical proofs"""
        # Only for very simple cases
        if 'norm_num' in declaration or ('abs' in declaration and '<' in declaration and '0.0' in declaration):
            return 'by norm_num'
        return None
        
    def apply_simple_inequality_fix(self, file_path, line_num, declaration):
        """Apply simple tactics to basic inequalities"""
        # Only for inequalities between constants
        if re.search(r'^\s*\d+\.?\d*\s*[<>≤≥]\s*\d+\.?\d*\s*:=', declaration):
            return 'by norm_num'
        # Simple constructor for conjunctions
        if '∧' in declaration and declaration.count('∧') == 1 and declaration.count(':=') == 0:
            return 'by constructor <;> norm_num'
        return None
        
    def process_file(self, file_path):
        """Process a file and apply simple fixes"""
        print(f"\nProcessing: {file_path.name}")
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except:
            print(f"  Could not read {file_path}")
            return 0
            
        modifications = []
        fixes_in_file = 0
        
        for i, line in enumerate(lines):
            if 'sorry' in line and not line.strip().startswith('--'):
                # Get declaration context
                declaration_lines = []
                j = i
                while j >= 0:
                    if any(kw in lines[j] for kw in ['theorem', 'lemma', 'def']):
                        while j <= i:
                            declaration_lines.append(lines[j])
                            j += 1
                        break
                    j -= 1
                    
                if declaration_lines:
                    declaration = ''.join(declaration_lines)
                    
                    # Try simple fixes
                    fix = None
                    
                    # Try numerical fix first
                    fix = self.apply_simple_numerical_fix(file_path, i + 1, declaration)
                    
                    # If no numerical fix, try inequality
                    if not fix:
                        fix = self.apply_simple_inequality_fix(file_path, i + 1, declaration)
                        
                    if fix:
                        # Store the modification
                        modifications.append((i, fix))
                        fixes_in_file += 1
                        
                        # Extract theorem name for logging
                        match = re.search(r'(theorem|lemma|def)\s+(\w+)', declaration)
                        name = match.group(2) if match else 'unknown'
                        print(f"  Fixed: {name} (line {i + 1}) with '{fix}'")
                        
        # Apply all modifications in reverse order to preserve line numbers
        if modifications:
            for line_idx, fix in reversed(modifications):
                lines[line_idx] = lines[line_idx].replace('sorry', fix)
                
            # Write back to file
            with open(file_path, 'w') as f:
                f.writelines(lines)
                
            self.files_modified.add(file_path.name)
            self.fixes_applied += fixes_in_file
            print(f"  Applied {fixes_in_file} fixes")
            
        return fixes_in_file

def main():
    fixer = SimpleFixer()
    
    # Target specific files with simple sorries
    target_files = [
        Path("../formal/Gravity/AnalysisHelpers.lean"),
        Path("../formal/Gravity/ExperimentalPredictions.lean"),
    ]
    
    print("=== SIMPLE FIX APPLIER ===")
    print("Applying only the safest, most straightforward fixes...")
    print("-" * 60)
    
    for file_path in target_files:
        if file_path.exists():
            fixer.process_file(file_path)
        else:
            print(f"\nSkipping (not found): {file_path}")
            
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Total fixes applied: {fixer.fixes_applied}")
    print(f"Files modified: {len(fixer.files_modified)}")
    if fixer.files_modified:
        print("Modified files:")
        for fname in fixer.files_modified:
            print(f"  - {fname}")
    
    print("\nNote: Only the simplest, most likely to work fixes were applied.")
    print("More complex sorries still need manual attention or AI assistance.")

if __name__ == "__main__":
    main() 