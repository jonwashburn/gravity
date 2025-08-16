#!/usr/bin/env python3
"""
Analyze and categorize sorry statements in Recognition Science gravity files
This helps identify which sorries can be easily resolved
"""

import re
from pathlib import Path
from collections import defaultdict

class SorryAnalyzer:
    def __init__(self):
        self.sorries_by_type = defaultdict(list)
        self.total_sorries = 0
        
    def categorize_sorry(self, declaration, context):
        """Categorize a sorry based on its context"""
        categories = []
        
        # Check for numerical verification
        if any(word in declaration.lower() for word in ['numerical', 'compute', 'φ^', 'phi^', 'norm_num']):
            categories.append('numerical')
            
        # Check for PDE/analysis proofs
        if any(word in declaration for word in ['fderiv', '∇', 'nabla', 'laplacian', 'div']):
            categories.append('pde_analysis')
            
        # Check for inequality proofs
        if any(op in declaration for op in ['≤', '≥', '<', '>', '≈']):
            categories.append('inequality')
            
        # Check for limit/approximation proofs
        if any(word in declaration for word in ['limit', 'approx', '≈', '≪', '≫']):
            categories.append('approximation')
            
        # Check for existence proofs
        if '∃' in declaration or 'exists' in declaration.lower():
            categories.append('existence')
            
        # Check for uniqueness proofs
        if '∃!' in declaration or 'unique' in declaration.lower():
            categories.append('uniqueness')
            
        # Check for field equation related
        if any(word in declaration for word in ['field_eq', 'field_constraint', 'mond', 'screening']):
            categories.append('field_equation')
            
        # Check for simple rewrites
        if len(declaration.split('\n')) <= 3 and ':=' in declaration:
            categories.append('simple_rewrite')
            
        if not categories:
            categories.append('other')
            
        return categories
        
    def find_sorries(self, file_path: Path):
        """Find all sorries in a file with context"""
        sorries = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except:
            print(f"  Could not read {file_path}")
            return sorries
            
        for i, line in enumerate(lines):
            if 'sorry' in line and not line.strip().startswith('--'):
                # Find the theorem/lemma declaration
                declaration_lines = []
                j = i
                indent_level = None
                
                # Go backwards to find the start
                while j >= 0:
                    current_line = lines[j]
                    # Check if this is a theorem/lemma start
                    if any(kw in current_line for kw in ['theorem', 'lemma', 'def', 'instance']):
                        # Found start, now collect forward
                        k = j
                        while k <= i:
                            declaration_lines.append(lines[k])
                            k += 1
                        break
                    j -= 1
                    
                if declaration_lines:
                    declaration = ''.join(declaration_lines).strip()
                    
                    # Extract name
                    match = re.search(r'(theorem|lemma|def|instance)\s+(\w+)', declaration)
                    name = match.group(2) if match else 'unknown'
                    
                    # Get surrounding context
                    context_start = max(0, i - 10)
                    context_end = min(len(lines), i + 3)
                    context = ''.join(lines[context_start:context_end])
                    
                    # Categorize
                    categories = self.categorize_sorry(declaration, context)
                    
                    sorries.append({
                        'line': i + 1,
                        'name': name,
                        'declaration': declaration,
                        'categories': categories,
                        'file': file_path.name
                    })
                    
        return sorries
        
    def analyze_file(self, file_path: Path):
        """Analyze a single file"""
        print(f"\nAnalyzing: {file_path.name}")
        
        sorries = self.find_sorries(file_path)
        self.total_sorries += len(sorries)
        
        print(f"  Found {len(sorries)} sorry statements")
        
        for sorry in sorries:
            for category in sorry['categories']:
                self.sorries_by_type[category].append(sorry)
                
        return sorries
        
    def print_easy_targets(self):
        """Print sorries that should be easy to resolve"""
        print("\n" + "="*60)
        print("EASY TARGETS (likely to be resolved quickly)")
        print("="*60)
        
        # Numerical verifications
        if 'numerical' in self.sorries_by_type:
            print(f"\nNumerical Verifications ({len(self.sorries_by_type['numerical'])} sorries):")
            for sorry in self.sorries_by_type['numerical'][:5]:  # Show first 5
                print(f"  - {sorry['name']} ({sorry['file']}:{sorry['line']})")
                print(f"    {sorry['declaration'][:80]}...")
                
        # Simple rewrites
        if 'simple_rewrite' in self.sorries_by_type:
            print(f"\nSimple Rewrites ({len(self.sorries_by_type['simple_rewrite'])} sorries):")
            for sorry in self.sorries_by_type['simple_rewrite'][:5]:
                print(f"  - {sorry['name']} ({sorry['file']}:{sorry['line']})")
                
        # Inequalities (often solvable with linarith or similar)
        if 'inequality' in self.sorries_by_type:
            print(f"\nInequalities ({len(self.sorries_by_type['inequality'])} sorries):")
            for sorry in self.sorries_by_type['inequality'][:5]:
                print(f"  - {sorry['name']} ({sorry['file']}:{sorry['line']})")
                
    def print_statistics(self):
        """Print statistics about sorry categories"""
        print("\n" + "="*60)
        print("SORRY STATISTICS BY CATEGORY")
        print("="*60)
        
        # Sort by count
        sorted_categories = sorted(self.sorries_by_type.items(), 
                                 key=lambda x: len(x[1]), 
                                 reverse=True)
        
        for category, sorries in sorted_categories:
            percentage = len(sorries) / self.total_sorries * 100
            print(f"{category:20} {len(sorries):4d} ({percentage:5.1f}%)")
            
        print(f"\nTotal sorries: {self.total_sorries}")
        
    def generate_proof_templates(self):
        """Generate template proofs for common patterns"""
        templates = []
        
        # Numerical verification template
        templates.append({
            'category': 'numerical',
            'template': '''by
  norm_num
  -- If norm_num doesn't work, try:
  -- simp only [phi_val, E_coh_val]
  -- norm_num'''
        })
        
        # Inequality template
        templates.append({
            'category': 'inequality',
            'template': '''by
  -- For simple inequalities:
  linarith
  -- For more complex ones:
  -- apply mul_le_mul
  -- · exact h1
  -- · exact h2
  -- · linarith
  -- · linarith'''
        })
        
        # Field equation template
        templates.append({
            'category': 'field_equation',
            'template': '''by
  -- Expand definitions
  simp [field_operator, mond_function]
  -- Apply the field equation
  rw [field_eq_constraint]
  -- Simplify
  simp [mu_zero_sq, lambda_p]'''
        })
        
        return templates

def main():
    analyzer = SorryAnalyzer()
    
    # Target files
    target_files = [
        Path("../formal/Gravity/FieldEq.lean"),
        Path("../formal/Gravity/Pressure.lean"),
        Path("../formal/Gravity/InfoStrain.lean"),
        Path("../formal/Gravity/XiScreening.lean"),
        Path("../formal/Gravity/MasterTheorem.lean"),
        Path("../formal/Gravity/ExperimentalPredictions.lean"),
        Path("../formal/Gravity/AnalysisHelpers.lean"),
        Path("../formal/Gravity/ConsciousnessGaps.lean"),
        Path("../formal/Gravity/InformationFirst.lean"),
        Path("../formal/ParticleMassesRevised 2.lean"),
        Path("../formal/NumericalVerification 2.lean"),
    ]
    
    print("=== RECOGNITION SCIENCE SORRY ANALYZER ===")
    print("Analyzing sorry statements to identify easy targets...")
    print("-" * 60)
    
    # Analyze all files
    all_sorries = []
    for file_path in target_files:
        if file_path.exists():
            sorries = analyzer.analyze_file(file_path)
            all_sorries.extend(sorries)
        else:
            print(f"\nSkipping (not found): {file_path}")
    
    # Print analysis
    analyzer.print_statistics()
    analyzer.print_easy_targets()
    
    # Generate proof templates
    print("\n" + "="*60)
    print("SUGGESTED PROOF TEMPLATES")
    print("="*60)
    
    templates = analyzer.generate_proof_templates()
    for template in templates:
        print(f"\nFor {template['category']} proofs:")
        print(template['template'])
        
    # Save detailed analysis
    with open("sorry_analysis.txt", "w") as f:
        f.write("# Detailed Sorry Analysis\n\n")
        
        for category, sorries in analyzer.sorries_by_type.items():
            f.write(f"\n## {category.upper()} ({len(sorries)} sorries)\n\n")
            for sorry in sorries:
                f.write(f"### {sorry['name']} ({sorry['file']}:{sorry['line']})\n")
                f.write("```lean\n")
                f.write(sorry['declaration'])
                f.write("\n```\n\n")
                
    print(f"\nDetailed analysis saved to: sorry_analysis.txt")

if __name__ == "__main__":
    main() 