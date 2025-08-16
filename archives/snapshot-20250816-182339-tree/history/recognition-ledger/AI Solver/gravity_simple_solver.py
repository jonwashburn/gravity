#!/usr/bin/env python3
"""
Simplified Recognition Science Gravity Solver
Standalone version for solving sorry statements in RS gravity proofs
"""

import os
import re
from pathlib import Path
from openai import OpenAI

class SimpleGravitySolver:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4"  # Use gpt-4 if o3 not available
        
        # Statistics
        self.stats = {
            'total_sorries': 0,
            'attempted': 0,
            'generated': 0
        }
        
    def find_sorries(self, file_path: Path):
        """Find all sorries in a file with context"""
        sorries = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if 'sorry' in line and not line.strip().startswith('--'):
                # Find the theorem/lemma declaration
                declaration_lines = []
                j = i
                while j >= 0:
                    if any(kw in lines[j] for kw in ['theorem', 'lemma', 'def']):
                        # Found start of declaration
                        while j <= i:
                            declaration_lines.append(lines[j])
                            j += 1
                        break
                    j -= 1
                    
                if declaration_lines:
                    declaration = ''.join(declaration_lines).strip()
                    
                    # Extract name
                    match = re.search(r'(theorem|lemma|def)\s+(\w+)', declaration)
                    name = match.group(2) if match else 'unknown'
                    
                    # Get surrounding context
                    context_start = max(0, i - 20)
                    context_end = min(len(lines), i + 5)
                    context = ''.join(lines[context_start:context_end])
                    
                    sorries.append({
                        'line': i + 1,
                        'name': name,
                        'declaration': declaration,
                        'context': context,
                        'file': str(file_path)
                    })
                    
        return sorries
        
    def generate_proof(self, sorry_info):
        """Generate a proof for a sorry statement"""
        
        prompt = f"""You are an expert Lean 4 theorem prover working on Recognition Science gravity theory.

## RECOGNITION SCIENCE BACKGROUND

Key principles:
- Golden ratio φ = (1 + √5)/2 = 1.618... emerges from cost minimization
- Recognition pressure P drives gravity (not mass directly)
- MOND behavior emerges at low accelerations
- All parameters derive from first principles

Key constants:
- a_0 = 1.85e-10 m/s² (MOND scale)
- ℓ_1 = 0.97 kpc, ℓ_2 = 24.3 kpc (recognition lengths)
- ρ_gap = 1e-24 kg/m³ (screening threshold)

## CONTEXT

{sorry_info['context']}

## TARGET THEOREM

{sorry_info['declaration']}

## YOUR TASK

Generate a Lean 4 proof to replace the 'sorry'. Consider:
1. What needs to be proven
2. What lemmas/theorems are available
3. Common proof techniques (simp, rw, apply, exact, calc)
4. For numerical proofs use norm_num
5. For inequalities use standard tactics

Output ONLY the proof code that will replace 'sorry'. Start with 'by' if it's a tactic proof.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Lean 4 expert. Output only valid Lean proof code."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            proof = response.choices[0].message.content.strip()
            
            # Clean up the proof
            if '```' in proof:
                # Extract content between ```
                match = re.search(r'```(?:lean)?\s*\n(.*?)\n```', proof, re.DOTALL)
                if match:
                    proof = match.group(1)
                    
            return proof
            
        except Exception as e:
            print(f"  Error generating proof: {e}")
            return None
            
    def process_file(self, file_path: Path):
        """Process a single file"""
        print(f"\nProcessing: {file_path.name}")
        
        if not file_path.exists():
            print(f"  File not found: {file_path}")
            return
            
        sorries = self.find_sorries(file_path)
        self.stats['total_sorries'] += len(sorries)
        
        print(f"  Found {len(sorries)} sorry statements")
        
        results = []
        
        for sorry_info in sorries:
            print(f"\n  Theorem: {sorry_info['name']} (line {sorry_info['line']})")
            self.stats['attempted'] += 1
            
            proof = self.generate_proof(sorry_info)
            
            if proof:
                print(f"  Generated proof:")
                print(f"  {proof[:100]}..." if len(proof) > 100 else f"  {proof}")
                self.stats['generated'] += 1
                
                results.append({
                    'theorem': sorry_info['name'],
                    'line': sorry_info['line'],
                    'original': sorry_info['declaration'],
                    'proof': proof
                })
            else:
                print(f"  Failed to generate proof")
                
        return results
        
    def save_results(self, all_results, output_file="gravity_proofs.txt"):
        """Save all generated proofs to a file"""
        with open(output_file, 'w') as f:
            f.write("# Generated Proofs for Recognition Science Gravity\n\n")
            
            for file_results in all_results:
                f.write(f"\n## File: {file_results['file']}\n\n")
                
                for result in file_results['results']:
                    f.write(f"### {result['theorem']} (line {result['line']})\n\n")
                    f.write("Original:\n```lean\n")
                    f.write(result['original'])
                    f.write("\n```\n\n")
                    f.write("Generated proof:\n```lean\n")
                    f.write(result['proof'])
                    f.write("\n```\n\n")
                    f.write("-" * 60 + "\n\n")
                    
    def report_statistics(self):
        """Report statistics"""
        print(f"\n{'='*60}")
        print("STATISTICS")
        print('='*60)
        print(f"Total sorries found: {self.stats['total_sorries']}")
        print(f"Proofs attempted: {self.stats['attempted']}")
        print(f"Proofs generated: {self.stats['generated']}")
        if self.stats['attempted'] > 0:
            success_rate = self.stats['generated'] / self.stats['attempted'] * 100
            print(f"Success rate: {success_rate:.1f}%")
                
def main():
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    solver = SimpleGravitySolver(api_key)
    
    # Target files - high priority gravity files
    target_files = [
        "recognition-ledger/formal/Gravity/FieldEq.lean",
        "recognition-ledger/formal/Gravity/Pressure.lean",
        "recognition-ledger/formal/Gravity/InfoStrain.lean",
    ]
    
    # Convert to Path objects
    target_files = [Path(f) for f in target_files]
    
    print("=== SIMPLE GRAVITY SOLVER ===")
    print(f"Using model: {solver.model}")
    print("-" * 60)
    
    all_results = []
    
    for file_path in target_files:
        results = solver.process_file(file_path)
        if results:
            all_results.append({
                'file': str(file_path),
                'results': results
            })
    
    # Save results
    if all_results:
        output_file = "gravity_proofs_generated.txt"
        solver.save_results(all_results, output_file)
        print(f"\nResults saved to: {output_file}")
    
    # Report statistics
    solver.report_statistics()

if __name__ == "__main__":
    main() 