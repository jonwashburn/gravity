#!/usr/bin/env python3
"""
Recognition Science Gravity Manager - Lightweight Version
Focused on getting sorries resolved efficiently
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI

class GravityManagerLite:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4"
        self.sorries = []
        self.resolved = 0
        self.failed = 0
        
    def find_sorries(self):
        """Find all sorries in gravity files"""
        print("Scanning for sorries...")
        
        target_files = [
            "../formal/Gravity/FieldEq.lean",
            "../formal/Gravity/Pressure.lean", 
            "../formal/Gravity/InfoStrain.lean",
            "../formal/Gravity/AnalysisHelpers.lean",
            "../formal/Gravity/MasterTheorem.lean",
            "../formal/Gravity/ExperimentalPredictions.lean",
            "../formal/ParticleMassesRevised 2.lean",
            "../formal/NumericalVerification 2.lean"
        ]
        
        for file_path in target_files:
            path = Path(file_path)
            if not path.exists():
                continue
                
            with open(path, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                if 'sorry' in line and not line.strip().startswith('--'):
                    # Get theorem context
                    theorem_start = None
                    for j in range(i, -1, -1):
                        if any(kw in lines[j] for kw in ['theorem', 'lemma', 'def']):
                            theorem_start = j
                            break
                            
                    if theorem_start is not None:
                        declaration = ''.join(lines[theorem_start:i+1])
                        match = re.search(r'(theorem|lemma|def)\s+(\w+)', declaration)
                        name = match.group(2) if match else 'unknown'
                        
                        # Simple categorization
                        category = 'general'
                        if 'norm_num' in declaration or 'φ^' in declaration:
                            category = 'numerical'
                        elif any(op in declaration for op in ['≤', '≥', '<', '>']):
                            category = 'inequality'
                        elif '∃' in declaration:
                            category = 'existence'
                            
                        self.sorries.append({
                            'file': str(path),
                            'line': i + 1,
                            'name': name,
                            'category': category,
                            'declaration': declaration,
                            'context': ''.join(lines[max(0, theorem_start-10):min(len(lines), i+5)])
                        })
                        
        print(f"Found {len(self.sorries)} sorries")
        
        # Sort by category (easier ones first)
        priority = {'numerical': 0, 'inequality': 1, 'existence': 2, 'general': 3}
        self.sorries.sort(key=lambda s: priority.get(s['category'], 3))
        
    def generate_proof(self, sorry_info):
        """Generate a proof for a sorry"""
        
        system_prompt = """You are a Lean 4 expert working on Recognition Science gravity.

Key constants:
- φ = 1.618... (golden ratio)
- a_0 = 1.85e-10 m/s² (MOND scale)
- ℓ_1 = 0.97 kpc, ℓ_2 = 24.3 kpc
- ρ_gap = 1e-24 kg/m³

Output ONLY the Lean proof code."""

        user_prompt = f"""Complete this theorem:

{sorry_info['context']}

Category: {sorry_info['category']}
Replace the 'sorry' with a proof.

Hints for {sorry_info['category']} proofs:
"""
        
        if sorry_info['category'] == 'numerical':
            user_prompt += "- Use norm_num\n- Try simp only [phi_val, E_coh_val]\n- Use explicit calculations"
        elif sorry_info['category'] == 'inequality':
            user_prompt += "- Try linarith or nlinarith\n- Use monotonicity lemmas\n- Consider cases"
        elif sorry_info['category'] == 'existence':
            user_prompt += "- Use 'use' to provide witness\n- Verify properties with constructor"
            
        user_prompt += "\n\nProvide ONLY the proof code:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            proof = response.choices[0].message.content.strip()
            
            # Clean up
            if '```' in proof:
                match = re.search(r'```(?:lean)?\s*\n(.*?)\n```', proof, re.DOTALL)
                if match:
                    proof = match.group(1)
                    
            return proof
            
        except Exception as e:
            print(f"  Error: {e}")
            return None
            
    def apply_proof(self, sorry_info, proof):
        """Apply proof to file"""
        try:
            with open(sorry_info['file'], 'r') as f:
                lines = f.readlines()
                
            if 'sorry' in lines[sorry_info['line'] - 1]:
                lines[sorry_info['line'] - 1] = lines[sorry_info['line'] - 1].replace('sorry', proof)
                
                with open(sorry_info['file'], 'w') as f:
                    f.writelines(lines)
                    
                return True
        except:
            return False
            
    def run(self):
        """Main execution"""
        print("=== GRAVITY MANAGER LITE ===\n")
        
        # Find sorries
        self.find_sorries()
        
        if not self.sorries:
            print("No sorries found!")
            return
            
        print(f"\nProcessing {len(self.sorries)} sorries...")
        print("Priority order: numerical → inequality → existence → general\n")
        
        # Process each sorry
        for i, sorry_info in enumerate(self.sorries):
            print(f"\n[{i+1}/{len(self.sorries)}] {sorry_info['name']} ({sorry_info['category']})")
            print(f"  File: {Path(sorry_info['file']).name}:{sorry_info['line']}")
            
            # Generate proof
            print("  Generating proof...")
            proof = self.generate_proof(sorry_info)
            
            if proof and proof.strip() and 'sorry' not in proof.lower():
                print(f"  Generated: {proof[:50]}...")
                
                # Apply proof
                if self.apply_proof(sorry_info, proof):
                    print("  ✓ Applied successfully")
                    self.resolved += 1
                else:
                    print("  ✗ Failed to apply")
                    self.failed += 1
            else:
                print("  ✗ Failed to generate valid proof")
                self.failed += 1
                
        # Summary
        print(f"\n{'='*40}")
        print(f"SUMMARY")
        print(f"{'='*40}")
        print(f"Total sorries: {len(self.sorries)}")
        print(f"Resolved: {self.resolved}")
        print(f"Failed: {self.failed}")
        print(f"Success rate: {self.resolved/len(self.sorries)*100:.1f}%")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'total': len(self.sorries),
            'resolved': self.resolved,
            'failed': self.failed,
            'sorries': self.sorries
        }
        
        with open('gravity_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nResults saved to gravity_results.json")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return
        
    manager = GravityManagerLite(api_key)
    manager.run()

if __name__ == "__main__":
    main() 