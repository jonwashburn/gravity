#!/usr/bin/env python3
"""
Recognition Science Gravity Solver - O3 Version
Solves sorries in the RS Gravity Lean formalization
"""

import os
import re
import json
from pathlib import Path
from openai import OpenAI
import time

class GravityO3Solver:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "o3"
        
        # Settings
        self.max_iterations = 3
        self.max_completion_tokens = 800
        
        # Statistics
        self.stats = {
            'total_sorries': 0,
            'resolved_sorries': 0,
            'llm_calls': 0,
            'compile_successes': 0,
            'compile_failures': 0
        }
        
        # Cache for successful proofs
        self.proof_cache = {}
        
    def find_sorries(self, file_path: Path):
        """Find all sorries in a file"""
        sorries = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if 'sorry' in line and not line.strip().startswith('--'):
                # Find the theorem/lemma declaration
                declaration_lines = []
                j = i
                while j >= 0:
                    if any(kw in lines[j] for kw in ['theorem', 'lemma', 'def', 'instance']):
                        # Found start of declaration
                        while j <= i:
                            declaration_lines.append(lines[j])
                            j += 1
                        break
                    j -= 1
                    
                if declaration_lines:
                    declaration = ''.join(declaration_lines).strip()
                    
                    # Extract name
                    match = re.search(r'(theorem|lemma|def|instance)\s+(\w+)', declaration)
                    name = match.group(2) if match else 'unknown'
                    
                    sorries.append({
                        'line': i + 1,
                        'name': name,
                        'declaration': declaration,
                        'file': str(file_path)
                    })
                    
        return sorries
    
    def extract_context(self, file_path: Path, sorry_line: int):
        """Extract relevant context around a sorry"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        context = {
            'imports': [],
            'namespace': None,
            'local_defs': [],
            'theorem': ''
        }
        
        # Get imports
        for line in lines[:50]:
            if line.strip().startswith('import'):
                context['imports'].append(line.strip())
                
        # Get namespace
        for line in lines[:sorry_line]:
            if line.strip().startswith('namespace'):
                context['namespace'] = line.strip()
                
        # Get theorem
        theorem_start = None
        for i in range(sorry_line - 1, -1, -1):
            if any(kw in lines[i] for kw in ['theorem', 'lemma', 'def']):
                theorem_start = i
                break
                
        if theorem_start:
            theorem_lines = []
            i = theorem_start
            while i < min(sorry_line + 1, len(lines)):
                theorem_lines.append(lines[i])
                i += 1
            context['theorem'] = ''.join(theorem_lines)
            
        # Get nearby definitions (20 lines before)
        start = max(0, theorem_start - 20) if theorem_start else max(0, sorry_line - 20)
        for i in range(start, sorry_line):
            line = lines[i]
            if any(kw in line for kw in ['def ', 'instance ', 'structure ']):
                def_lines = [line]
                j = i + 1
                indent = len(line) - len(line.lstrip())
                while j < sorry_line and (lines[j].strip() == '' or 
                                         len(lines[j]) - len(lines[j].lstrip()) > indent):
                    def_lines.append(lines[j])
                    j += 1
                context['local_defs'].append(''.join(def_lines))
                
        return context
    
    def generate_proof(self, sorry_info, context, iteration=0, previous_error=None):
        """Generate proof using o3"""
        
        prompt = f"""You are an expert Lean 4 theorem prover working on Recognition Science gravity theory.

## CONTEXT

### Key Constants:
- φ (golden ratio) = (1 + √5) / 2
- E_coh = 0.090 eV (coherence energy)
- l_P = Planck length
- G = gravitational constant

### Imports:
{chr(10).join(context['imports'][:10])}

### Namespace:
{context['namespace'] or 'None'}

### Local Definitions:
{chr(10).join(context['local_defs'][:5])}

### Target Theorem:
{context['theorem']}

"""

        if previous_error:
            prompt += f"""### Previous Error:
{previous_error}

Fix ONLY this specific error.
"""

        prompt += """## YOUR TASK

Generate a Lean 4 proof to replace the sorry.

IMPORTANT RULES:
1. For numerical proofs: Use `norm_num` or `simp; norm_num`
2. For golden ratio: Use `unfold φ; norm_num` or `rw [φ_squared]; norm_num`
3. For definitions: Use `rfl` or `unfold [term]; rfl`
4. For Recognition lengths: Use the definitions in RecognitionLengths.lean
5. Keep proofs concise (under 10 lines preferred)
6. Output ONLY valid Lean 4 code

Common patterns:
- `by norm_num` for simple numerical facts
- `by simp [definition]; norm_num` for unfolding and computing
- `by rfl` for definitional equalities
- `by unfold φ; norm_num` for golden ratio calculations
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Lean 4 expert. Output only valid Lean code."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=self.max_completion_tokens
            )
            
            proof = response.choices[0].message.content.strip()
            self.stats['llm_calls'] += 1
            
            # Clean up the proof
            proof = self.clean_proof(proof)
            
            return proof
            
        except Exception as e:
            print(f"  Error calling o3: {e}")
            return None
    
    def clean_proof(self, proof: str):
        """Clean up generated proof"""
        
        # Remove markdown code blocks
        if '```' in proof:
            match = re.search(r'```(?:lean)?\s*\n(.*?)\n```', proof, re.DOTALL)
            if match:
                proof = match.group(1)
                
        # Remove non-Lean lines
        lines = proof.split('\n')
        clean_lines = []
        
        for line in lines:
            # Keep empty lines, comments, and Lean code
            if (line.strip() == '' or 
                line.strip().startswith('--') or
                line.strip().startswith('by') or
                line.strip().startswith('·') or
                any(line.strip().startswith(kw) for kw in 
                    ['exact', 'apply', 'rw', 'simp', 'intro', 'have', 
                     'use', 'constructor', 'unfold', 'norm_num', 'rfl',
                     'field_simp', 'ring', 'linarith', 'omega'])):
                clean_lines.append(line)
                
        return '\n'.join(clean_lines)
    
    def check_compilation(self, file_path: Path, line_num: int, proof: str):
        """Check if proof compiles by creating temp file"""
        
        # Read original file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Create backup
        backup_path = file_path.with_suffix('.lean.backup')
        with open(backup_path, 'w') as f:
            f.writelines(lines)
            
        # Replace sorry with proof
        if 'sorry' in lines[line_num - 1]:
            lines[line_num - 1] = lines[line_num - 1].replace('sorry', proof)
        else:
            print(f"  Warning: No 'sorry' found on line {line_num}")
            return False, "No sorry found"
            
        # Write modified file
        with open(file_path, 'w') as f:
            f.writelines(lines)
            
        # Run lake build
        import subprocess
        result = subprocess.run(
            ['lake', 'build'],
            capture_output=True,
            text=True,
            cwd=file_path.parent.parent.parent  # Run from recognition-ledger/ directory
        )
        
        success = result.returncode == 0
        error = result.stderr if not success else None
        
        if not success:
            # Restore backup
            with open(backup_path, 'r') as f:
                original = f.read()
            with open(file_path, 'w') as f:
                f.write(original)
                
        # Remove backup
        backup_path.unlink()
        
        return success, error
    
    def extract_error(self, error_msg: str):
        """Extract first meaningful error"""
        if not error_msg:
            return None
            
        # Look for error patterns
        patterns = [
            r'error: (.*?)(?:\n|$)',
            r'failed to synthesize instance(.*?)(?:\n|$)',
            r'type mismatch(.*?)(?:\n|$)',
            r'unknown identifier \'(.*?)\'',
            r'invalid field \'(.*?)\'',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_msg, re.IGNORECASE)
            if match:
                return match.group(0)[:200]
                
        # Return first line with 'error'
        for line in error_msg.split('\n'):
            if 'error' in line.lower():
                return line[:200]
                
        return error_msg[:200]
    
    def solve_sorry(self, file_path: Path, sorry_info):
        """Solve a single sorry with iterations"""
        
        print(f"\n  Solving {sorry_info['name']} at line {sorry_info['line']}...")
        
        # Check cache
        cache_key = f"{file_path.name}:{sorry_info['name']}"
        if cache_key in self.proof_cache:
            print(f"  ✓ Cache hit!")
            proof = self.proof_cache[cache_key]
            self.apply_proof(file_path, sorry_info['line'], proof)
            self.stats['resolved_sorries'] += 1
            return True
            
        # Extract context
        context = self.extract_context(file_path, sorry_info['line'])
        
        # Try multiple iterations
        previous_error = None
        for iteration in range(self.max_iterations):
            print(f"  Iteration {iteration + 1}/{self.max_iterations}...")
            
            # Generate proof
            proof = self.generate_proof(sorry_info, context, iteration, previous_error)
            if not proof:
                continue
                
            print(f"  Generated: {proof[:80]}...")
            
            # Check compilation
            success, error = self.check_compilation(file_path, sorry_info['line'], proof)
            
            if success:
                print(f"  ✓ Success!")
                self.stats['compile_successes'] += 1
                self.stats['resolved_sorries'] += 1
                self.proof_cache[cache_key] = proof
                return True
            else:
                print(f"  ✗ Compilation failed")
                self.stats['compile_failures'] += 1
                previous_error = self.extract_error(error)
                if previous_error:
                    print(f"    Error: {previous_error}")
                    
        return False
    
    def apply_proof(self, file_path: Path, line_num: int, proof: str):
        """Apply proof to file"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        if 'sorry' in lines[line_num - 1]:
            lines[line_num - 1] = lines[line_num - 1].replace('sorry', proof)
            
        with open(file_path, 'w') as f:
            f.writelines(lines)
    
    def solve_file(self, file_path: Path, max_sorries=None):
        """Solve all sorries in a file"""
        
        print(f"\n{'='*60}")
        print(f"Processing: {file_path.name}")
        print('='*60)
        
        # Find sorries
        sorries = self.find_sorries(file_path)
        self.stats['total_sorries'] += len(sorries)
        
        if not sorries:
            print("No sorries found!")
            return
            
        print(f"Found {len(sorries)} sorries")
        
        # Limit if requested
        if max_sorries:
            sorries = sorries[:max_sorries]
            print(f"Processing first {max_sorries} sorries")
            
        # Process each sorry
        for sorry_info in sorries:
            self.solve_sorry(file_path, sorry_info)
            time.sleep(0.5)  # Rate limiting
            
    def report_stats(self):
        """Report final statistics"""
        print(f"\n{'='*60}")
        print("FINAL STATISTICS")
        print('='*60)
        print(f"Total sorries: {self.stats['total_sorries']}")
        print(f"Resolved: {self.stats['resolved_sorries']}")
        print(f"Success rate: {self.stats['resolved_sorries'] / max(1, self.stats['total_sorries']):.1%}")
        print(f"LLM calls: {self.stats['llm_calls']}")
        print(f"Compile successes: {self.stats['compile_successes']}")
        print(f"Compile failures: {self.stats['compile_failures']}")


def main():
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
        
    solver = GravityO3Solver(api_key)
    
    # Target files in priority order
    target_files = [
        # Start with simpler files
        Path("../formal/Gravity/Constants.lean"),
        Path("../formal/Gravity/RecognitionLengths.lean"),
        Path("../formal/Gravity/LNALOpcodes.lean"),
        Path("../formal/Gravity/PhiLadder.lean"),
        
        # Then more complex
        Path("../formal/Gravity/Strain.lean"),
        Path("../formal/Gravity/FortyFiveGap.lean"),
        Path("../formal/Gravity/EightBeatConservation.lean"),
        Path("../formal/Gravity/VoxelWalks.lean"),
        
        # Finally the main theories
        Path("../formal/Gravity/CosmicLedger.lean"),
        Path("../formal/Gravity/LNALGravityTheory.lean"),
        Path("../formal/Gravity/ConsciousnessCompiler.lean"),
        
        # And analysis files
        Path("../formal/Gravity/AnalysisHelpers.lean"),
        Path("../formal/Gravity/FieldEq.lean"),
        Path("../formal/Gravity/InfoStrain.lean"),
    ]
    
    # Process each file
    for file_path in target_files:
        if file_path.exists():
            solver.solve_file(file_path, max_sorries=5)  # Limit per file
        else:
            print(f"\nSkipping {file_path} - not found")
            
    # Report final stats
    solver.report_stats()


if __name__ == "__main__":
    main() 