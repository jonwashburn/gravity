#!/usr/bin/env python3
"""
Recognition Science Gravity Solver - O3 Version
Implements AI-based proof completion for RS gravity theorems
"""

import os
import re
import json
from pathlib import Path
from openai import OpenAI
from compile_checker import CompileChecker
from context_extractor import ContextExtractor
from proof_cache import ProofCache

class GravityRSO3Solver:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "o3"
        
        # Initialize components
        self.cache = ProofCache()
        self.compiler = CompileChecker()
        self.extractor = ContextExtractor()
        
        # Advanced settings
        self.max_iterations = 3  # Per sorry
        self.max_completion_tokens = 800  # Increased for complex proofs
        self.enable_reinforcement = True
        self.enable_minimal_context = True
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'llm_calls': 0,
            'compile_successes': 0,
            'compile_failures': 0,
            'total_sorries': 0,
            'resolved_sorries': 0,
            'iterations_per_sorry': []
        }
        
        # Load golden examples specific to RS gravity
        self.golden_examples = self.load_golden_examples()
        
        # Failure pattern cache
        self.failure_patterns = {}
        
        # Dependency graph
        self.dependency_graph = {}
        
    def load_golden_examples(self):
        """Load successfully proven theorems as examples"""
        examples = []
        
        # Example 1: Golden ratio property
        examples.append({
            'declaration': 'theorem golden_ratio_squared : φ * φ = φ + 1',
            'proof': '''by
  rw [phi_def]
  field_simp
  ring'''
        })
        
        # Example 2: Recognition cost positivity
        examples.append({
            'declaration': 'theorem recognition_cost_positive (x : ℝ) (hx : x > 0) : J x > 0',
            'proof': '''by
  simp [J]
  apply div_pos
  · apply add_pos hx
    exact div_pos one_pos hx
  · exact two_pos'''
        })
        
        # Example 3: MOND function bounds
        examples.append({
            'declaration': 'theorem mond_function_bounded (u : ℝ) : 0 ≤ mond_function u ∧ mond_function u ≤ 1',
            'proof': '''by
  simp [mond_function]
  constructor
  · apply div_nonneg
    · exact abs_nonneg u
    · apply sqrt_nonneg
  · rw [div_le_one]
    · apply le_sqrt_of_sq_le_sq
      · exact abs_nonneg u
      · simp
        exact le_add_of_nonneg_right (sq_nonneg _)
    · apply sqrt_pos
      exact add_pos_of_pos_of_nonneg one_pos (sq_nonneg _)'''
        })
        
        # Example 4: Screening function behavior
        examples.append({
            'declaration': 'theorem screening_approaches_one (ρ : ℝ) (hρ : ρ > 0) : ρ ≫ ρ_gap → screening_function ρ hρ ≈ 1',
            'proof': '''by
  intro h_large
  simp [screening_function, ρ_gap]
  -- For ρ ≫ ρ_gap, 1/(1 + ρ_gap/ρ) ≈ 1
  have h_small : ρ_gap / ρ < 0.1 := by
    rw [div_lt_iff hρ]
    exact h_large
  calc
    1 / (1 + ρ_gap / ρ) ≥ 1 / (1 + 0.1) := by
      apply div_le_div_of_le_left one_pos
      · exact add_pos one_pos (div_pos ρ_gap_pos hρ)
      · exact add_le_add_left (le_of_lt h_small) 1
    _ > 0.9 := by norm_num'''
        })
        
        # Example 5: Field equation structure
        examples.append({
            'declaration': 'theorem field_eq_linearity (P : ℝ → ℝ) (c : ℝ) : field_operator (c • P) = c • field_operator P',
            'proof': '''by
  ext x
  simp [field_operator, Pi.smul_apply]
  -- Linearity follows from linearity of derivatives
  rw [fderiv_const_smul, fderiv_const_smul]
  simp'''
        })
        
        return examples
        
    def extract_minimal_context(self, file_path: Path, sorry_line: int):
        """Extract only essential context for the proof"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        context = {
            'imports': [],
            'dependencies': [],
            'target_theorem': '',
            'local_definitions': [],
            'rs_constants': []
        }
        
        # Get imports
        for line in lines[:30]:  # More imports in RS files
            if line.strip().startswith('import'):
                context['imports'].append(line.strip())
                
        # Extract RS-specific constants
        rs_constants = ['φ', 'E_coh', 'τ_0', 'ℓ_1', 'ℓ_2', 'a_0', 'ρ_gap', 'λ_p', 'μ_0']
        for const in rs_constants:
            for i, line in enumerate(lines):
                if f'def {const}' in line or f'notation "{const}"' in line:
                    context['rs_constants'].append(line.strip())
                    break
                    
        # Find theorem declaration
        theorem_start = None
        for i in range(sorry_line - 1, -1, -1):
            if any(kw in lines[i] for kw in ['theorem', 'lemma', 'def']):
                theorem_start = i
                break
                
        if theorem_start:
            # Extract full theorem
            theorem_lines = []
            i = theorem_start
            while i < sorry_line:
                theorem_lines.append(lines[i])
                i += 1
            context['target_theorem'] = ''.join(theorem_lines)
            
            # Extract theorem name for dependency analysis
            match = re.search(r'(theorem|lemma)\s+(\w+)', context['target_theorem'])
            if match:
                theorem_name = match.group(2)
                
                # Find what this theorem uses
                used_names = re.findall(r'\b([A-Z]\w+|[a-z]+_\w+)\b', context['target_theorem'])
                context['dependencies'] = list(set(used_names))
                
        # Get definitions of dependencies
        for dep in context['dependencies'][:10]:  # Limit to avoid huge context
            for i, line in enumerate(lines):
                if f'def {dep}' in line or f'theorem {dep}' in line or f'lemma {dep}' in line:
                    # Extract this definition
                    def_lines = [line]
                    j = i + 1
                    indent = len(line) - len(line.lstrip())
                    while j < len(lines) and (lines[j].strip() == '' or len(lines[j]) - len(lines[j].lstrip()) > indent):
                        def_lines.append(lines[j])
                        j += 1
                    context['local_definitions'].append(''.join(def_lines))
                    break
                    
        return context
        
    def generate_proof_with_cot(self, sorry_info, context, iteration=0, previous_error=None):
        """Generate proof with chain-of-thought reasoning"""
        
        # Build the prompt
        prompt = f"""You are an expert Lean 4 theorem prover working on Recognition Science gravity theory.

## RECOGNITION SCIENCE CONTEXT

Recognition Science derives all physics from information-theoretic principles:
- Golden ratio φ = 1.618... emerges from cost minimization
- Recognition pressure P = J_in - J_out drives gravity
- MOND behavior emerges naturally at low accelerations
- Screening function S(ρ) = 1/(1 + ρ_gap/ρ) suppresses effects at low density
- All parameters derive from first principles (zero free parameters)

Key constants:
- φ (golden ratio) = (1 + √5)/2
- E_coh = 0.090 eV (coherence quantum)
- a_0 = 1.85e-10 m/s² (MOND scale from recognition)
- ℓ_1 = 0.97 kpc, ℓ_2 = 24.3 kpc (recognition lengths)
- ρ_gap = 1e-24 kg/m³ (screening threshold)

## CONTEXT

### Imports:
{chr(10).join(context['imports'])}

### RS Constants:
{chr(10).join(context['rs_constants'])}

### Key Definitions:
{chr(10).join(context['local_definitions'][:5])}

### Golden Examples:
"""
        
        for ex in self.golden_examples:
            prompt += f"\nExample:\n{ex['declaration']}\n{ex['proof']}\n"
            
        prompt += f"""
### Target Theorem:
{context['target_theorem']}

"""

        if previous_error:
            prompt += f"""### Previous Attempt Error:
{previous_error}

Please fix ONLY the specific error mentioned above.
"""

        prompt += """## YOUR TASK

First, write a comment outlining your proof strategy:
-- Step 1: What needs to be shown
-- Step 2: Key lemmas to use
-- Step 3: How to connect them

Then write the Lean 4 proof code.

IMPORTANT:
- Use Recognition Science constants (φ, E_coh, a_0, ℓ_1, ℓ_2, ρ_gap)
- For numerical calculations, use norm_num or explicit computation
- For PDE proofs, use the provided analysis helpers
- Keep proof under 15 lines if possible
- Match the style of the golden examples
- Output ONLY Lean code (comments starting with -- are OK)
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Lean 4 expert specializing in Recognition Science. Output only valid Lean code."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=self.max_completion_tokens
            )
            
            proof = response.choices[0].message.content.strip()
            
            # Post-process to remove non-Lean content
            proof = self.post_process_proof(proof)
            
            return proof
            
        except Exception as e:
            print(f"  Error calling o3: {e}")
            return None
            
    def post_process_proof(self, proof: str):
        """Clean up proof to ensure only Lean code"""
        
        # Remove markdown code blocks
        if '```' in proof:
            # Extract content between ```lean and ```
            match = re.search(r'```(?:lean)?\s*\n(.*?)\n```', proof, re.DOTALL)
            if match:
                proof = match.group(1)
                
        # Remove lines that don't look like Lean
        lines = proof.split('\n')
        clean_lines = []
        
        for line in lines:
            # Keep empty lines, comments, and Lean code
            if (line.strip() == '' or 
                line.strip().startswith('--') or
                line.strip().startswith('by') or
                line.strip().startswith('·') or
                line.strip().startswith('exact') or
                line.strip().startswith('apply') or
                line.strip().startswith('rw') or
                line.strip().startswith('simp') or
                line.strip().startswith('intro') or
                line.strip().startswith('have') or
                line.strip().startswith('use') or
                line.strip().startswith('constructor') or
                line.strip().startswith('calc') or
                line.strip().startswith('norm_num') or
                line.strip().startswith('field_simp') or
                line.strip().startswith('ring') or
                any(line.strip().startswith(kw) for kw in ['theorem', 'lemma', 'def'])):
                clean_lines.append(line)
                
        return '\n'.join(clean_lines)
        
    def extract_first_error(self, error_msg: str):
        """Extract the first meaningful error from compiler output"""
        
        # Look for error patterns
        error_patterns = [
            r'error: (.*?)(?:\n|$)',
            r'failed to synthesize instance(.*?)(?:\n|$)',
            r'type mismatch(.*?)(?:\n|$)',
            r'unknown identifier \'(.*?)\'',
            r'invalid field \'(.*?)\'',
            r'application type mismatch(.*?)(?:\n|$)',
            r'failed to prove(.*?)(?:\n|$)',
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, error_msg, re.IGNORECASE)
            if match:
                return match.group(0)[:300]  # More context for complex errors
                
        # Fallback: return first line with 'error'
        for line in error_msg.split('\n'):
            if 'error' in line.lower():
                return line[:300]
                
        return error_msg[:300]
        
    def solve_sorry_with_reinforcement(self, file_path: Path, sorry_info):
        """Solve with iterative reinforcement learning"""
        
        # Check cache first
        cached_proof = self.cache.get_proof(sorry_info['declaration'])
        if cached_proof:
            print(f"  Found cached proof!")
            self.stats['cache_hits'] += 1
            self.apply_proof(file_path, sorry_info['line'], cached_proof)
            self.stats['resolved_sorries'] += 1
            return cached_proof
        
        # Extract minimal context
        if self.enable_minimal_context:
            context = self.extract_minimal_context(file_path, sorry_info['line'])
        else:
            # Fallback to full context
            ctx = self.extractor.extract_context(file_path, sorry_info['line'])
            context = {
                'imports': [],
                'dependencies': [],
                'target_theorem': sorry_info['declaration'],
                'local_definitions': [self.extractor.format_context_for_prompt(ctx)],
                'rs_constants': []
            }
            
        best_proof = None
        best_error = None
        
        for iteration in range(self.max_iterations):
            print(f"  Iteration {iteration + 1}/{self.max_iterations}...")
            
            # Generate proof with chain of thought
            proof = self.generate_proof_with_cot(
                sorry_info, context, iteration, best_error
            )
            self.stats['llm_calls'] += 1
            
            if not proof:
                continue
                
            print(f"  Generated proof: {proof[:80]}...")
            
            # Check compilation
            success, error = self.compiler.check_proof(
                file_path, sorry_info['line'], proof
            )
            
            if success:
                print(f"  ✓ Proof compiles!")
                self.cache.store_proof(sorry_info['declaration'], proof, True)
                self.stats['compile_successes'] += 1
                self.stats['resolved_sorries'] += 1
                self.stats['iterations_per_sorry'].append(iteration + 1)
                
                # Apply the proof
                self.apply_proof(file_path, sorry_info['line'], proof)
                return proof
            else:
                # Extract first error for next iteration
                best_error = self.extract_first_error(error)
                best_proof = proof
                print(f"  ✗ Error: {best_error}")
                self.stats['compile_failures'] += 1
                
                # Cache failure pattern
                self.cache_failure_pattern(sorry_info['name'], best_error)
                
        # All iterations failed
        self.stats['iterations_per_sorry'].append(self.max_iterations)
        return None
        
    def cache_failure_pattern(self, theorem_name: str, error: str):
        """Track common failure patterns"""
        
        # Extract key from error (e.g., missing lemma name)
        missing_match = re.search(r'unknown identifier \'(\w+)\'', error)
        if missing_match:
            missing_id = missing_match.group(1)
            if missing_id not in self.failure_patterns:
                self.failure_patterns[missing_id] = []
            self.failure_patterns[missing_id].append(theorem_name)
            
    def build_dependency_graph(self, file_paths):
        """Build dependency graph for optimal solving order"""
        
        all_sorries = {}
        
        for file_path in file_paths:
            if not file_path.exists():
                continue
                
            sorries = self.find_sorries(file_path)
            for sorry in sorries:
                all_sorries[sorry['name']] = sorry
                self.stats['total_sorries'] += 1
                
        # Analyze dependencies
        for name, sorry_info in all_sorries.items():
            deps = re.findall(r'\b([A-Z]\w+|[a-z]+_\w+)\b', sorry_info['declaration'])
            self.dependency_graph[name] = [d for d in deps if d in all_sorries]
            
        # Return topological order (leaves first)
        visited = set()
        order = []
        
        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for dep in self.dependency_graph.get(name, []):
                visit(dep)
            order.append(name)
            
        for name in all_sorries:
            visit(name)
            
        return [(all_sorries[name]['file'], all_sorries[name]) for name in order]
        
    def apply_proof(self, file_path: Path, line_num: int, proof: str):
        """Apply a proof to the file"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Replace the sorry
        if 'sorry' in lines[line_num - 1]:
            lines[line_num - 1] = lines[line_num - 1].replace('sorry', proof)
            
        with open(file_path, 'w') as f:
            f.writelines(lines)
            
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
                    if any(kw in lines[j] for kw in ['theorem', 'lemma', 'def', 'instance']):
                        # Found start of declaration
                        while j < i:
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
        
    def solve_files(self, file_paths):
        """Solve sorries in optimal order"""
        
        print("=== RECOGNITION SCIENCE GRAVITY O3 SOLVER ===")
        print("Features enabled:")
        print("- Iterative reinforcement learning")
        print("- Minimal context extraction")
        print("- Chain-of-thought reasoning")
        print("- Dependency-ordered solving")
        print("- Failure pattern caching")
        print("- RS-specific golden examples")
        print("-" * 60)
        
        # Build dependency graph and get optimal order
        print("\nAnalyzing dependencies...")
        sorry_queue = self.build_dependency_graph(file_paths)
        print(f"Found {len(sorry_queue)} total sorries")
        
        # Process in order
        for file_path, sorry_info in sorry_queue:
            print(f"\n--- Solving: {sorry_info['name']} ---")
            
            if self.enable_reinforcement:
                self.solve_sorry_with_reinforcement(Path(file_path), sorry_info)
            else:
                # Fallback to single attempt
                self.solve_sorry(Path(file_path), sorry_info)
                
        # Report statistics
        self.report_statistics()
        
    def report_statistics(self):
        """Report detailed statistics"""
        
        print(f"\n{'='*60}")
        print("FINAL STATISTICS")
        print('='*60)
        print(f"Total sorries found: {self.stats['total_sorries']}")
        print(f"Sorries resolved: {self.stats['resolved_sorries']}")
        print(f"Success rate: {self.stats['resolved_sorries'] / max(1, self.stats['total_sorries']):.1%}")
        print(f"Cache hits: {self.stats['cache_hits']}")
        print(f"LLM calls: {self.stats['llm_calls']}")
        
        if self.stats['iterations_per_sorry']:
            avg_iterations = sum(self.stats['iterations_per_sorry']) / len(self.stats['iterations_per_sorry'])
            print(f"Average iterations per sorry: {avg_iterations:.1f}")
            
        if self.failure_patterns:
            print("\nCommon missing identifiers:")
            for missing_id, theorems in sorted(self.failure_patterns.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
                print(f"  {missing_id}: needed by {len(theorems)} theorems")
                
def main():
    # Use the OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    solver = GravityRSO3Solver(api_key)
    
    # Target files - prioritize gravity modules
    target_files = [
        # Core gravity files first
        Path("recognition-ledger/formal/Gravity/FieldEq.lean"),
        Path("recognition-ledger/formal/Gravity/InfoStrain.lean"),
        Path("recognition-ledger/formal/Gravity/Pressure.lean"),
        Path("recognition-ledger/formal/Gravity/XiScreening.lean"),
        Path("recognition-ledger/formal/Gravity/MasterTheorem.lean"),
        Path("recognition-ledger/formal/Gravity/ExperimentalPredictions.lean"),
        
        # Then numerical verification
        Path("recognition-ledger/formal/NumericalVerification 2.lean"),
        Path("recognition-ledger/formal/NumericalTests.lean"),
        
        # Then particle physics
        Path("recognition-ledger/formal/ParticleMassesRevised 2.lean"),
        Path("recognition-ledger/formal/GoldenRatio_CLEAN.lean"),
    ]
    
    # Convert to absolute paths
    base_path = Path.cwd().parent  # Go up from AI Solver directory
    target_files = [base_path / f for f in target_files]
    
    solver.solve_files(target_files)

if __name__ == "__main__":
    main() 