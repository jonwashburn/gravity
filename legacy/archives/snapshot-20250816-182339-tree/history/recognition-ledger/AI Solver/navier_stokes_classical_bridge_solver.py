#!/usr/bin/env python3
"""
Navier-Stokes Classical Bridge Solver
====================================

Uses the Recognition Science classical bridge lemmas to solve Navier-Stokes sorries.
Based on the advanced O3 solver but with updated prompting strategy.
"""

import os
import re
import json
from pathlib import Path
from openai import OpenAI
from compile_checker import CompileChecker
from context_extractor import ContextExtractor
from proof_cache import ProofCache

class ClassicalBridgeSolver:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "o3"
        
        # Initialize components
        self.cache = ProofCache()
        self.compiler = CompileChecker()
        self.extractor = ContextExtractor()
        
        # Advanced settings
        self.max_iterations = 3  # Per sorry
        self.max_completion_tokens = 600
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
        
        # Classical bridge lemmas
        self.bridge_lemmas = self.load_bridge_lemmas()
        
        # Golden examples using bridge lemmas
        self.golden_examples = self.load_golden_examples()
        
        # Failure pattern cache
        self.failure_patterns = {}
        
        # Dependency graph
        self.dependency_graph = {}
        
    def load_bridge_lemmas(self):
        """Load the classical bridge lemmas we can reference"""
        return {
            'vorticity_cascade_bound': {
                'statement': 'theorem vorticity_cascade_bound (ω_max : ℝ → ℝ) (h_smooth : ∀ t, 0 ≤ ω_max t) : ∃ C₀ > 0, ∀ t ≥ 0, ω_max t ≤ C₀ * (1 + t / recognition_tick) * exp (cascade_cutoff * t / recognition_tick)',
                'usage': 'Controls vorticity growth with φ⁻⁴ cascade cutoff',
                'constants': ['cascade_cutoff = phi^(-4) ≈ 0.1459', 'recognition_tick = 7.33e-15']
            },
            'energy_dissipation_bound': {
                'statement': 'theorem energy_dissipation_bound (E : ℝ → ℝ) (ν : ℝ) (hν : ν > 0) (E_initial : ℝ) (hE : E 0 = E_initial) : ∃ K > 0, ∀ t ≥ 0, E t ≤ E_initial * exp (-K * phi^2 * ν * t)',
                'usage': 'Energy decays exponentially with φ² factor',
                'constants': ['phi^2 ≈ 2.618']
            },
            'modified_gronwall': {
                'statement': 'theorem modified_gronwall (f : ℝ → ℝ) (hf : Continuous f) (h_bound : ∀ t ≥ 0, f t ≤ f 0 + (log phi / recognition_tick) * t * f 0) : ∀ t ≥ 0, f t ≤ f 0 * exp ((log phi / recognition_tick) * t)',
                'usage': 'Grönwall inequality with φ-dependent growth rate',
                'constants': ['log phi ≈ 0.4812']
            },
            'critical_time_scale': {
                'statement': 'theorem critical_time_scale (ω_max : ℝ → ℝ) (h_vort : ∀ t, 0 ≤ ω_max t) : ∀ t ≤ recognition_tick, ω_max t ≤ ω_max 0 * (1 + phi * t / recognition_tick)',
                'usage': 'Short-time bound using recognition tick',
                'constants': ['recognition_tick = 7.33e-15']
            },
            'enstrophy_production_bound': {
                'statement': 'theorem enstrophy_production_bound (Z : ℝ → ℝ) (hZ : ∀ t, 0 ≤ Z t) : ∃ M > 0, ∀ t ≥ 0, Z t ≤ Z 0 * exp (M * cascade_cutoff * t)',
                'usage': 'Enstrophy growth limited by cascade cutoff',
                'constants': ['cascade_cutoff ≈ 0.1459']
            }
        }
        
    def load_golden_examples(self):
        """Load examples that use classical bridge lemmas"""
        examples = []
        
        # Example 1: Using vorticity cascade bound
        examples.append({
            'declaration': 'theorem vorticity_stays_bounded (ω : ℝ → ℝ → ℝ) (h_smooth : ∀ t, Smooth (ω t)) : ∃ C > 0, ∀ t ≥ 0, ‖ω t‖_∞ < C',
            'proof': '''by
  -- Use the vorticity cascade bound from RSClassicalBridge
  let ω_max t := ⨆ x, |ω t x|
  have h_nonneg : ∀ t, 0 ≤ ω_max t := by
    intro t
    simp [ω_max]
    apply le_ciSup_of_le
    exact abs_nonneg _
  obtain ⟨C₀, hC₀, h_bound⟩ := RSClassical.vorticity_cascade_bound ω_max h_nonneg
  use C₀ * 2  -- Factor of 2 for safety
  intro t ht
  rw [norm_eq_sup]
  exact lt_of_le_of_lt (h_bound t ht) (by linarith)'''
        })
        
        # Example 2: Using energy dissipation
        examples.append({
            'declaration': 'theorem energy_decays (E : ℝ → ℝ) (ν : ℝ) (hν : ν > 0) : E 0 < ∞ → ∀ t > 0, E t < E 0',
            'proof': '''by
  intro h_finite t ht
  -- Apply energy dissipation bound with φ² factor
  obtain ⟨K, hK, h_decay⟩ := RSClassical.energy_dissipation_bound E ν hν (E 0) rfl
  calc E t ≤ E 0 * exp (-K * phi^2 * ν * t) := h_decay t (le_of_lt ht)
       _ < E 0 * 1 := by
         apply mul_lt_mul_of_pos_left
         · exact exp_lt_one_of_neg (mul_neg_of_neg_of_pos (by linarith) (mul_pos (mul_pos hK (sq_pos_of_pos phi_pos)) (mul_pos hν ht)))
         · exact h_finite
       _ = E 0 := mul_one _'''
        })
        
        # Example 3: Using critical time scale
        examples.append({
            'declaration': 'theorem short_time_regularity (ω : ℝ → ℝ → ℝ) : ∀ t ≤ recognition_tick, ‖ω t‖_∞ ≤ ‖ω 0‖_∞ * 2',
            'proof': '''by
  intro t ht
  let ω_max s := ‖ω s‖_∞
  have h_nonneg : ∀ s, 0 ≤ ω_max s := norm_nonneg
  have h_bound := RSClassical.critical_time_scale ω_max h_nonneg t ht
  calc ‖ω t‖_∞ = ω_max t := rfl
       _ ≤ ω_max 0 * (1 + phi * t / recognition_tick) := h_bound
       _ ≤ ω_max 0 * (1 + phi) := by
         apply mul_le_mul_of_nonneg_left
         · apply add_le_add_left
           exact div_le_one_of_le (mul_le_of_le_one_left (le_of_lt phi_pos) ht) recognition_tick_pos
         · exact h_nonneg 0
       _ ≤ ω_max 0 * 2 := by
         apply mul_le_mul_of_nonneg_left
         · linarith [phi_lt_two]
         · exact h_nonneg 0
       _ = ‖ω 0‖_∞ * 2 := rfl'''
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
            'local_definitions': []
        }
        
        # Get imports
        for line in lines[:30]:  # Check more lines for imports
            if line.strip().startswith('import'):
                context['imports'].append(line.strip())
                
        # Always include our bridge import
        if 'import NavierStokesLedger.RSClassicalBridge' not in context['imports']:
            context['imports'].append('import NavierStokesLedger.RSClassicalBridge')
            
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
                
        # Get key definitions
        key_terms = ['VectorField', 'Energy', 'Vorticity', 'L2Norm', 'NavierStokes']
        for term in key_terms:
            for i, line in enumerate(lines):
                if f'def {term}' in line:
                    # Extract this definition
                    def_lines = [line]
                    j = i + 1
                    indent = len(line) - len(line.lstrip())
                    while j < len(lines) and (lines[j].strip() == '' or len(lines[j]) - len(lines[j].lstrip()) > indent):
                        def_lines.append(lines[j])
                        j += 1
                        if j - i > 10:  # Limit definition length
                            break
                    context['local_definitions'].append(''.join(def_lines))
                    break
                    
        return context
        
    def generate_proof_with_bridge_lemmas(self, sorry_info, context, iteration=0, previous_error=None):
        """Generate proof using classical bridge lemmas"""
        
        # Analyze what type of proof this might need
        theorem_text = context['target_theorem'].lower()
        suggested_lemmas = []
        
        if 'vorticity' in theorem_text or 'vort' in theorem_text:
            suggested_lemmas.append('vorticity_cascade_bound')
            if 'short' in theorem_text or 'small' in theorem_text:
                suggested_lemmas.append('critical_time_scale')
                
        if 'energy' in theorem_text:
            suggested_lemmas.append('energy_dissipation_bound')
            
        if 'gronwall' in theorem_text or 'growth' in theorem_text:
            suggested_lemmas.append('modified_gronwall')
            
        if 'enstrophy' in theorem_text:
            suggested_lemmas.append('enstrophy_production_bound')
            
        # Build the prompt
        prompt = f"""You are an expert Lean 4 theorem prover. You have access to proven lemmas from Recognition Science that provide exact constants for Navier-Stokes estimates.

## AVAILABLE BRIDGE LEMMAS

You can use these lemmas from RSClassical namespace:

"""
        
        # Add relevant bridge lemmas
        for lemma_name in suggested_lemmas[:3]:  # Limit to top 3
            if lemma_name in self.bridge_lemmas:
                lemma = self.bridge_lemmas[lemma_name]
                prompt += f"""### {lemma_name}
{lemma['statement']}
-- {lemma['usage']}
-- Constants: {', '.join(lemma['constants'])}

"""

        prompt += f"""## CONTEXT

### Imports:
{chr(10).join(context['imports'][:10])}

### Key Definitions:
{chr(10).join(context['local_definitions'][:3])}

### Golden Examples Using Bridge Lemmas:
"""
        
        for ex in self.golden_examples[:2]:  # Show 2 examples
            prompt += f"\nExample:\n{ex['declaration']}\n{ex['proof']}\n"
            
        prompt += f"""
### Target Theorem:
{context['target_theorem']}

"""

        if previous_error:
            prompt += f"""### Previous Attempt Error:
{previous_error}

Fix this specific error. Common solutions:
- If "unknown identifier", check imports or use fully qualified names
- If "type mismatch", check exact types needed
- If "failed to synthesize", provide missing typeclass instances
"""

        prompt += """## YOUR TASK

Write a Lean 4 proof using the bridge lemmas when applicable.

Key constants to use:
- phi ≈ 1.618 (golden ratio)
- cascade_cutoff = phi^(-4) ≈ 0.1459
- recognition_tick = 7.33e-15
- log phi ≈ 0.4812

IMPORTANT:
- Prefer using the proven bridge lemmas over proving from scratch
- Use RSClassical.lemma_name to access bridge lemmas
- Keep proofs concise (under 15 lines)
- Output ONLY valid Lean 4 code
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
                line.strip().startswith('/-') or
                line.strip().startswith('by') or
                line.strip().startswith('·') or
                line.strip().startswith('exact') or
                line.strip().startswith('apply') or
                line.strip().startswith('rw') or
                line.strip().startswith('simp') or
                line.strip().startswith('intro') or
                line.strip().startswith('have') or
                line.strip().startswith('obtain') or
                line.strip().startswith('use') or
                line.strip().startswith('calc') or
                line.strip().startswith('constructor') or
                line.strip().startswith('let') or
                '_' in line or  # Keep lines with underscores (often part of calc)
                any(line.strip().startswith(kw) for kw in ['theorem', 'lemma', 'def'])):
                clean_lines.append(line)
                
        return '\n'.join(clean_lines)
        
    def solve_sorry_with_reinforcement(self, file_path: Path, sorry_info):
        """Solve with iterative reinforcement learning"""
        
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
                'local_definitions': [self.extractor.format_context_for_prompt(ctx)]
            }
            
        best_proof = None
        best_error = None
        
        for iteration in range(self.max_iterations):
            print(f"  Iteration {iteration + 1}/{self.max_iterations}...")
            
            # Generate proof with bridge lemmas
            proof = self.generate_proof_with_bridge_lemmas(
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
        
    def extract_first_error(self, error_msg: str):
        """Extract the first meaningful error from compiler output"""
        
        # Look for error patterns
        error_patterns = [
            r'error: (.*?)(?:\n|$)',
            r'failed to synthesize instance(.*?)(?:\n|$)',
            r'type mismatch(.*?)(?:\n|$)',
            r'unknown identifier \'(.*?)\'',
            r'invalid field \'(.*?)\'',
            r'invalid occurrence of universe level \'(.*?)\'',
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, error_msg, re.IGNORECASE)
            if match:
                return match.group(0)[:200]  # First 200 chars
                
        # Fallback: return first line with 'error'
        for line in error_msg.split('\n'):
            if 'error' in line.lower():
                return line[:200]
                
        return error_msg[:200]
        
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
        
        print("=== NAVIER-STOKES CLASSICAL BRIDGE SOLVER ===")
        print("Using Recognition Science classical lemmas:")
        for lemma_name, lemma in self.bridge_lemmas.items():
            print(f"- {lemma_name}: {lemma['usage']}")
        print("\nFeatures enabled:")
        print("- Iterative reinforcement learning")
        print("- Minimal context extraction")
        print("- Bridge lemma suggestions")
        print("- Dependency-ordered solving")
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
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return
    
    solver = ClassicalBridgeSolver(api_key)
    
    # Target files - prioritize files that likely need our bridge lemmas
    target_files = [
        Path("../NavierStokesLedger/VorticityBound.lean"),
        Path("../NavierStokesLedger/EnergyEstimates.lean"),
        Path("../NavierStokesLedger/BealeKatoMajda.lean"),
        Path("../NavierStokesLedger/GlobalRegularity.lean"),
        Path("../NavierStokesLedger/TimeDependent.lean"),
    ]
    
    solver.solve_files(target_files)

if __name__ == "__main__":
    main() 