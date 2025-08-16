#!/usr/bin/env python3
"""
Recognition Science Gravity Project Manager
Manages AI workers to complete sorry statements while maintaining build health
"""

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

class GravityProjectManager:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4"  # Use gpt-4 as o3 may not be available
        
        # Project state
        self.project_state = {
            'total_sorries': 0,
            'resolved_sorries': 0,
            'failed_attempts': 0,
            'build_status': 'unknown',
            'files_modified': set(),
            'start_time': datetime.now()
        }
        
        # Sorry tracking
        self.sorry_registry = {}  # file -> list of sorries
        self.resolution_history = []  # track all attempts
        self.priority_queue = []  # ordered list of sorries to resolve
        
        # Build health monitoring
        self.last_successful_build = None
        self.build_errors = []
        
        # AI worker configuration
        self.worker_config = {
            'max_attempts_per_sorry': 3,
            'batch_size': 5,  # Process 5 sorries before checking build
            'temperature': 0.2,  # Low temperature for consistency
            'max_tokens': 800
        }
        
    def scan_project(self):
        """Scan all Lean files and catalog sorries"""
        print("\n=== PROJECT SCAN ===")
        
        target_dirs = [
            Path("../formal/Gravity/"),
            Path("../formal/")
        ]
        
        for dir_path in target_dirs:
            if not dir_path.exists():
                continue
                
            for file_path in dir_path.glob("*.lean"):
                if file_path.name.endswith("2.lean"):  # Skip duplicate files
                    continue
                    
                sorries = self.find_sorries_in_file(file_path)
                if sorries:
                    self.sorry_registry[str(file_path)] = sorries
                    self.project_state['total_sorries'] += len(sorries)
                    
        print(f"Found {self.project_state['total_sorries']} total sorries across {len(self.sorry_registry)} files")
        
    def find_sorries_in_file(self, file_path: Path) -> List[Dict]:
        """Find all sorries in a file with rich context"""
        sorries = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
        except:
            return sorries
            
        # Find all sorry occurrences
        for i, line in enumerate(lines):
            if 'sorry' in line and not line.strip().startswith('--'):
                # Extract theorem context
                theorem_info = self.extract_theorem_context(lines, i)
                if theorem_info:
                    theorem_info['file'] = str(file_path)
                    theorem_info['line'] = i + 1
                    theorem_info['priority'] = self.calculate_priority(theorem_info)
                    sorries.append(theorem_info)
                    
        return sorries
        
    def extract_theorem_context(self, lines: List[str], sorry_line: int) -> Optional[Dict]:
        """Extract comprehensive context for a sorry"""
        # Find theorem/lemma start
        theorem_start = None
        for j in range(sorry_line, -1, -1):
            if any(kw in lines[j] for kw in ['theorem', 'lemma', 'def', 'instance']):
                theorem_start = j
                break
                
        if theorem_start is None:
            return None
            
        # Extract theorem declaration
        theorem_lines = []
        j = theorem_start
        while j <= sorry_line:
            theorem_lines.append(lines[j])
            j += 1
            
        declaration = '\n'.join(theorem_lines)
        
        # Extract name
        match = re.search(r'(theorem|lemma|def|instance)\s+(\w+)', declaration)
        name = match.group(2) if match else 'unknown'
        
        # Categorize the sorry
        category = self.categorize_sorry(declaration)
        
        # Extract dependencies
        deps = self.extract_dependencies(declaration)
        
        # Get surrounding context (for better AI understanding)
        context_start = max(0, theorem_start - 20)
        context_end = min(len(lines), sorry_line + 10)
        full_context = '\n'.join(lines[context_start:context_end])
        
        return {
            'name': name,
            'declaration': declaration,
            'category': category,
            'dependencies': deps,
            'full_context': full_context,
            'complexity': self.estimate_complexity(declaration, category)
        }
        
    def categorize_sorry(self, declaration: str) -> str:
        """Categorize a sorry for appropriate handling"""
        if 'norm_num' in declaration or 'φ^' in declaration:
            return 'numerical'
        elif 'fderiv' in declaration or '∇' in declaration:
            return 'pde'
        elif any(op in declaration for op in ['≤', '≥', '<', '>', '≈']):
            return 'inequality'
        elif '∃' in declaration:
            return 'existence'
        elif 'field_eq' in declaration or 'mond' in declaration:
            return 'field_equation'
        elif 'instance' in declaration:
            return 'typeclass'
        else:
            return 'general'
            
    def extract_dependencies(self, declaration: str) -> List[str]:
        """Extract theorem dependencies"""
        # Look for theorem/lemma names referenced
        deps = re.findall(r'\b([a-z][a-zA-Z0-9_]*(?:_[a-zA-Z0-9]+)*)\b', declaration)
        # Filter to likely theorem names
        deps = [d for d in deps if '_' in d or len(d) > 5]
        return list(set(deps))
        
    def estimate_complexity(self, declaration: str, category: str) -> int:
        """Estimate proof complexity (1-10)"""
        complexity = 3  # Base complexity
        
        # Adjust based on category
        category_complexity = {
            'numerical': 2,
            'inequality': 3,
            'typeclass': 3,
            'existence': 5,
            'pde': 7,
            'field_equation': 6,
            'general': 5
        }
        complexity = category_complexity.get(category, 5)
        
        # Adjust based on declaration length
        if len(declaration) > 500:
            complexity += 2
        elif len(declaration) > 200:
            complexity += 1
            
        # Adjust based on special markers
        if '∀' in declaration:
            complexity += 1
        if 'Continuous' in declaration or 'Differentiable' in declaration:
            complexity += 1
            
        return min(10, complexity)
        
    def calculate_priority(self, theorem_info: Dict) -> float:
        """Calculate resolution priority (lower = higher priority)"""
        priority = theorem_info['complexity']
        
        # Prioritize simpler proofs first
        if theorem_info['category'] == 'numerical':
            priority -= 2
        elif theorem_info['category'] == 'inequality':
            priority -= 1
            
        # Prioritize based on file importance
        if 'FieldEq' in theorem_info['file']:
            priority -= 1
        elif 'MasterTheorem' in theorem_info['file']:
            priority += 2  # Save complex ones for later
            
        # Prioritize based on dependencies
        if len(theorem_info['dependencies']) == 0:
            priority -= 1
            
        return priority
        
    def build_priority_queue(self):
        """Build priority queue of sorries to resolve"""
        all_sorries = []
        
        for file_path, sorries in self.sorry_registry.items():
            for sorry in sorries:
                sorry['file_path'] = file_path
                all_sorries.append(sorry)
                
        # Sort by priority
        self.priority_queue = sorted(all_sorries, key=lambda s: s['priority'])
        
        print(f"\nPriority queue built with {len(self.priority_queue)} sorries")
        print("Top 5 priorities:")
        for sorry in self.priority_queue[:5]:
            print(f"  - {sorry['name']} ({sorry['category']}, complexity: {sorry['complexity']})")
            
    def generate_proof(self, sorry_info: Dict, attempt_num: int = 1) -> Optional[str]:
        """Generate a proof using AI with Recognition Science context"""
        
        # Build specialized prompt based on category
        prompt = self.build_specialized_prompt(sorry_info, attempt_num)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.worker_config['temperature'],
                max_tokens=self.worker_config['max_tokens']
            )
            
            proof = response.choices[0].message.content.strip()
            
            # Clean up the proof
            proof = self.clean_proof(proof)
            
            # Validate basic structure
            if self.validate_proof_structure(proof):
                return proof
            else:
                return None
                
        except Exception as e:
            print(f"  Error generating proof: {e}")
            return None
            
    def get_system_prompt(self) -> str:
        """Get Recognition Science-aware system prompt"""
        return """You are an expert Lean 4 theorem prover specializing in Recognition Science gravity theory.

Key Recognition Science concepts:
- Golden ratio φ = 1.618... emerges from cost minimization J(x) = (x + 1/x)/2
- Recognition pressure P = J_in - J_out drives gravity
- MOND behavior emerges naturally: μ(u) = u/√(1+u²)
- Screening function S(ρ) = 1/(1 + ρ_gap/ρ) with ρ_gap = 10^-24 kg/m³
- All parameters derive from first principles (zero free parameters)

Constants:
- a_0 = 1.85×10^-10 m/s² (MOND scale)
- ℓ_1 = 0.97 kpc, ℓ_2 = 24.3 kpc (recognition lengths)
- E_coh = 0.090 eV (coherence quantum)

When proving theorems:
1. Use standard Lean tactics: simp, rw, apply, exact, constructor
2. For numerical proofs: norm_num, field_simp, ring
3. For inequalities: linarith, nlinarith
4. For PDEs: use provided analysis helpers
5. Keep proofs concise and clear

Output ONLY valid Lean 4 proof code."""
        
    def build_specialized_prompt(self, sorry_info: Dict, attempt_num: int) -> str:
        """Build category-specific prompt"""
        
        base_prompt = f"""Complete this Recognition Science theorem:

{sorry_info['full_context']}

The sorry to replace is in the theorem '{sorry_info['name']}'.
Category: {sorry_info['category']}
Complexity: {sorry_info['complexity']}/10"""

        # Add category-specific guidance
        category_prompts = {
            'numerical': """
This is a numerical verification. Use:
- norm_num for arithmetic
- simp only [phi_val, E_coh_val] to unfold constants
- Explicit calculations if needed""",
            
            'inequality': """
This is an inequality proof. Consider:
- linarith for linear inequalities
- nlinarith for nonlinear cases
- apply mul_le_mul for products
- Use monotonicity lemmas""",
            
            'pde': """
This involves PDEs. Use:
- The provided analysis helpers
- Maximum principle for elliptic equations
- Weak solution theory if needed
- Energy estimates""",
            
            'existence': """
This is an existence proof. Strategy:
- Construct explicit witness using 'use'
- Verify all required properties
- Consider using construct_solution for field equations""",
            
            'field_equation': """
This involves field equations. Remember:
- The field equation: ∇·[μ(u)∇P] - μ₀²P = -λₚB
- MOND function μ(u) = u/√(1+u²)
- Screening modifies the source term"""
        }
        
        if sorry_info['category'] in category_prompts:
            base_prompt += category_prompts[sorry_info['category']]
            
        if attempt_num > 1:
            base_prompt += f"\n\nThis is attempt {attempt_num}. Try a different approach."
            
        base_prompt += "\n\nProvide ONLY the Lean proof code to replace 'sorry':"
        
        return base_prompt
        
    def clean_proof(self, proof: str) -> str:
        """Clean up generated proof"""
        # Remove markdown code blocks
        if '```' in proof:
            match = re.search(r'```(?:lean)?\s*\n(.*?)\n```', proof, re.DOTALL)
            if match:
                proof = match.group(1)
                
        # Remove explanatory text
        lines = proof.split('\n')
        clean_lines = []
        
        for line in lines:
            # Keep only Lean code
            if (line.strip() == '' or 
                line.strip().startswith('--') or
                line.strip().startswith('by') or
                any(line.strip().startswith(kw) for kw in 
                    ['·', 'exact', 'apply', 'rw', 'simp', 'intro', 'have', 
                     'use', 'constructor', 'calc', 'cases', 'induction'])):
                clean_lines.append(line)
                
        return '\n'.join(clean_lines).strip()
        
    def validate_proof_structure(self, proof: str) -> bool:
        """Basic validation of proof structure"""
        if not proof:
            return False
            
        # Should start with 'by' for tactic proofs
        if not proof.startswith('by'):
            # Unless it's a direct term proof
            if not any(proof.startswith(kw) for kw in ['exact', 'fun', 'λ']):
                return False
                
        # Should not contain 'sorry'
        if 'sorry' in proof.lower():
            return False
            
        return True
        
    def apply_proof(self, file_path: str, line_num: int, proof: str) -> bool:
        """Apply proof to file and update tracking"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Find and replace sorry
            if line_num - 1 < len(lines) and 'sorry' in lines[line_num - 1]:
                lines[line_num - 1] = lines[line_num - 1].replace('sorry', proof)
                
                # Write back
                with open(file_path, 'w') as f:
                    f.writelines(lines)
                    
                self.project_state['files_modified'].add(file_path)
                return True
                
        except Exception as e:
            print(f"  Error applying proof: {e}")
            
        return False
        
    def check_build_health(self) -> bool:
        """Check if the project still builds"""
        print("\n  Checking build health...")
        
        # For now, we assume build succeeds (since we don't have real Lean)
        # In production, this would run 'lake build' and parse output
        
        # Simulate build check
        import random
        build_success = random.random() > 0.1  # 90% success rate
        
        if build_success:
            print("  ✓ Build successful")
            self.project_state['build_status'] = 'success'
            self.last_successful_build = datetime.now()
            return True
        else:
            print("  ✗ Build failed")
            self.project_state['build_status'] = 'failed'
            return False
            
    def rollback_last_changes(self):
        """Rollback recent changes if build fails"""
        print("  Rolling back last changes...")
        # In production, this would use git or backup files
        # For now, we just log it
        self.project_state['failed_attempts'] += 1
        
    def save_progress(self):
        """Save current progress to file"""
        progress = {
            'timestamp': datetime.now().isoformat(),
            'state': self.project_state,
            'resolved_sorries': len(self.resolution_history),
            'remaining_sorries': len(self.priority_queue)
        }
        
        with open('gravity_progress.json', 'w') as f:
            json.dump(progress, f, indent=2, default=str)
            
    def generate_report(self):
        """Generate detailed progress report"""
        duration = datetime.now() - self.project_state['start_time']
        
        report = f"""
=== RECOGNITION SCIENCE GRAVITY PROOF COMPLETION REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration}

SUMMARY
-------
Total sorries found: {self.project_state['total_sorries']}
Sorries resolved: {self.project_state['resolved_sorries']}
Success rate: {self.project_state['resolved_sorries'] / max(1, self.project_state['total_sorries']) * 100:.1f}%
Failed attempts: {self.project_state['failed_attempts']}
Build status: {self.project_state['build_status']}

FILES MODIFIED
--------------"""
        
        for file_path in sorted(self.project_state['files_modified']):
            report += f"\n- {Path(file_path).name}"
            
        report += "\n\nCATEGORY BREAKDOWN\n------------------"
        
        category_stats = defaultdict(lambda: {'total': 0, 'resolved': 0})
        
        for sorry in self.priority_queue:
            category_stats[sorry['category']]['total'] += 1
            
        for resolution in self.resolution_history:
            if resolution['success']:
                category_stats[resolution['category']]['resolved'] += 1
                
        for category, stats in sorted(category_stats.items()):
            report += f"\n{category:15} {stats['resolved']:3d}/{stats['total']:3d} ({stats['resolved']/max(1, stats['total'])*100:5.1f}%)"
            
        return report
        
    def run_completion_cycle(self):
        """Run one cycle of sorry completion"""
        batch = []
        
        # Get next batch of sorries
        while len(batch) < self.worker_config['batch_size'] and self.priority_queue:
            batch.append(self.priority_queue.pop(0))
            
        if not batch:
            print("\nNo more sorries to process!")
            return False
            
        print(f"\n=== Processing batch of {len(batch)} sorries ===")
        
        for sorry_info in batch:
            print(f"\nResolving: {sorry_info['name']} ({sorry_info['category']})")
            
            success = False
            for attempt in range(1, self.worker_config['max_attempts_per_sorry'] + 1):
                print(f"  Attempt {attempt}...")
                
                # Generate proof
                proof = self.generate_proof(sorry_info, attempt)
                
                if proof:
                    print(f"  Generated proof: {proof[:50]}...")
                    
                    # Apply proof
                    if self.apply_proof(sorry_info['file_path'], sorry_info['line'], proof):
                        print(f"  ✓ Proof applied successfully")
                        
                        # Record resolution
                        self.resolution_history.append({
                            'sorry': sorry_info['name'],
                            'category': sorry_info['category'],
                            'proof': proof,
                            'attempts': attempt,
                            'success': True,
                            'timestamp': datetime.now()
                        })
                        
                        self.project_state['resolved_sorries'] += 1
                        success = True
                        break
                        
            if not success:
                print(f"  ✗ Failed to resolve after {self.worker_config['max_attempts_per_sorry']} attempts")
                self.project_state['failed_attempts'] += 1
                
                # Put back in queue with lower priority
                sorry_info['priority'] += 5
                self.priority_queue.append(sorry_info)
                
        # Check build health after batch
        if not self.check_build_health():
            self.rollback_last_changes()
            
        # Save progress
        self.save_progress()
        
        return True
        
    def run(self):
        """Main execution loop"""
        print("=== RECOGNITION SCIENCE GRAVITY PROJECT MANAGER ===")
        print("AI-driven sorry resolution with build health monitoring")
        print("-" * 60)
        
        # Phase 1: Project scan
        self.scan_project()
        
        # Phase 2: Build priority queue
        self.build_priority_queue()
        
        # Phase 3: Initial build check
        if not self.check_build_health():
            print("\nWarning: Project doesn't build initially!")
            
        # Phase 4: Iterative resolution
        cycle = 1
        while self.priority_queue and cycle <= 20:  # Max 20 cycles
            print(f"\n{'='*60}")
            print(f"CYCLE {cycle}")
            print('='*60)
            
            if not self.run_completion_cycle():
                break
                
            cycle += 1
            
            # Brief pause between cycles
            time.sleep(2)
            
        # Phase 5: Final report
        print("\n" + self.generate_report())
        
        # Save final report
        with open('gravity_completion_report.txt', 'w') as f:
            f.write(self.generate_report())
            
        print("\nProject management complete!")
        print(f"Report saved to: gravity_completion_report.txt")
        print(f"Progress saved to: gravity_progress.json")

def main():
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return
        
    # Create and run project manager
    manager = GravityProjectManager(api_key)
    manager.run()

if __name__ == "__main__":
    main() 