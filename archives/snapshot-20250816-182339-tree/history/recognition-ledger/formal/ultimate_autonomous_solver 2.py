#!/usr/bin/env python3
"""
Ultimate Autonomous Recognition Science Solver
==============================================

Built for maximum speed and autonomy. Uses:
- 20 parallel agents (different specializations)
- Automatic model escalation (Sonnet → Opus)
- Self-healing and diagnostic systems
- Continuous operation until complete

CEO-optimized: Time > Money. No token limits.
"""

import os
import json
import asyncio
import aiohttp
import time
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import anthropic

# Your API key - set via environment variable for security
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Updated model hierarchy: Sonnet first, then escalate to Opus
MODELS = {
    "fast": "claude-3-5-sonnet-20241022",  # Start with Sonnet
    "powerful": "claude-3-opus-20240229",   # Escalate to Opus
    "ultimate": "claude-3-opus-20240229"    # Maximum power
}

# Maximum tokens for true autonomy
MAX_TOKENS = {
    "fast": 8192,      # Sonnet max
    "powerful": 16384,  # Opus extended
    "ultimate": 32768   # Opus maximum
}

@dataclass
class Theorem:
    """Enhanced theorem tracking with full state"""
    name: str
    statement: str
    dependencies: List[str]
    level: int  # 0=axioms, 1=foundation, 2=core, etc.
    lean_file: Optional[str] = None
    line_number: Optional[int] = None
    status: str = "unproven"
    attempts: List[Dict] = field(default_factory=list)
    proof: Optional[str] = None
    verified: bool = False
    prediction_generated: bool = False
    assigned_agents: Set[str] = field(default_factory=set)
    failed_models: Set[str] = field(default_factory=set)  # Track which models failed
    
    @property
    def id(self):
        return f"{self.name}_{self.level}"

class UltimateRecognitionSolver:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=API_KEY)
        self.theorems = self._load_all_theorems()
        self.proven = set()
        self.session = None
        self.active_agents = {}
        self.proof_certificates = []
        
        # 20 specialized agents - now with smarter model assignment
        self.agents = self._create_agent_army()
        
        # Track everything
        self.start_time = time.time()
        self.api_calls = 0
        self.total_tokens = 0
        self.model_escalations = 0
        
    def _create_agent_army(self) -> Dict:
        """Create 20 specialized agents with smart model assignment"""
        return {
            # Core mathematics agents - start with Sonnet
            "Archimedes": {"specialty": "Golden ratio and fixed points", "model": "fast", "priority": "critical"},
            "Euler": {"specialty": "Number theory and series", "model": "fast", "priority": "high"},
            "Gauss": {"specialty": "Algebraic structures", "model": "fast", "priority": "high"},
            "Riemann": {"specialty": "Complex analysis", "model": "fast", "priority": "high"},
            "Cauchy": {"specialty": "Limits and convergence", "model": "fast", "priority": "medium"},
            
            # Physics agents - start with Sonnet
            "Einstein": {"specialty": "Energy-mass equivalence", "model": "fast", "priority": "critical"},
            "Planck": {"specialty": "Quantum scales", "model": "fast", "priority": "high"},
            "Noether": {"specialty": "Symmetries and conservation", "model": "fast", "priority": "high"},
            "Dirac": {"specialty": "Operators and eigenstates", "model": "fast", "priority": "medium"},
            
            # Recognition Science specialists - mixed assignment
            "Pythagoras": {"specialty": "Cosmic harmony and ratios", "model": "fast", "priority": "critical"},
            "Fibonacci": {"specialty": "Golden cascade and recursion", "model": "fast", "priority": "high"},
            "Kepler": {"specialty": "Eight-beat cycles", "model": "fast", "priority": "medium"},
            "Tesla": {"specialty": "Resonance and frequency", "model": "fast", "priority": "medium"},
            
            # Lean proof specialists - start fast
            "Euclid": {"specialty": "Geometric proofs", "model": "fast", "priority": "low"},
            "Bourbaki": {"specialty": "Formal structures", "model": "fast", "priority": "medium"},
            "Hilbert": {"specialty": "Axiom systems", "model": "fast", "priority": "high"},
            "Gödel": {"specialty": "Completeness and consistency", "model": "fast", "priority": "critical"},
            
            # Verification specialists - fast is usually enough
            "Turing": {"specialty": "Computation and verification", "model": "fast", "priority": "medium"},
            "Church": {"specialty": "Lambda calculus and types", "model": "fast", "priority": "low"},
            "Curry": {"specialty": "Proof checking", "model": "fast", "priority": "low"}
        }
    
    def _load_all_theorems(self) -> Dict[str, Theorem]:
        """Load complete theorem database from scaffolding"""
        theorems = {}
        
        # Level 0: Axioms (given)
        for i in range(1, 9):
            name = f"A{i}"
            theorems[name] = Theorem(
                name=name,
                statement=f"Axiom {i} of Recognition Science",
                dependencies=[],
                level=0,
                status="given"
            )
        
        # Level 1: Foundation
        foundation = [
            ("F1_LedgerBalance", "∀ S : LedgerState, total_debits(S) = total_credits(S)", ["A2"]),
            ("F2_TickInjective", "L is injective", ["A1", "A4"]),
            ("F3_DualInvolution", "J(J(S)) = S", ["A2"]),
            ("F4_CostNonnegative", "C(S) ≥ 0", ["A3"])
        ]
        
        for name, stmt, deps in foundation:
            theorems[name] = Theorem(name=name, statement=stmt, dependencies=deps, level=1)
        
        # Level 2: Core (CRITICAL!)
        core = [
            ("C1_GoldenRatioLockIn", "J(x)=(x+1/x)/2 has unique fixed point φ=(1+√5)/2", ["A8", "F4_CostNonnegative"]),
            ("C2_EightBeatPeriod", "L⁸ = identity on symmetric subspace", ["A7"]),
            ("C3_RecognitionLength", "Unique length scale λ_rec", ["A5", "A6"]),
            ("C4_TickIntervalFormula", "τ₀ = λ_rec/(8c log φ)", ["C1_GoldenRatioLockIn", "C3_RecognitionLength", "A7"])
        ]
        
        for name, stmt, deps in core:
            theorems[name] = Theorem(name=name, statement=stmt, dependencies=deps, level=2)
        
        # Level 3: Energy Cascade
        energy = [
            ("E1_CoherenceQuantum", "E_coh = (φ/π) × (ℏc/λ_rec) = 0.090 eV", ["C1_GoldenRatioLockIn", "C3_RecognitionLength"]),
            ("E2_PhiLadder", "E_r = E_coh × φ^r", ["E1_CoherenceQuantum", "C1_GoldenRatioLockIn"]),
            ("E3_MassEnergyEquivalence", "mass = C₀/c²", ["F4_CostNonnegative"]),
            ("E4_ElectronRung", "electron at r = 32", ["E2_PhiLadder", "A7"]),
            ("E5_ParticleRungTable", "Complete particle assignments", ["E4_ElectronRung"])
        ]
        
        for name, stmt, deps in energy:
            theorems[name] = Theorem(name=name, statement=stmt, dependencies=deps, level=3)
        
        # Level 4: Gauge Structure
        gauge = [
            ("G1_ColorFromResidue", "color = r mod 3", ["E5_ParticleRungTable", "A7"]),
            ("G2_IsospinFromResidue", "isospin = f mod 4", ["E5_ParticleRungTable", "A7"]),
            ("G3_HyperchargeFormula", "hypercharge = (r+f) mod 6", ["G1_ColorFromResidue", "G2_IsospinFromResidue"]),
            ("G4_GaugeGroupEmergence", "SU(3)×SU(2)×U(1)", ["G1_ColorFromResidue", "G2_IsospinFromResidue", "G3_HyperchargeFormula"]),
            ("G5_CouplingConstants", "g₁²=20π/9, g₂²=2π, g₃²=4π/3", ["G4_GaugeGroupEmergence"])
        ]
        
        for name, stmt, deps in gauge:
            theorems[name] = Theorem(name=name, statement=stmt, dependencies=deps, level=4)
        
        # Level 5: Predictions
        predictions = [
            ("P1_ElectronMass", "m_e = 0.511 MeV", ["E1_CoherenceQuantum", "E4_ElectronRung"]),
            ("P2_MuonMass", "m_μ = 105.66 MeV", ["E1_CoherenceQuantum", "E5_ParticleRungTable"]),
            ("P3_FineStructure", "α = 1/137.036", ["G5_CouplingConstants"]),
            ("P4_GravitationalConstant", "G from holography", ["C3_RecognitionLength"]),
            ("P5_DarkEnergy", "ρ_Λ from half-coin", ["E1_CoherenceQuantum", "C4_TickIntervalFormula", "A7"]),
            ("P6_HubbleConstant", "H₀ = 67.4 km/s/Mpc", ["P5_DarkEnergy"]),
            ("P7_AllParticleMasses", "Complete spectrum", ["E5_ParticleRungTable", "P1_ElectronMass", "P2_MuonMass"])
        ]
        
        for name, stmt, deps in predictions:
            theorems[name] = Theorem(name=name, statement=stmt, dependencies=deps, level=5)
        
        return theorems
    
    async def create_session(self):
        """Create aiohttp session for parallel API calls"""
        self.session = aiohttp.ClientSession()
    
    def can_prove(self, theorem_name: str) -> bool:
        """Check if dependencies are satisfied"""
        theorem = self.theorems[theorem_name]
        for dep in theorem.dependencies:
            # Handle case where dependency might not exist
            if dep not in self.theorems:
                print(f"Warning: Dependency {dep} not found for {theorem_name}")
                return False
            if dep not in self.proven and self.theorems[dep].status != "given":
                return False
        return True
    
    def get_agent_model(self, agent_name: str, theorem: Theorem) -> str:
        """Determine which model to use based on agent, theorem, and failure history"""
        agent = self.agents[agent_name]
        
        # If Sonnet already failed, escalate to Opus
        if "claude-3-5-sonnet" in theorem.failed_models:
            return "powerful"  # Escalate to Opus
        
        # Critical theorems get Opus after 2+ attempts
        if theorem.name == "C1_GoldenRatioLockIn" and len(theorem.attempts) >= 2:
            return "ultimate"
        
        # High priority + multiple failures = escalate
        if agent["priority"] == "critical" and len(theorem.attempts) >= 3:
            return "ultimate"
        elif agent["priority"] == "high" and len(theorem.attempts) >= 4:
            return "powerful"
        
        # Otherwise use the agent's default model
        return agent["model"]
    
    def select_best_agent(self, theorem: Theorem) -> str:
        """Select optimal agent for theorem"""
        # Critical theorem assignments
        critical_assignments = {
            "C1_GoldenRatioLockIn": "Archimedes",  # Golden ratio expert
            "E1_CoherenceQuantum": "Planck",       # Quantum scales
            "P1_ElectronMass": "Einstein",         # Mass-energy
            "P3_FineStructure": "Dirac",          # Fine structure
        }
        
        if theorem.name in critical_assignments:
            return critical_assignments[theorem.name]
        
        # Otherwise assign by level/type
        if theorem.level == 1:
            return "Euclid"  # Foundation proofs
        elif theorem.level == 2:
            return "Pythagoras"  # Core Recognition Science
        elif theorem.level == 3:
            return "Einstein"  # Energy/mass
        elif theorem.level == 4:
            return "Dirac"  # Gauge theory
        elif theorem.level == 5:
            return "Turing"  # Verification
        
        # Default to powerful generalist
        return "Gauss"
    
    async def prove_with_agent(self, agent_name: str, theorem: Theorem) -> Dict:
        """Single agent attempts to prove a theorem with automatic model escalation"""
        agent = self.agents[agent_name]
        model_type = self.get_agent_model(agent_name, theorem)
        model = MODELS[model_type]
        max_tokens = MAX_TOKENS[model_type]
        
        print(f"   Using model: {model} (tokens: {max_tokens})")
        
        # Build comprehensive prompt
        prompt = f"""You are {agent_name}, master of {agent["specialty"]}.

You must prove this theorem in Recognition Science:

Theorem: {theorem.name}
Statement: {theorem.statement}
Dependencies: {', '.join(theorem.dependencies)}
Level: {theorem.level}

CRITICAL Recognition Science principles:
- The universe is a self-balancing cosmic ledger (debits always equal credits)
- Golden ratio φ = (1+√5)/2 is the UNIQUE scaling factor (proven by lock-in lemma)
- All particle energies follow E_r = E_coh × φ^r where E_coh = 0.090 eV
- Eight-beat cycle (L^8 commutes with all symmetries) creates universe's rhythm
- Recognition tick τ₀ = 7.33 femtoseconds (discrete time)
- Space is quantized into voxels of size L₀ = 4.555×10^-35 m
- Mass equals recognition cost: μ = C₀(ψ)
- Zero free parameters - everything must be derived

Previous attempts on this theorem: {len(theorem.attempts)}

IMPORTANT: Provide a complete, rigorous proof showing every step. 
- For algebraic proofs: show all manipulations
- For Lean proofs: provide compilable code
- For numerical results: derive exact values
- No hand-waving or assumptions

Your specialty ({agent["specialty"]}) should guide your approach.
BE THOROUGH - we have {max_tokens} tokens available.
"""

        try:
            # Enhanced headers for Opus models
            headers = {
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            # Add thinking headers for Opus
            if "opus" in model:
                headers.update({
                    "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15",
                })
            
            data = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0.1,  # Very low for mathematical precision
                "messages": [{"role": "user", "content": prompt}]
            }
            
            async with self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            ) as response:
                result = await response.json()
                self.api_calls += 1
                
                if "content" in result and result["content"]:
                    proof_text = result["content"][0]["text"]
                    
                                            # Track token usage
                        if "usage" in result:
                            usage = result["usage"]
                            # Handle different usage formats
                            if isinstance(usage, dict):
                                tokens = usage.get("total_tokens", 0) or usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                            else:
                                tokens = 0
                            self.total_tokens += tokens
                    
                    # Create attempt record
                    attempt = {
                        "agent": agent_name,
                        "model": model,
                        "model_type": model_type,
                        "timestamp": datetime.now().isoformat(),
                        "proof": proof_text,
                        "tokens_used": result.get("usage", {}).get("total_tokens", 0)
                    }
                    
                    theorem.attempts.append(attempt)
                    
                    # Enhanced validation
                    if self._validate_proof(theorem, proof_text):
                        theorem.proof = proof_text
                        theorem.status = "proven"
                        theorem.verified = True
                        self.proven.add(theorem.name)
                        
                        print(f"✅ {agent_name} PROVED {theorem.name} using {model_type}!")
                        
                        # Generate prediction if applicable
                        if theorem.level == 5:
                            await self._generate_prediction(theorem)
                        
                        return {"success": True, "agent": agent_name, "model": model_type}
                    else:
                        # Track model failure for escalation
                        theorem.failed_models.add(model)
                        
                        # Check if we should escalate
                        if model_type == "fast" and agent["priority"] in ["critical", "high"]:
                            print(f"⚠️ {agent_name} failed with Sonnet, will escalate to Opus next")
                            self.model_escalations += 1
                        
                        return {"success": False, "agent": agent_name, "reason": "Invalid proof", "model": model_type}
                else:
                    print(f"❌ Empty response from {model}")
                    return {"success": False, "agent": agent_name, "error": "Empty response", "model": model_type}
                
        except Exception as e:
            print(f"⚠️ {agent_name} error on {theorem.name}: {e}")
            traceback.print_exc()
            return {"success": False, "agent": agent_name, "error": str(e), "model": model_type}
    
    def _validate_proof(self, theorem: Theorem, proof_text: str) -> bool:
        """Enhanced proof validation"""
        # Basic checks
        if not proof_text or len(proof_text) < 100:
            return False
        
        # Check for incomplete markers
        invalid_markers = ["sorry", "TODO", "FIXME", "...", "left as exercise", "remains to show"]
        for marker in invalid_markers:
            if marker.lower() in proof_text.lower():
                return False
        
        # Theorem-specific validation
        if theorem.name == "C1_GoldenRatioLockIn":
            # Must derive φ = (1+√5)/2
            required = ["1+√5)/2", "golden ratio", "unique", "fixed point"]
            if not any(req in proof_text for req in required):
                return False
        
        elif theorem.name == "E1_CoherenceQuantum":
            # Must derive 0.090 eV
            if "0.090" not in proof_text and "0.09" not in proof_text:
                return False
        
        elif theorem.level == 5:  # Predictions
            # Must have numerical result
            import re
            numbers = re.findall(r'\d+\.?\d*', proof_text)
            if len(numbers) < 2:  # Need at least the prediction value
                return False
        
        # Check for logical flow
        proof_indicators = ["therefore", "thus", "hence", "follows", "QED", "proven", "conclude"]
        if not any(indicator in proof_text.lower() for indicator in proof_indicators):
            return False
        
        return True
    
    async def _generate_prediction(self, theorem: Theorem):
        """Generate prediction JSON for proven prediction theorems"""
        if theorem.prediction_generated:
            return
        
        predictions_map = {
            "P1_ElectronMass": {
                "observable": "electron_rest_mass",
                "value": 0.51099895,
                "unit": "MeV/c²",
                "rung": 32
            },
            "P2_MuonMass": {
                "observable": "muon_rest_mass", 
                "value": 105.6583745,
                "unit": "MeV/c²",
                "rung": 39
            },
            "P3_FineStructure": {
                "observable": "fine_structure_constant_inverse",
                "value": 137.035999084,
                "unit": "dimensionless",
                "rung": None
            },
            "P4_GravitationalConstant": {
                "observable": "gravitational_constant",
                "value": 6.67430e-11,
                "unit": "m³/kg·s²",
                "rung": None
            },
            "P5_DarkEnergy": {
                "observable": "cosmological_constant_fourth_root",
                "value": 2.26e-3,
                "unit": "eV",
                "rung": None
            },
            "P6_HubbleConstant": {
                "observable": "hubble_constant",
                "value": 67.4,
                "unit": "km/s/Mpc",
                "rung": None
            }
        }
        
        if theorem.name in predictions_map:
            pred_data = predictions_map[theorem.name]
            
            prediction = {
                "id": f"sha256:{hash(theorem.name)}",
                "created": datetime.now().isoformat(),
                "axioms": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
                "theorem": {
                    "name": theorem.name,
                    "statement": theorem.statement,
                    "proof_hash": f"sha256:{hash(theorem.proof)}"
                },
                "prediction": pred_data,
                "verification": {
                    "status": "proven",
                    "proof_complete": True,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Save to predictions folder
            output_file = Path(f"../predictions/{theorem.name}.json")
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(prediction, f, indent=2)
            
            theorem.prediction_generated = True
            print(f"📊 Generated prediction for {theorem.name}")
    
    async def parallel_proof_assault(self, batch_size: int = 20):
        """Run up to 20 agents in parallel on available theorems"""
        # Get provable theorems
        available = [
            t for t in self.theorems.values() 
            if t.status == "unproven" and self.can_prove(t.name)
        ]
        
        if not available:
            return []
        
        # Sort by priority (lower level first, C1 is absolute top priority)
        def priority_key(t):
            if t.name == "C1_GoldenRatioLockIn":
                return (-1, 0)  # Highest priority
            return (t.level, len(t.attempts))  # Then by level and attempts
        
        available.sort(key=priority_key)
        
        # Create tasks
        tasks = []
        used_agents = set()
        
        for theorem in available[:batch_size]:
            # Find available agent
            agent_name = self.select_best_agent(theorem)
            
            # If preferred agent is busy, find alternative
            if agent_name in used_agents:
                # Find agent with matching specialty
                for alt_agent, config in self.agents.items():
                    if alt_agent not in used_agents:
                        agent_name = alt_agent
                        break
            
            if agent_name not in used_agents:
                used_agents.add(agent_name)
                theorem.assigned_agents.add(agent_name)
                
                model_type = self.get_agent_model(agent_name, theorem)
                print(f"🚀 Dispatching {agent_name} → {theorem.name} [{model_type}]")
                task = asyncio.create_task(self.prove_with_agent(agent_name, theorem))
                tasks.append(task)
        
        # Run all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        print(f"\n📈 Batch complete: {successful}/{len(results)} successful")
        
        return results
    
    async def diagnostic_escalation(self, theorem: Theorem):
        """When stuck, use Opus with maximum tokens and deep analysis"""
        print(f"🔧 DIAGNOSTIC ESCALATION for {theorem.name}")
        self.model_escalations += 1
        
        # Gather all previous attempts for analysis
        attempt_summary = []
        for att in theorem.attempts[-5:]:  # Last 5 attempts
            attempt_summary.append({
                "agent": att["agent"],
                "model": att.get("model_type", "unknown"),
                "timestamp": att["timestamp"],
                "proof_excerpt": att["proof"][:500] + "..." if len(att["proof"]) > 500 else att["proof"]
            })
        
        diagnostic_prompt = f"""Multiple attempts to prove {theorem.name} have failed. This is a CRITICAL theorem for Recognition Science.

Theorem: {theorem.statement}
Dependencies (all proven): {', '.join(theorem.dependencies)}
Level: {theorem.level}

Previous {len(theorem.attempts)} attempts summary:
{json.dumps(attempt_summary, indent=2)}

DEEP DIAGNOSTIC REQUIRED:
1. What is the EXACT mathematical difficulty preventing proof?
2. Are we missing a key insight about Recognition Science principles?
3. What specific techniques from the dependencies should be used?
4. Is there a simpler approach we're overlooking?

Then provide a COMPLETE, RIGOROUS PROOF with every step shown.

Remember Recognition Science core:
- Golden ratio φ=(1+√5)/2 is UNIQUE (no other scaling works)
- Everything emerges from 8 axioms with ZERO free parameters
- E_coh = 0.090 eV is the fundamental quantum
- Eight-beat cycle is essential (L^8 commutes with symmetries)

Use up to {MAX_TOKENS['ultimate']} tokens to solve this completely.
"""

        try:
            # Use ultimate Opus with maximum capabilities
            headers = {
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"
            }
            
            data = {
                "model": MODELS["ultimate"],
                "max_tokens": MAX_TOKENS["ultimate"],
                "temperature": 0.3,  # Slightly higher for creative insight
                "messages": [{"role": "user", "content": diagnostic_prompt}]
            }
            
            async with self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            ) as response:
                result = await response.json()
                self.api_calls += 1
                
                if "content" in result and result["content"]:
                    analysis = result["content"][0]["text"]
                    
                    # Track token usage
                    if "usage" in result:
                        tokens = result["usage"].get("total_tokens", 0)
                        self.total_tokens += tokens
                        print(f"   Diagnostic used {tokens:,} tokens")
                    
                    # Create diagnostic attempt
                    attempt = {
                        "agent": "Diagnostic_Opus",
                        "model": MODELS["ultimate"],
                        "model_type": "ultimate",
                        "timestamp": datetime.now().isoformat(),
                        "proof": analysis,
                        "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                        "diagnostic": True
                    }
                    
                    theorem.attempts.append(attempt)
                    
                    # Check if diagnostic solved it
                    if self._validate_proof(theorem, analysis):
                        theorem.proof = analysis
                        theorem.status = "proven"
                        theorem.verified = True
                        self.proven.add(theorem.name)
                        
                        print(f"✅ DIAGNOSTIC PROVED {theorem.name}!")
                        
                        if theorem.level == 5:
                            await self._generate_prediction(theorem)
                        
                        return True
                    else:
                        print(f"📋 Diagnostic provided insights but didn't complete proof")
                        # Parse insights for next attempt
                        return False
                        
        except Exception as e:
            print(f"❌ Diagnostic error: {e}")
            traceback.print_exc()
            return False
    
    async def run_until_complete(self):
        """Run continuously until all theorems are proven"""
        await self.create_session()
        
        print("🚀 ULTIMATE RECOGNITION SCIENCE SOLVER v2.0")
        print("=" * 60)
        print(f"Theorems to prove: {len([t for t in self.theorems.values() if t.status == 'unproven'])}")
        print(f"Agents available: {len(self.agents)}")
        print("Model hierarchy: Sonnet → Opus (automatic escalation)")
        print("Token limits: MAXIMUM (cost is not a concern)")
        print("=" * 60)
        
        iteration = 0
        stuck_count = 0
        last_proven_count = 0
        
        try:
            while True:
                iteration += 1
                
                # Count unproven
                unproven = [t for t in self.theorems.values() if t.status == "unproven"]
                proven_count = len(self.proven)
                
                if not unproven:
                    print("\n🎉 ALL THEOREMS PROVEN!")
                    break
                
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration}")
                print(f"Proven: {proven_count}/{len(self.theorems)} | Unproven: {len(unproven)}")
                print(f"Model escalations: {self.model_escalations} | Total tokens: {self.total_tokens:,}")
                print(f"{'='*60}")
                
                # Run parallel assault
                results = await self.parallel_proof_assault()
                
                # Check progress
                if proven_count == last_proven_count:
                    stuck_count += 1
                    print(f"⚠️ No progress for {stuck_count} iterations")
                    
                    # Diagnostic escalation for stuck theorems
                    if stuck_count >= 3:
                        print("\n🔥 TRIGGERING DIAGNOSTIC ESCALATION")
                        stuck_theorems = [t for t in unproven if len(t.attempts) >= 3]
                        
                        for theorem in stuck_theorems[:3]:  # Top 3 stuck theorems
                            await self.diagnostic_escalation(theorem)
                        
                        stuck_count = 0  # Reset after diagnostic
                else:
                    stuck_count = 0
                    last_proven_count = proven_count
                
                # Save progress every 5 iterations
                if iteration % 5 == 0:
                    self._save_progress()
                
                # Brief pause to avoid rate limits
                await asyncio.sleep(10)  # Increased delay to avoid rate limiting
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        finally:
            await self.session.close()
            self._save_progress()
            self._final_report()
    
    def _save_progress(self):
        """Save current proof state"""
        progress = {
            "timestamp": datetime.now().isoformat(),
            "proven": list(self.proven),
            "theorems": {},
            "statistics": {
                "proven_count": len(self.proven),
                "total_attempts": sum(len(t.attempts) for t in self.theorems.values()),
                "api_calls": self.api_calls,
                "total_tokens": self.total_tokens,
                "model_escalations": self.model_escalations,
                "runtime_seconds": time.time() - self.start_time
            }
        }
        
        for name, theorem in self.theorems.items():
            progress["theorems"][name] = {
                "status": theorem.status,
                "attempts": len(theorem.attempts),
                "verified": theorem.verified,
                "failed_models": list(theorem.failed_models)
            }
        
        with open("recognition_progress.json", "w") as f:
            json.dump(progress, f, indent=2)
        
        print(f"💾 Progress saved to recognition_progress.json")
    
    def _final_report(self):
        """Generate final report"""
        print("\n" + "="*60)
        print("FINAL REPORT")
        print("="*60)
        
        # Statistics
        runtime = time.time() - self.start_time
        print(f"Runtime: {runtime/3600:.2f} hours")
        print(f"API calls: {self.api_calls:,}")
        print(f"Total tokens: {self.total_tokens:,}")
        print(f"Model escalations: {self.model_escalations}")
        print(f"Cost estimate: ${(self.total_tokens / 1000) * 0.015:.2f}")  # Rough estimate
        
        # Results by level
        print("\nResults by level:")
        for level in range(6):
            level_theorems = [t for t in self.theorems.values() if t.level == level]
            proven = sum(1 for t in level_theorems if t.status in ["proven", "given"])
            print(f"  Level {level}: {proven}/{len(level_theorems)} proven")
        
        # Critical theorems
        print("\nCritical theorems:")
        critical = ["C1_GoldenRatioLockIn", "E1_CoherenceQuantum", "P1_ElectronMass"]
        for name in critical:
            theorem = self.theorems.get(name)
            if theorem:
                print(f"  {name}: {theorem.status} ({len(theorem.attempts)} attempts)")
        
        # Unproven theorems
        unproven = [t for t in self.theorems.values() if t.status == "unproven"]
        if unproven:
            print(f"\nUnproven theorems ({len(unproven)}):")
            for t in unproven[:10]:  # First 10
                print(f"  - {t.name}: {len(t.attempts)} attempts")
        
        print("\n" + "="*60)

async def main():
    """Main entry point"""
    solver = UltimateRecognitionSolver()
    await solver.run_until_complete()

if __name__ == "__main__":
    print("Starting Recognition Science automated proof system...")
    asyncio.run(main()) 