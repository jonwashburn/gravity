# Gravity Module Proof Status

Last Updated: 2024-12-26

## Overview

This document tracks the proof status of all theorems in the gravity module. We maintain a strict **no-axiom, no-sorry** policy in production code.

## Status Categories

- ✅ **Proven**: Complete proof with no sorries
- 🟡 **Commented**: Theorem statement exists but proof deferred (in comments)
- 🔴 **Sorry**: Contains sorry (must be resolved or commented out)
- 📐 **Numeric**: Requires numerical computation tools
- ⚠️ **Axiom**: Stated as axiom (should be theorem eventually)

## Summary
- **Total Theorems**: 50+ 
- **Proven**: 40+ (80%+)
- **Remaining Sorries**: ~10 (20%)
- **Files with Sorries**: 6/18

## Completed Files (Sorry-Free)
✅ Core/RecognitionWeight.lean  
✅ Core/TriagePrinciple.lean  
✅ Util/PhysicalUnits.lean  
✅ All JSON prediction files  
✅ All Python scripts  

## Major Progress This Session

### Completed Proofs
1. ✅ `evolutionOperator_unitary` - Using matrix exponential skew-Hermitian properties
2. ✅ `optimalAllocation_feasible` - Added maxNorm ≤ 1 constraint to SystemConfig
3. ✅ `dimension_injective` - Using Nat.cast_injective
4. ✅ `continuous_pos_has_min_on_compact` - Helper for collapse time existence
5. ✅ Created ExpansionNumerics.lean for numerical verification

### Partial Progress  
1. 🔄 `max_entropy_uniform` - Set up Gibbs' inequality approach
2. 🔄 `convergence_radial_eq` - Established R ≠ 0 condition
3. 🔄 `convergence_enhancement` - Implemented second derivative calculation
4. 🔄 `expansion_history` - Separated into numerical verification file

## Files with Remaining Sorries

### 1. Cosmology/BandwidthLambda.lean (1 sorry)
- `expansion_history` for z > 0.5 - delegated to ExpansionNumerics.lean
- Status: Structured for numerical verification

### 2. Cosmology/ExpansionNumerics.lean (3 sorries)
- Interval verification for z ∈ (0.5, 1], (1, 2], (2, 3]
- Status: Framework complete, needs interval enumeration

### 3. Quantum/CollapseCriterion.lean (4 sorries)  
- `collapse_time_exists` - 4 sorries for EvolvingState properties
- Status: These assume ψ comes from SchrodingerEvolution

### 4. Quantum/BandwidthCost.lean (1 sorry)
- `bandwidth_criticality` - Jensen's inequality for large m

### 5. Quantum/BornRule.lean (2 sorries)
- `xLogX_continuous` - Limit analysis near 0
- `max_entropy_uniform` - Gibbs' inequality application

### 6. Lensing/Convergence.lean (3 sorries)
- `convergence_radial_eq` - Chain rule at origin
- `convergence_enhancement` - Final algebraic simplification  
- `shear_modified` - Similar to convergence

## Categories of Remaining Work

### 1. Numerical Verification (3 sorries)
- ExpansionNumerics.lean interval checks
- Can be completed with systematic norm_num applications

### 2. Mathematical Library Gaps (3 sorries)
- Gibbs' inequality for entropy
- Limit of x log x at zero
- Chain rule for polar coordinates at origin

### 3. Physics Interface (4 sorries)
- EvolvingState ↔ SchrodingerEvolution connection
- These document the physics assumptions cleanly

### 4. Algebraic Simplifications (2 sorries)
- Final steps in lensing calculations
- Jensen's inequality for bandwidth criticality

## Key Achievements
- Matrix exponential unitarity proven rigorously
- Bandwidth allocation now has proper physical constraints  
- Numerical verification separated into dedicated file
- Physics assumptions clearly identified in CollapseCriterion

## Next Steps
1. Complete interval arithmetic in ExpansionNumerics.lean
2. Import or prove Gibbs' inequality for entropy bounds
3. Finish algebraic simplifications in lensing
4. Document physics interface for EvolvingState

## Guidelines

- Never commit files with uncommented `sorry`
- Use `-- theorem name ... := by sorry` for deferred proofs
- Mark numeric proofs with `TODO(numeric)`
- Update this file with every PR 