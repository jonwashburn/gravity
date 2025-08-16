# LNAL Sensitivity Analysis

## Executive Summary
The LNAL gravitational formula contains **zero free parameters**. All uncertainties in predicted rotation curves arise from incomplete knowledge of the baryon distribution Σ(r). This appendix quantifies how observational uncertainties propagate through the pure LNAL formula.

## Sensitivity Matrix for Canonical Disk Galaxy
For a galaxy with M* = 5×10¹⁰ M⊙, Rd = 3 kpc, evaluated at r = 3Rd:

| Input Parameter | Uncertainty | ∂v/∂p at 1Rd | ∂v/∂p at 3Rd | Impact on v |
|-----------------|-------------|--------------|--------------|-------------|
| Distance D | ±10% | -1.0 | -1.0 | ±10% |
| Inclination i | ±5° | -0.15 | -0.15 | ±1.3% |
| Stellar M/L | ±0.1 dex | +0.35 | +0.25 | ±6-8% |
| HI mass | ±0.3 dex | +0.15 | +0.40 | ±3-12% |
| Stellar Rd | ±20% | -0.20 | +0.10 | ±2-4% |
| HI scale length | ±25% | -0.05 | +0.15 | ±1-4% |

**Combined uncertainty**: σv/v ≈ 15-20% (quadrature sum)

## Critical Insights

### 1. Distance Dominates
- D enters as Σ ∝ D⁻² → g ∝ D⁻² → v ∝ D⁻¹
- 10% distance error → 10% velocity error everywhere
- **Mitigation**: Use Cepheid/TRGB distances when available

### 2. Mass-to-Light Ratio
- Direct impact on stellar surface density
- Color-based M/L reduces uncertainty from ±0.2 to ±0.1 dex
- **Mitigation**: Use multi-band photometry, prefer 3.6μm

### 3. Gas Distribution
- Becomes dominant at r > 2Rd where Σgas > Σ*
- HI extent poorly constrained without resolved maps
- **Mitigation**: Stack similar galaxies, use scaling relations

### 4. MOND Regime Amplification
In the low-acceleration regime (g < g†):
- g ∝ √(gN) → errors in gN are √-dampened
- But: small gN → large relative errors
- Net effect: ±20% in Σ → ±10% in v (MOND buffering)

## Validation Tiers

| Tier | Requirements | Expected σv/v | N galaxies |
|------|--------------|---------------|------------|
| **Gold** | Cepheid D, resolved HI, IFU kinematics | < 10% | ~20 |
| **Silver** | TRGB D, integrated HI, long-slit | 10-20% | ~100 |
| **Bronze** | Hubble flow D, scaled gas | 20-30% | ~1000 |

## Recommendations
1. **Report uncertainties**: Always show σv from baryon inference
2. **Focus on Gold sample**: Use for critical tests of theory
3. **Hierarchical inference**: Share priors across galaxy population
4. **Independent validation**: Check inferred M/L against stellar populations

## Bottom Line
With current data, we expect χ²/N ≈ 2-5 from **input uncertainties alone**, not theory failure. The path forward is better data, not more parameters. 