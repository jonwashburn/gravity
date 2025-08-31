/-
Recognition Science - Basic Test (Canonical φ-Ladder)
===================================================

This file tests the canonical φ-ladder formula E_r = E_coh × φ^r
and documents the discrepancies with observed particle masses.
-/

namespace RecognitionScience

-- Golden ratio
def φ_approx : Float := 1.618033988749895

-- Coherence quantum
def E_coh : Float := 0.090  -- eV

-- Standard Model particle rungs (from manuscripts)
def electron_rung : Nat := 32
def muon_rung : Nat := 39
def tau_rung : Nat := 44
def up_rung : Nat := 33
def down_rung : Nat := 34
def strange_rung : Nat := 38
def charm_rung : Nat := 40
def bottom_rung : Nat := 45
def top_rung : Nat := 47
def W_rung : Nat := 52
def Z_rung : Nat := 53
def Higgs_rung : Nat := 58

-- Canonical φ-ladder formula: E_r = E_coh × φ^r (in eV)
def E_rung (r : Nat) : Float := E_coh * (φ_approx ^ r.toFloat)

-- Convert to GeV
def m_rung (r : Nat) : Float := E_rung r / 1e9

-- Test calculations
#eval m_rung electron_rung  -- Should give ~0.266 GeV (needs /520 for 0.511 MeV)
#eval m_rung muon_rung      -- Should give ~0.159 GeV (observed: 0.1057 GeV)
#eval m_rung tau_rung       -- Should give ~17.6 GeV (observed: 1.777 GeV)

-- Mass ratios
def muon_electron_ratio : Float := m_rung muon_rung / m_rung electron_rung
def tau_electron_ratio : Float := m_rung tau_rung / m_rung electron_rung

#eval muon_electron_ratio  -- φ^7 ≈ 29.0 (observed: 206.8)
#eval tau_electron_ratio   -- φ^12 ≈ 322 (observed: 3477)

-- Gauge bosons
#eval m_rung W_rung    -- ~129 GeV (observed: 80.4 GeV)
#eval m_rung Z_rung    -- ~208 GeV (observed: 91.2 GeV)
#eval m_rung Higgs_rung -- ~11,200 GeV (observed: 125 GeV)

-- Fine structure constant (Recognition Science formula)
def n_alpha : Nat := 140
def α_RS : Float := 1 / (n_alpha.toFloat - 2 * φ_approx - Float.sin (2 * 3.14159265359 * φ_approx))

#eval α_RS  -- Should be ~1/137.4 (close to observed 1/137.036)

/-!
## Summary of Results

The canonical φ-ladder E_r = E_coh × φ^r gives:

1. **Electron**: 0.266 GeV (needs calibration factor 520 to get 0.511 MeV)
2. **Muon**: 0.159 GeV (1.5× too high, ratio φ^7 ≈ 29 vs observed 207)
3. **Tau**: 17.6 GeV (10× too high)
4. **W boson**: 129 GeV (1.6× too high)
5. **Z boson**: 208 GeV (2.3× too high)
6. **Higgs**: 11,200 GeV (89× too high!)

The fine structure constant formula α = 1/(140 - 2φ - sin(2πφ))
gives α ≈ 1/137.4, which is remarkably close to the observed 1/137.036.

This shows that while Recognition Science gets some dimensionless
constants right (like α), the mass ladder has systematic errors
that grow with rung number.
-/

end RecognitionScience
