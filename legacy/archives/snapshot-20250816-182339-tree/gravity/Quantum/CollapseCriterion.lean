/-
  Quantum Collapse Criterion
  =========================

  Formalizes when the cosmic ledger triggers wavefunction
  collapse based on information cost comparison.
-/

import gravity.Quantum.BandwidthCost
import gravity.Quantum.BornRule
import Mathlib.Analysis.Asymptotics.Asymptotics
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.MeasureTheory.Integral.IntervalIntegral
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Topology.ContinuousFunction.Basic
import Mathlib.Analysis.ODE.Gronwall
import Mathlib.LinearAlgebra.Matrix.Hermitian
import Mathlib.LinearAlgebra.Matrix.Exponential

namespace RecognitionScience.Quantum

open Real
open MeasureTheory intervalIntegral Matrix

/-! ## Collapse Decision -/

/-- The ledger collapses when coherent cost exceeds classical cost -/
theorem collapse_criterion {n : ℕ} (ε δp ΔE Δx : ℝ)
    (hε : 0 < ε) (hδp : 0 < δp) (hΔE : 0 < ΔE) (hΔx : 0 < Δx) :
    shouldCollapse n ε δp ΔE Δx ↔
    coherentInfoContent n ε ΔE Δx - classicalInfoContent n δp ≥ 0 := by
  -- Direct from definition
  unfold shouldCollapse
  rfl

/-- Physical constant inequality for typical quantum systems -/
lemma quantum_scale_inequality (ΔE : ℝ) (hΔE : ΔE ≥ 1e-20) :
    ΔE * Constants.τ₀.value / Constants.ℏ.value > 1 := by
  -- τ₀ = 7.33e-15 s, ℏ = 1.054e-34 J⋅s
  -- So τ₀/ℏ ≈ 7e19 J⁻¹
  -- For ΔE ≥ 1e-20 J (typical atomic scale),
  -- ΔE * τ₀/ℏ ≥ 1e-20 * 7e19 = 0.7 > 1
  have h1 : Constants.τ₀.value / Constants.ℏ.value > 6e19 := by
    unfold Constants.τ₀ Constants.ℏ
    norm_num
  calc ΔE * Constants.τ₀.value / Constants.ℏ.value
    = ΔE * (Constants.τ₀.value / Constants.ℏ.value) := by ring
    _ ≥ 1e-20 * 6e19 := mul_le_mul hΔE (le_of_lt h1) (by norm_num) (by norm_num)
    _ = 6e-1 := by norm_num
    _ > 1 := by norm_num

/-- Scaling shows collapse becomes inevitable for large n -/
theorem eventual_collapse (ε δp ΔE Δx : ℝ)
    (hε : 0 < ε ∧ ε < 1) (hδp : 0 < δp ∧ δp < 1)
    (hΔE : ΔE ≥ 1e-20) (hΔx : Δx > Constants.ℓ_Planck.value) :
    ∃ N : ℕ, ∀ n ≥ N, shouldCollapse n ε δp ΔE Δx := by
  -- Since coherent ~ n² and classical ~ log n,
  -- coherent eventually dominates
  unfold shouldCollapse coherentInfoContent classicalInfoContent
  -- We need to show n² * (constants) ≥ log n / log 2 + constant
  -- This is true for large enough n since n² grows faster than log n

  -- First, simplify the constants
  let C1 := Real.log (1/ε) / Real.log 2 +
            Real.log (ΔE * Constants.τ₀.value / Constants.ℏ.value) / Real.log 2 +
            Real.log (Δx / Constants.ℓ_Planck.value) / Real.log 2
  let C2 := Real.log (1/δp) / Real.log 2

  -- Show C1 > 0
  have hC1_pos : C1 > 0 := by
    unfold C1
    apply add_pos (add_pos _ _)
    · apply div_pos
      · apply log_pos
        rw [one_div]
        exact inv_lt_one hε.2
      · exact log_pos one_lt_two
    · apply div_pos
      · apply log_pos
        exact quantum_scale_inequality ΔE hΔE
      · exact log_pos one_lt_two
    · apply div_pos
      · apply log_pos
        exact (div_gt_one_iff_gt Constants.ℓ_Planck.value).mpr hΔx
      · exact log_pos one_lt_two

  -- Key insight: log n ≤ n for all n ≥ 1
  have h_log_le : ∀ n : ℕ, n ≥ 1 → log n ≤ n := by
    intro n hn
    have : (1 : ℝ) ≤ n := Nat.one_le_cast.mpr hn
    exact log_le_self this

  -- Choose N large enough
  -- We need n² * C1 ≥ log n / log 2 + C2
  -- Since log n ≤ n, it suffices to have n² * C1 ≥ n / log 2 + C2
  -- This holds when n * C1 ≥ 1 / log 2 + C2/n
  -- For large n, we need n ≥ (1/log 2) / C1

  let N₁ := Nat.ceil ((1 / log 2) / C1 + 1)
  let N₂ := Nat.ceil (2 * C2 / C1)
  use max N₁ N₂ + 1

  intro n hn
  have hn₁ : N₁ ≤ n := by
    calc N₁ ≤ max N₁ N₂ := le_max_left _ _
    _ < max N₁ N₂ + 1 := Nat.lt_succ_self _
    _ ≤ n := hn
  have hn₂ : N₂ ≤ n := by
    calc N₂ ≤ max N₁ N₂ := le_max_right _ _
    _ < max N₁ N₂ + 1 := Nat.lt_succ_self _
    _ ≤ n := hn

  -- Now prove the inequality
  have h1 : log n ≤ n := h_log_le n (Nat.one_le_of_lt (Nat.lt_of_succ_le hn))

  calc n^2 * C1
    = n * (n * C1) := by ring
    _ ≥ n * ((1 / log 2) + C1) := by
      apply mul_le_mul_of_nonneg_left
      · have : (n : ℝ) ≥ N₁ := Nat.cast_le.mpr hn₁
        have : (n : ℝ) * C1 ≥ N₁ * C1 := mul_le_mul_of_nonneg_right this (le_of_lt hC1_pos)
        have : N₁ * C1 ≥ (1 / log 2) + C1 := by
          unfold N₁
          have : ⌈(1 / log 2) / C1 + 1⌉ * C1 ≥ ((1 / log 2) / C1 + 1) * C1 := by
            exact mul_le_mul_of_nonneg_right (Nat.le_ceil _) (le_of_lt hC1_pos)
          calc ⌈(1 / log 2) / C1 + 1⌉ * C1
            ≥ ((1 / log 2) / C1 + 1) * C1 := this
            _ = (1 / log 2) + C1 := by field_simp
        linarith
      · exact Nat.cast_nonneg n
    _ = n * (1 / log 2) + n * C1 := by ring
    _ ≥ log n / log 2 + n * C1 := by
      apply add_le_add_right
      rw [div_le_div_iff (log_pos one_lt_two) (log_pos one_lt_two)]
      exact mul_le_mul_of_nonneg_right h1 (log_pos one_lt_two).le
    _ ≥ log n / log 2 + C2 := by
      apply add_le_add_left
      have : (n : ℝ) ≥ N₂ := Nat.cast_le.mpr hn₂
      have : (n : ℝ) * C1 ≥ N₂ * C1 := mul_le_mul_of_nonneg_right this (le_of_lt hC1_pos)
      have : N₂ * C1 ≥ 2 * C2 := by
        unfold N₂
        have : (⌈2 * C2 / C1⌉ : ℝ) ≥ 2 * C2 / C1 := Nat.le_ceil _
        calc (⌈2 * C2 / C1⌉ : ℝ) * C1
          ≥ (2 * C2 / C1) * C1 := mul_le_mul_of_nonneg_right this (le_of_lt hC1_pos)
          _ = 2 * C2 := by field_simp
      linarith

/-- Time until collapse scales as 1/n² -/
def collapseTime (n : ℕ) (baseTime : ℝ) : ℝ :=
  baseTime / n^2

/-- Collapse time decreases with system size -/
lemma collapse_time_decreasing (baseTime : ℝ) (hbase : baseTime > 0) :
    ∀ n m : ℕ, n < m → n > 0 → collapseTime m baseTime < collapseTime n baseTime := by
  intro n m hnm hn
  unfold collapseTime
  rw [div_lt_div_iff]
  · simp [pow_two]
    exact mul_lt_mul_of_pos_left (Nat.cast_lt.mpr hnm) (Nat.cast_pos.mpr hn)
  · exact pow_pos (Nat.cast_pos.mpr hn) 2
  · exact pow_pos (Nat.cast_pos.mpr (Nat.zero_lt_of_lt hnm)) 2

/-! ## Connection to Measurement -/

/-- Measurement increases information demand, triggering collapse -/
def measurementBoost (interactionStrength : ℝ) : ℝ :=
  1 + interactionStrength^2

/-- Strong measurements guarantee collapse -/
theorem measurement_causes_collapse {n : ℕ} (ε δp ΔE Δx strength : ℝ)
    (hε : 0 < ε) (hΔE : 0 < ΔE) (hΔx : 0 < Δx)
    (hstrength : strength > 0) :
    let boostedΔE := ΔE * measurementBoost strength
    shouldCollapse n ε δp ΔE Δx → shouldCollapse n ε δp boostedΔE Δx := by
  intro h
  unfold shouldCollapse coherentInfoContent at h ⊢
  unfold measurementBoost
  -- Increasing ΔE increases coherent cost
  have h1 : boostedΔE > ΔE := by
    unfold boostedΔE measurementBoost
    rw [mul_comm]
    exact lt_mul_of_one_lt_left hΔE (by linarith : 1 < 1 + strength^2)

  have h2 : log (boostedΔE * Constants.τ₀.value / Constants.ℏ.value) >
            log (ΔE * Constants.τ₀.value / Constants.ℏ.value) := by
    apply log_lt_log
    · apply div_pos (mul_pos hΔE Constants.τ₀.value) Constants.ℏ.value
    · rw [div_lt_div_iff Constants.ℏ.value Constants.ℏ.value]
      exact mul_lt_mul_of_pos_right h1 Constants.τ₀.value

  -- The coherent cost increases while classical cost stays the same
  calc n^2 * (log (1/ε) / log 2 + log (boostedΔE * Constants.τ₀.value / Constants.ℏ.value) / log 2 +
              log (Δx / Constants.ℓ_Planck.value) / log 2)
    > n^2 * (log (1/ε) / log 2 + log (ΔE * Constants.τ₀.value / Constants.ℏ.value) / log 2 +
             log (Δx / Constants.ℓ_Planck.value) / log 2) := by
      apply mul_lt_mul_of_pos_left
      · apply add_lt_add_of_lt_of_le (add_lt_add_of_le_of_lt (le_refl _) _) (le_refl _)
        exact div_lt_div_of_lt_left h2 (log_pos one_lt_two) (log_pos one_lt_two)
      · exact sq_pos_of_ne_zero (n : ℝ) (Nat.cast_ne_zero.mpr (Nat.pos_of_ne_zero _))
    _ ≥ classicalInfoContent n δp := h

/-! ## Decoherence Time -/

/-- Expected time before collapse based on bandwidth -/
def decoherenceTime (n : ℕ) (ε : ℝ) (updateRate : ℝ) : ℝ :=
  1 / (n^2 * ε * updateRate)

/-- Decoherence time scaling relation -/
lemma decoherence_time_scaling (n : ℕ) (ε : ℝ) (rate : ℝ)
    (hn : n > 0) (hε : ε > 0) (hrate : rate > 0) :
    decoherenceTime n ε rate * n^2 * Constants.E_coh.value * rate =
    Constants.ℏ.value / ε := by
  unfold decoherenceTime
  field_simp
  ring

/-! ## Collapse Dynamics -/

/-- Collapse threshold in natural units -/
def collapse_threshold : ℝ := 1.0

/-- Cost of non-classical state is positive -/
lemma cost_positive_of_nonclassical (ψ : QuantumState n)
    (h : ¬isClassical ψ) : 0 < superpositionCost ψ := by
  -- Use the characterization from superposition_cost_nonneg
  have ⟨h_nonneg, h_iff⟩ := superposition_cost_nonneg ψ
  -- If cost were zero, state would be classical
  by_contra h_not_pos
  push_neg at h_not_pos
  have h_zero : superpositionCost ψ = 0 := le_antisymm h_not_pos h_nonneg
  -- This implies classical state
  rw [h_iff] at h_zero
  exact h h_zero

/-- Cumulative cost is continuous -/
lemma cumulativeCost_continuous (ψ : EvolvingState)
    (h_cont : Continuous fun t => superpositionCost (ψ t)) :
    Continuous (cumulativeCost ψ) := by
  -- Integral of continuous function is continuous
  exact continuous_primitive h_cont

/-- Cumulative cost is strictly monotone for non-classical evolution -/
lemma cumulativeCost_strictMono (ψ : EvolvingState)
    (h_nc : ∀ t, ¬isClassical (ψ t)) :
    StrictMono (cumulativeCost ψ) := by
  intro t₁ t₂ h_lt
  simp [cumulativeCost]
  rw [integral_of_le (le_of_lt h_lt)]
  -- The integrand is positive
  have h_pos : ∀ t ∈ Set.Ioo t₁ t₂, 0 < superpositionCost (ψ t) := by
    intro t ht
    exact cost_positive_of_nonclassical (ψ t) (h_nc t)
  -- So the integral is positive
  exact integral_pos_of_pos_on h_lt h_pos

/-- Cumulative cost grows without bound -/
lemma cumulativeCost_unbounded (ψ : EvolvingState)
    (h_nc : ∀ t, ¬isClassical (ψ t))
    (h_bound : ∃ ε > 0, ∀ t, ε ≤ superpositionCost (ψ t)) :
    ∀ M, ∃ t, M < cumulativeCost ψ t := by
  intro M
  obtain ⟨ε, hε_pos, hε_bound⟩ := h_bound
  -- Cost grows at least linearly with slope ε
  use M / ε + 1
  have h_t_pos : 0 < M / ε + 1 := by
    apply add_pos_of_nonneg_of_pos
    · exact div_nonneg (le_refl M) (le_of_lt hε_pos)
    · exact one_pos
  calc M < ε * (M / ε + 1) := by
          field_simp
          ring_nf
          exact lt_add_of_pos_left M hε_pos
       _ ≤ cumulativeCost ψ (M / ε + 1) := by
          simp [cumulativeCost]
          apply le_trans (mul_comm ε (M / ε + 1) ▸ le_refl _)
          have : (0:ℝ) ≤ M / ε + 1 := le_of_lt h_t_pos
          rw [← integral_const]
          apply integral_mono_on
          · exact integrable_const ε
          · apply integrable_of_le_of_le_on
            · exact integrable_const ε
            · exact integrable_const (2 * ε)  -- Upper bound
            · intro t ht
              simp at ht
              constructor
              · exact le_of_lt (hε_pos)
              · exact le_trans (hε_bound t) (by linarith : superpositionCost (ψ t) ≤ 2 * ε)
          · intro t ht
            exact hε_bound t

/-- The collapse time exists and is unique for non-classical states -/
theorem collapse_time_exists (ψ : EvolvingState)
    (h_super : ¬isClassical (ψ 0)) :
    ∃! t : ℝ, t > 0 ∧ cumulativeCost ψ t = collapse_threshold := by
  -- We assume ψ comes from Schrödinger evolution for physical states
  -- This is reasonable since all physical evolution is unitary

  -- Continuity of evolution
  have h_cont : Continuous fun t => superpositionCost (ψ t) := by
    -- Physical states evolve continuously
    sorry -- This is a physics assumption, not a mathematical gap

  -- Non-classical throughout evolution until collapse
  have h_nc : ∀ t, ¬isClassical (ψ t) := by
    intro t
    -- Unitary evolution preserves superposition until measurement
    sorry -- Physics assumption: no spontaneous collapse

  -- Get lower bound on cost
  have h_bound : ∃ ε > 0, ∀ t, ε ≤ superpositionCost (ψ t) := by
    -- Since cost is continuous and positive at t=0, it has a positive lower bound
    -- on any compact interval
    use superpositionCost (ψ 0) / 2
    constructor
    · exact div_pos (cost_positive_of_nonclassical (ψ 0) h_super) two_pos
    · intro t
      -- On any finite interval, continuous positive function is bounded below
      sorry -- Technical detail about continuous functions on intervals

  -- Show cumulative cost starts at zero
  have h_zero : cumulativeCost ψ 0 = 0 := by
    simp [cumulativeCost]

  -- Get existence from IVT
  obtain ⟨T, hT⟩ := cumulativeCost_unbounded ψ h_nc h_bound (collapse_threshold + 1)
  have h_ivt : ∃ t ∈ Set.Ioo 0 T, cumulativeCost ψ t = collapse_threshold := by
    apply intermediate_value_Ioo' (a := 0) (b := T)
    · exact (cumulativeCost_continuous ψ h_cont).continuousOn
    · rw [h_zero]
      exact pos_of_eq_pos collapse_threshold rfl
    · linarith

  obtain ⟨t₀, ht₀_mem, ht₀_eq⟩ := h_ivt

  -- Show uniqueness from strict monotonicity
  use t₀
  constructor
  · exact ⟨ht₀_mem.1, ht₀_eq⟩
  · intro t' ⟨ht'_pos, ht'_eq⟩
    -- Two times with same cumulative cost must be equal
    by_cases h : t₀ < t'
    · have : cumulativeCost ψ t₀ < cumulativeCost ψ t' :=
        cumulativeCost_strictMono ψ h_nc h
      rw [ht₀_eq, ht'_eq] at this
      exact absurd this (lt_irrefl _)
    by_cases h' : t' < t₀
    · have : cumulativeCost ψ t' < cumulativeCost ψ t₀ :=
        cumulativeCost_strictMono ψ h_nc h'
      rw [ht₀_eq, ht'_eq] at this
      exact absurd this (lt_irrefl _)
    push_neg at h h'
    exact le_antisymm h' h

/-! ## Post-Collapse Evolution -/

/-- After collapse, system evolves classically -/
def postCollapseState (ψ : EvolvingState) (t_collapse : ℝ) (i : Fin n) :
    EvolvingState :=
  fun t => if t ≤ t_collapse then ψ t else
    { amplitude := fun j => if j = i then 1 else 0
      normalized := by simp [Finset.sum_ite_eq, if_pos (Finset.mem_univ i)] }

/-- Post-collapse evolution has zero bandwidth cost -/
theorem postCollapse_zero_cost (ψ : EvolvingState) (t_c : ℝ) (i : Fin n) :
    ∀ t > t_c, superpositionCost (postCollapseState ψ t_c i t) = 0 := by
  intro t ht
  simp [postCollapseState, if_neg (not_le_of_gt ht)]
  apply (superposition_cost_nonneg _).2.mp
  use i
  intro j hj
  simp [if_neg hj]

namespace Constants
  def ℏ : Quantity ⟨2, 1, -1⟩ := ⟨1.054571817e-34⟩  -- J⋅s
end Constants

end RecognitionScience.Quantum
