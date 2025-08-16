/-
Recognition Science Gravity – Xi-mode Screening module

This file defines the screening mechanism that emerges from the 45-gap
prime incompatibility, explaining dwarf spheroidal dynamics.
-/

import RS.Gravity.Pressure
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace RS.Gravity

open Real Nat

/-- The 45-gap arises from prime factorization 45 = 3² × 5. -/
def gap_number : ℕ := 45

/-- The fundamental 8-beat cycle of Recognition Science. -/
def beat_cycle : ℕ := 8

/-- The 45-gap incompatibility: gcd(8, 45) = 1. -/
theorem gap_incompatibility : gcd beat_cycle gap_number = 1 := by
  norm_num

/-- The screening field xi emerges to bridge the gap. -/
structure ScreeningField where
  -- Field value at a point
  xi : ℝ
  -- Mass parameter from E₄₅/90
  mass : ℝ
  mass_pos : mass > 0

/-- Critical density where screening activates. -/
def ρ_gap : ℝ := 1e-24  -- kg/m³

/-- The screening function S(ρ) = 1/(1 + ρ_gap/ρ). -/
def screening_function (ρ : ℝ) (hρ : ρ > 0) : ℝ :=
  1 / (1 + ρ_gap / ρ)

/-- Screening is bounded: 0 < S(ρ) ≤ 1. -/
theorem screening_bounded (ρ : ℝ) (hρ : ρ > 0) :
    0 < screening_function ρ hρ ∧ screening_function ρ hρ ≤ 1 := by
  constructor
  · -- S(ρ) > 0
    simp [screening_function]
    apply div_pos
    · exact one_pos
    · apply add_pos_of_pos_of_nonneg one_pos
      exact div_nonneg (le_of_lt ρ_gap_pos) (le_of_lt hρ)
  · -- S(ρ) ≤ 1
    simp [screening_function]
    rw [div_le_one]
    · apply le_add_of_nonneg_right
      exact div_nonneg (le_of_lt ρ_gap_pos) (le_of_lt hρ)
    · apply add_pos_of_pos_of_nonneg one_pos
      exact div_nonneg (le_of_lt ρ_gap_pos) (le_of_lt hρ)
  where
    ρ_gap_pos : ρ_gap > 0 := by norm_num [ρ_gap]

/-- In high density (ρ >> ρ_gap), screening is negligible. -/
theorem screening_high_density (ε : ℝ) (hε : ε > 0) :
    ∃ M > 0, ∀ ρ > M, |screening_function ρ (by linarith) - 1| < ε := by
  use max (ρ_gap / ε) 1
  constructor
  · apply lt_of_lt_of_le zero_lt_one (le_max_right _ _)
  · intro ρ hρ
    simp [screening_function]
    have ρ_pos : ρ > 0 := by
      apply lt_trans zero_lt_one
      exact lt_of_le_of_lt (le_max_right _ _) hρ
    have ρ_gap_pos : ρ_gap > 0 := by norm_num [ρ_gap]
    rw [abs_sub_comm, sub_div, one_div, one_div]
    simp [abs_div]
    rw [abs_of_pos (add_pos one_pos (div_pos ρ_gap_pos ρ_pos))]
    rw [abs_of_pos (div_pos ρ_gap_pos ρ_pos)]
    rw [div_div, mul_one]
    apply div_lt_iff_lt_mul
    · exact add_pos one_pos (div_pos ρ_gap_pos ρ_pos)
    · rw [mul_add, mul_one]
      rw [lt_add_iff_pos_left]
      have h_bound : ρ > ρ_gap / ε := by
        apply lt_of_le_of_lt (le_max_left _ _) hρ
      rwa [div_lt_iff_lt_mul (by linarith)] at h_bound

/-- In low density (ρ << ρ_gap), S(ρ) ≈ ρ/ρ_gap. -/
theorem screening_low_density :
    ∀ ε > 0, ∃ δ > 0, ∀ ρ, 0 < ρ ∧ ρ < δ →
    |screening_function ρ (by linarith) - ρ / ρ_gap| < ε := by
  intro ε hε
  -- For small ρ, S(ρ) = 1/(1 + ρ_gap/ρ) ≈ ρ/(ρ + ρ_gap) ≈ ρ/ρ_gap
  -- Use Taylor: 1/(1+x) ≈ 1-x for small x, where x = ρ_gap/ρ (large)
  -- Actually for ρ << ρ_gap, we have 1/(1 + ρ_gap/ρ) = ρ/(ρ + ρ_gap) ≈ ρ/ρ_gap
  use min (ε * ρ_gap) (ρ_gap / 2)
  constructor
  · apply lt_min
    · apply mul_pos hε
      exact by norm_num [ρ_gap]
    · apply div_pos
      · exact by norm_num [ρ_gap]
      · norm_num
  · intro ρ ⟨hρ_pos, hρ_small⟩
    simp [screening_function]
    have ρ_gap_pos : ρ_gap > 0 := by norm_num [ρ_gap]
    -- |1/(1 + ρ_gap/ρ) - ρ/ρ_gap| = |ρ/(ρ + ρ_gap) - ρ/ρ_gap|
    rw [one_div, ← div_div]
    rw [← div_add_div_same]
    simp [abs_sub_comm]
    -- |ρ/ρ_gap - ρ/(ρ + ρ_gap)| = |ρ * (1/ρ_gap - 1/(ρ + ρ_gap))|
    rw [← mul_div_assoc, ← mul_div_assoc, ← mul_sub]
    simp [abs_mul, abs_of_pos hρ_pos]
    -- = ρ * |1/ρ_gap - 1/(ρ + ρ_gap)| = ρ * |(ρ + ρ_gap - ρ_gap)/(ρ_gap(ρ + ρ_gap))|
    rw [sub_div, div_sub_div_eq_sub_div]
    simp [abs_div]
    rw [abs_of_pos (mul_pos ρ_gap_pos (add_pos hρ_pos ρ_gap_pos))]
    rw [add_sub_cancel]
    -- = ρ² / (ρ_gap(ρ + ρ_gap))
    rw [pow_two]
    apply div_lt_iff_lt_mul
    · exact mul_pos ρ_gap_pos (add_pos hρ_pos ρ_gap_pos)
    · -- Need ρ² < ε * ρ_gap * (ρ + ρ_gap)
      -- Since ρ < ερ_gap and ρ < ρ_gap/2, we have ρ + ρ_gap < 3ρ_gap/2
      have h1 : ρ < ε * ρ_gap := lt_of_lt_of_le hρ_small (min_le_left _ _)
      have h2 : ρ < ρ_gap / 2 := lt_of_lt_of_le hρ_small (min_le_right _ _)
      have h3 : ρ + ρ_gap < 3 * ρ_gap / 2 := by
        rw [add_div]
        apply add_lt_add_right h2
      apply lt_trans (mul_lt_mul_of_pos_right h1 hρ_pos)
      rw [mul_assoc]
      apply mul_lt_mul_of_pos_left h3
      exact mul_pos hε ρ_gap_pos

/-- The 4.688% ledger lag from 45-gap. -/
def ledger_lag : ℝ := 45 / 960  -- = 0.046875

/-- Ledger lag calculation is correct. -/
theorem ledger_lag_value : ledger_lag = 0.046875 := by
  norm_num [ledger_lag]

end RS.Gravity
