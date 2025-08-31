/-
  Physical Units and Constants
  ============================

  Dimensional analysis support for physics in Lean.
  Provides type-safe units and fundamental constants.
-/

import Mathlib.Data.Real.Basic

namespace RecognitionScience.Units

-- Basic dimensions
structure Dimension where
  length : ℤ
  mass : ℤ
  time : ℤ
  deriving DecidableEq

-- Quantity with dimension
structure Quantity (d : Dimension) where
  value : ℝ
  deriving Inhabited

-- Arithmetic for dimensioned quantities
instance : Add (Quantity d) where
  add q₁ q₂ := ⟨q₁.value + q₂.value⟩

instance : Sub (Quantity d) where
  sub q₁ q₂ := ⟨q₁.value - q₂.value⟩

instance : Mul (Quantity d₁) (Quantity d₂) where
  mul q₁ q₂ := ⟨q₁.value * q₂.value⟩

-- Standard dimensions
def dimensionless : Dimension := ⟨0, 0, 0⟩
def length : Dimension := ⟨1, 0, 0⟩
def mass : Dimension := ⟨0, 1, 0⟩
def time : Dimension := ⟨0, 0, 1⟩
def velocity : Dimension := ⟨1, 0, -1⟩
def acceleration : Dimension := ⟨1, 0, -2⟩
def energy : Dimension := ⟨2, 1, -2⟩
def power : Dimension := ⟨2, 1, -3⟩

-- Unit constructors
def meter (x : ℝ) : Quantity length := ⟨x⟩
def kilogram (x : ℝ) : Quantity mass := ⟨x⟩
def second (x : ℝ) : Quantity time := ⟨x⟩
def joule (x : ℝ) : Quantity energy := ⟨x⟩
def watt (x : ℝ) : Quantity power := ⟨x⟩

-- Fundamental constants
namespace Constants

def c : Quantity velocity := ⟨2.99792458e8⟩  -- m/s
def G : ℝ := 6.67430e-11  -- m³/kg/s²
def ℏ : Quantity ⟨2, 1, -1⟩ := ⟨1.054571817e-34⟩  -- J⋅s
def k_B : ℝ := 1.380649e-23  -- J/K

-- Planck units
def t_Planck : Quantity time := ⟨5.391247e-44⟩
def ℓ_Planck : Quantity length := ⟨1.616255e-35⟩
def m_Planck : Quantity mass := ⟨2.176434e-8⟩

-- Recognition Science constants
def τ₀ : Quantity time := ⟨7.33e-15⟩  -- fundamental tick
def E_coh : Quantity energy := ⟨1.44e-20⟩  -- coherence cost
def φ : ℝ := (1 + Real.sqrt 5) / 2  -- golden ratio

end Constants

lemma dimension_injective : Function.Injective dimension := by
  intro q1 q2 h
  -- h : dimension q1 = dimension q2
  -- We need to show q1 = q2
  cases q1 with
  | mass =>
    cases q2 with
    | mass => rfl
    | length => simp [dimension] at h
    | time => simp [dimension] at h
    | dimensionless => simp [dimension] at h
  | length =>
    cases q2 with
    | mass => simp [dimension] at h
    | length => rfl
    | time => simp [dimension] at h
    | dimensionless => simp [dimension] at h
  | time =>
    cases q2 with
    | mass => simp [dimension] at h
    | length => simp [dimension] at h
    | time => rfl
    | dimensionless => simp [dimension] at h
  | dimensionless =>
    cases q2 with
    | mass => simp [dimension] at h
    | length => simp [dimension] at h
    | time => simp [dimension] at h
    | dimensionless => rfl

end RecognitionScience.Units
