import Mathlib.Data.Real.Basic

/-!
# Physical Constants and Unit Conversions

This file defines all physical constants used in LNAL gravity theory
with proper units and conversions.
-/

namespace RecognitionScience.Constants

/-! ## Fundamental Constants -/

/-- Speed of light in m/s -/
def c : ℝ := 299792458  -- m/s (exact)

/-- Gravitational constant in m³/kg/s² -/
def G : ℝ := 6.67430e-11  -- m³/kg/s²

/-- Reduced Planck constant in J⋅s -/
def ℏ : ℝ := 1.054571817e-34  -- J⋅s

/-- Elementary charge in C -/
def e : ℝ := 1.602176634e-19  -- C (exact)

/-- Electron mass in kg -/
def m_e : ℝ := 9.1093837015e-31  -- kg

/-- Proton mass in kg -/
def m_p : ℝ := 1.67262192369e-27  -- kg

/-- Fine structure constant (dimensionless) -/
def α : ℝ := 7.2973525693e-3  -- ≈ 1/137.036

/-- Cosmological constant in m⁻² -/
def Λ : ℝ := 1.1056e-52  -- m⁻²

/-- Hubble constant in km/s/Mpc -/
def H₀ : ℝ := 67.4  -- km/s/Mpc (Planck 2018)

/-! ## Unit Conversions -/

/-- 1 parsec in meters -/
def pc : ℝ := 3.0857e16  -- m

/-- 1 kiloparsec in meters -/
def kpc : ℝ := 1000 * pc  -- m

/-- 1 megaparsec in meters -/
def Mpc : ℝ := 1e6 * pc  -- m

/-- 1 solar mass in kg -/
def M_sun : ℝ := 1.98847e30  -- kg

/-- 1 year in seconds -/
def year : ℝ := 365.25 * 24 * 3600  -- s

/-- 1 gigayear in seconds -/
def Gyr : ℝ := 1e9 * year  -- s

/-- 1 electron volt in joules -/
def eV : ℝ := e  -- J

/-- 1 MeV in joules -/
def MeV : ℝ := 1e6 * eV  -- J

/-- 1 GeV in joules -/
def GeV : ℝ := 1e9 * eV  -- J

/-! ## Recognition Science Constants -/

/-- Fundamental tick duration in seconds -/
def τ₀ : ℝ := 7.33e-15  -- s

/-- Fundamental voxel length in meters -/
def L₀ : ℝ := 0.335e-9  -- m

/-- LNAL acceleration scale in m/s² -/
def a₀ : ℝ := 1.85e-10  -- m/s²

/-- Inner recognition length in kpc -/
def ℓ₁ : ℝ := 0.97  -- kpc

/-- Outer recognition length in kpc -/
def ℓ₂ : ℝ := 24.3  -- kpc

/-- Base energy of φ-ladder in eV -/
def E₀ : ℝ := 0.090  -- eV

/-! ## Derived Quantities -/

/-- Planck length in meters -/
noncomputable def l_P : ℝ := Real.sqrt (ℏ * G / c^3)

/-- Planck time in seconds -/
noncomputable def t_P : ℝ := Real.sqrt (ℏ * G / c^5)

/-- Planck mass in kg -/
noncomputable def m_P : ℝ := Real.sqrt (ℏ * c / G)

/-- Critical density of universe in kg/m³ -/
noncomputable def ρ_crit : ℝ := 3 * (H₀ / Mpc * 1000)^2 / (8 * Real.pi * G)

/-- Dark energy density in kg/m³ -/
noncomputable def ρ_Λ : ℝ := c^2 * Λ / (8 * Real.pi * G)

/-- MOND acceleration scale (for comparison) in m/s² -/
def a_MOND : ℝ := 1.2e-10  -- m/s²

/-! ## Useful Relations -/

/-- Convert velocity in km/s to m/s -/
def km_per_s_to_m_per_s (v : ℝ) : ℝ := v * 1000

/-- Convert mass in M_⊙ to kg -/
def solar_masses_to_kg (m : ℝ) : ℝ := m * M_sun

/-- Convert distance in kpc to m -/
def kpc_to_m (d : ℝ) : ℝ := d * kpc

/-- Convert energy in eV to J -/
def eV_to_J (E : ℝ) : ℝ := E * eV

/-- Convert time in Gyr to s -/
def Gyr_to_s (t : ℝ) : ℝ := t * Gyr

/-! ## Dimensionless Ratios -/

/-- Proton to electron mass ratio -/
noncomputable def μ : ℝ := m_p / m_e  -- ≈ 1836.15

/-- Electromagnetic to gravitational force ratio -/
noncomputable def α_G : ℝ := e^2 / (4 * Real.pi * ε₀ * G * m_e * m_p)
  where ε₀ := 8.854187817e-12  -- F/m

/-- Ratio of universe age to Planck time -/
noncomputable def cosmic_time_ratio : ℝ := 13.8 * Gyr / t_P

/-- Ratio of observable universe to Planck length -/
noncomputable def cosmic_size_ratio : ℝ := c * 13.8 * Gyr / l_P

end RecognitionScience.Constants
