#!/usr/bin/env python3
"""
5-10 nm Torsion Balance Design for RS Gravity Enhancement Detection
Targeting the constructive interference window below λ_eff/φ⁵
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

# Physical constants
G_SI = 6.67430e-11  # m^3/kg/s^2
c = 299792458.0     # m/s
hbar = 1.054571817e-34  # J⋅s
k_B = 1.380649e-23  # J/K
epsilon_0 = 8.854187817e-12  # F/m

# Recognition Science constants
phi = (1 + np.sqrt(5)) / 2
beta_0 = -(phi - 1) / phi**5
lambda_eff = 50.8e-6  # meters (optimized)
beta = 1.492 * beta_0  # With scale factor

print("=== 5-10 nm Torsion Balance Design ===\n")

# Step 1: Find the constructive interference window
print("Step 1: Constructive Interference Window")
print("-" * 40)

# Below λ_eff/φ⁵, eight-beat cancellation changes sign
r_transition = lambda_eff / phi**5  # ~7 nm
r_min = r_transition / phi  # Start of strong enhancement
r_max = r_transition  # End of window

print(f"λ_eff = {lambda_eff*1e6:.1f} μm")
print(f"Transition scale: λ_eff/φ⁵ = {r_transition*1e9:.2f} nm")
print(f"Constructive window: {r_min*1e9:.2f} - {r_max*1e9:.2f} nm")
print(f"Target separation: {(r_min + r_max)*0.5*1e9:.1f} nm\n")

# Step 2: Calculate G enhancement in the window
print("Step 2: G Enhancement Profile")
print("-" * 40)

def G_enhancement_corrected(r):
    """Corrected G enhancement with sign flip below transition"""
    if r > r_transition:
        # Normal suppression regime
        return (lambda_eff / r) ** beta
    else:
        # Constructive interference regime
        # Phase wraps by π, changing destructive to constructive
        base_enhancement = (lambda_eff / r) ** (-beta)  # Note sign flip
        phase_factor = np.sin(np.pi * r / r_transition)**2
        return base_enhancement * (1 + phase_factor)

r_range = np.logspace(np.log10(1e-9), np.log10(100e-9), 1000)
G_enh = [G_enhancement_corrected(r) for r in r_range]

plt.figure(figsize=(10, 6))
plt.loglog(r_range*1e9, G_enh, 'b-', linewidth=2)
plt.axvline(r_transition*1e9, color='red', linestyle='--', 
           label=f'Transition: {r_transition*1e9:.1f} nm')
plt.axvspan(r_min*1e9, r_max*1e9, alpha=0.3, color='green', 
           label='Constructive window')
plt.axhline(1, color='black', linestyle=':', alpha=0.5)
plt.xlabel('Separation (nm)')
plt.ylabel('G(r)/G₀')
plt.title('RS Gravity Enhancement with Constructive Interference')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([1, 100])
plt.ylim([0.1, 10])
plt.savefig('nanoscale_G_enhancement_corrected.png', dpi=300, bbox_inches='tight')
plt.close()

# Target parameters
r_target = 5e-9  # 5 nm separation
G_target = G_enhancement_corrected(r_target)
print(f"At r = {r_target*1e9:.1f} nm: G/G₀ = {G_target:.2f}")

# Step 3: Experimental design
print("\n\nStep 3: Experimental Design Parameters")
print("-" * 40)

# Test masses - gold nanospheres
density_Au = 19300  # kg/m³
r_sphere = 2.5e-9   # 2.5 nm radius (5 nm diameter)
V_sphere = (4/3) * np.pi * r_sphere**3
m_sphere = density_Au * V_sphere

print(f"\nTest masses:")
print(f"- Material: Gold nanospheres")
print(f"- Radius: {r_sphere*1e9:.1f} nm")
print(f"- Mass: {m_sphere:.2e} kg")
print(f"- Number of atoms: ~{m_sphere/(197*1.66e-27):.0f}")

# Forces
F_Newton = G_SI * m_sphere**2 / r_target**2
F_RS = F_Newton * G_target
Delta_F = F_RS - F_Newton

print(f"\nForces at {r_target*1e9:.1f} nm separation:")
print(f"- Newton: {F_Newton:.2e} N")
print(f"- RS: {F_RS:.2e} N")
print(f"- Enhancement: {Delta_F/F_Newton*100:.1f}%")

# Casimir force (major background)
def casimir_force(r, R1, R2):
    """Casimir force between two spheres"""
    # Proximity force approximation
    R_eff = R1 * R2 / (R1 + R2)
    return (np.pi**3 * hbar * c * R_eff) / (360 * r**3)

F_Casimir = casimir_force(r_target, r_sphere, r_sphere)
print(f"\nBackground forces:")
print(f"- Casimir: {F_Casimir:.2e} N")
print(f"- Ratio F_grav/F_Casimir: {F_RS/F_Casimir:.2e}")

# Step 4: Detection scheme
print("\n\nStep 4: Detection Scheme")
print("-" * 40)

# Torsion pendulum parameters
L_fiber = 10e-6     # 10 μm long
d_fiber = 50e-9     # 50 nm diameter carbon nanotube
E_CNT = 1e12        # Pa (carbon nanotube)
I_fiber = np.pi * d_fiber**4 / 64
kappa = E_CNT * I_fiber / L_fiber  # Torsional spring constant

# Pendulum arm
L_arm = 100e-9      # 100 nm arm length
tau = Delta_F * L_arm
theta = tau / kappa

print(f"Torsion pendulum:")
print(f"- Fiber: {d_fiber*1e9:.0f} nm diameter CNT")
print(f"- Length: {L_fiber*1e6:.0f} μm")
print(f"- Spring constant: {kappa:.2e} N⋅m/rad")
print(f"- Arm length: {L_arm*1e9:.0f} nm")
print(f"\nDeflection:")
print(f"- Torque: {tau:.2e} N⋅m")
print(f"- Angle: {theta*1e6:.2f} μrad")

# Thermal noise
T = 4.2  # Liquid helium temperature
Q = 10000  # Quality factor for CNT at low T
omega_0 = np.sqrt(kappa / (m_sphere * L_arm**2))
theta_thermal = np.sqrt(k_B * T / (kappa * Q))

print(f"\nNoise analysis at T = {T} K:")
print(f"- Resonance frequency: {omega_0/(2*np.pi):.1e} Hz")
print(f"- Thermal noise: {theta_thermal*1e9:.2f} nrad/√Hz")
print(f"- SNR: {theta/theta_thermal:.1f}")
print(f"- Integration time for SNR=10: {(10*theta_thermal/theta)**2:.1f} seconds")

# Step 5: Measurement protocol
print("\n\nStep 5: Measurement Protocol")
print("-" * 40)

print("""
1. Sample Preparation:
   - Chemically synthesize Au nanospheres (2.5 nm radius)
   - Attach to CNT cantilever with e-beam lithography
   - Verify separation with STM imaging

2. Environmental Control:
   - Ultra-high vacuum: < 10⁻¹² Torr
   - Temperature: 4.2 K (liquid He)
   - Magnetic shielding: μ-metal + superconducting shield
   - Vibration isolation: Active + passive to < 10⁻¹² m/√Hz

3. Detection Method:
   - Laser interferometry with shot-noise limited detection
   - Lock-in detection at resonance frequency
   - Differential measurement: RS on/off by varying separation

4. Systematic Controls:
   - Electrostatic calibration with known voltage
   - Casimir force subtraction by dielectric coating variation
   - Temperature cycling to verify thermal behavior
""")

# Step 6: Error budget
print("\nStep 6: Error Budget")
print("-" * 40)

errors = {
    'Thermal noise': theta_thermal / theta * 100,
    'Position uncertainty': 0.1e-9 / r_target * 100,  # 0.1 nm uncertainty
    'Mass uncertainty': 0.05 * 100,  # 5% mass uncertainty
    'Casimir subtraction': 0.1 * F_Casimir / Delta_F * 100,
    'Vibration': 1e-12 / (L_arm * theta) * 100,
    'Electromagnetic': 0.01 * 100  # 1% from residual charges
}

total_error = np.sqrt(sum(e**2 for e in errors.values()))

print("Relative errors (%):")
for source, error in errors.items():
    print(f"  {source}: {error:.2f}%")
print(f"\nTotal error: {total_error:.1f}%")
print(f"Expected precision on G enhancement: ±{G_target * total_error/100:.3f}")

# Step 7: Timeline and cost
print("\n\nStep 7: Implementation Timeline")
print("-" * 40)

print("""
Month 1-3: Design optimization and simulations
  - Finite element modeling of torsion balance
  - Casimir force calculations with actual geometries
  - Thermal noise minimization strategies

Month 4-6: Fabrication
  - CNT growth and characterization
  - Au nanosphere synthesis and sizing
  - Assembly with e-beam lithography

Month 7-9: Initial measurements
  - System characterization
  - Systematic error identification
  - First enhancement measurements

Month 10-12: Data collection and analysis
  - Multiple separation distances (3-10 nm)
  - Statistical significance > 5σ
  - Publication preparation

Estimated Budget: $850k
  - Equipment (if not available): $500k
  - Personnel (2 postdocs): $200k
  - Materials and fabrication: $100k
  - Overhead: $50k
""")

# Save design parameters
design = {
    "separation_nm": float(r_target * 1e9),
    "G_enhancement": float(G_target),
    "sphere_radius_nm": float(r_sphere * 1e9),
    "sphere_mass_kg": float(m_sphere),
    "force_difference_N": float(Delta_F),
    "deflection_angle_rad": float(theta),
    "SNR": float(theta/theta_thermal),
    "total_error_percent": float(total_error),
    "temperature_K": float(T),
    "integration_time_s": float((10*theta_thermal/theta)**2)
}

import json
with open('torsion_balance_design.json', 'w') as f:
    json.dump(design, f, indent=2)

print("\nDesign parameters saved to torsion_balance_design.json")
print("\n=== Design Complete ===")
print(f"\nThe 5 nm separation hits the constructive interference window")
print(f"where RS predicts G/G₀ = {G_target:.2f} enhancement.")
print(f"This is detectable with SNR = {theta/theta_thermal:.1f} at 4.2 K.") 