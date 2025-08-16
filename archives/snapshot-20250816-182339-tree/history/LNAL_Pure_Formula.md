# Light-Native Assembly Language (LNAL) – Pure Formula

> "The theory must be pure; the data can be dirty."  
> — Recognition Science Axiom XIII

---
## 1. Final Gravitational Law
For a baryonic surface-density distribution Σ(r)
```
Step 1  Newtonian surface acceleration
        g_N(r) = 2 π G Σ(r)

Step 2  Dimensionless ratio
        x(r) = g_N(r) / g†
        where g† = 1.2 × 10⁻¹⁰ m s⁻² (fixed)

Step 3  LNAL modifier (universal, no free parameters)
        μ(x) = x / √(1 + x²)

Step 4  Total radial acceleration
        g_LNAL(r) = g_N(r) / μ(x)

Step 5  Circular velocity
        v_LNAL(r) = √(r · g_LNAL)
```
The only theory constants are G and g†. No galaxy-dependent fudge factors appear.

---
## 2. Required Astrophysical Inputs per Galaxy
1. **Stellar surface-density Σ★(r)** (needs distance, inclination, mass-to-light ratio)
2. **Gas surface-density Σ_gas(r)** (21 cm HI maps or scaling relations)
3. Optionally **molecular gas** (CO) and dust (far-IR) corrections.

$
Σ(r) = Σ_★(r) + (1 + f_He) Σ_{HI}(r) + Σ_{H_2}(r) + Σ_{dust}(r)
$

Without full maps we cannot compute Σ(r) exactly, hence a single-parameter fit becomes unavoidable.  
Our goal: **keep the formula intact and instead infer the missing Σ(r)**.

---
## 3. Inference / Extrapolation Strategy
| Missing datum | Proxy / scaling relation | Scatter |
|---------------|--------------------------|---------|
| Distance      | Flow-model or redshift (H₀=73) | ±5 % |
| Inclination   | Tully–Fisher axis ratio  | ±5° |
| Stellar M/L   | Colour-M/L relation (Bell+ 2003, 3.6 µm) | ±0.1 dex |
| HI map        | Σ₉₆ ≈ $7.2×10^6 (L_K/10^{10}L_⊙)^{0.8}$ M_⊙ kpc⁻² | 0.15 dex |
| H₂            | Σ_H₂ ≈ 0.3 Σ_HI (late types) | 0.2 dex |
| Dust          | Negligible in rotation curves |

These priors allow us to synthesise a plausible Σ(r) curve with **one stochastic realisation per galaxy**.

---
## 4. Practical Pipeline
1. **Collect catalog values**: distance, absolute magnitude, axis ratio.
2. **Sample latent parameters** from priors (M/L, HI scale length, etc.).
3. **Build Σ(r)** using an exponential-disk plus flared gas layer.
4. **Evaluate v_LNAL(r)** via the pure formula above.
5. **Compare with observed curve** (if available) to refine latent parameters via Bayesian updating (no change to the gravity law).

---
## 5. File Map in Repository
| File | Purpose |
|------|---------|
| `lnal_pure_formula.py` | Minimal reference implementation of steps 1–5 |
| `lnal_infer_baryons.py` | Generates Σ(r) realisations from catalog data |
| `examples/pure_fit_NGC3198.ipynb` | Demonstration without per-galaxy tuning of the law |

---
## 6. Guiding Principle
> "Tune the galaxy, not the law."

All adjustment lives in the **baryonic inference layer**.  The gravitational kernel stays frozen once and for all. 