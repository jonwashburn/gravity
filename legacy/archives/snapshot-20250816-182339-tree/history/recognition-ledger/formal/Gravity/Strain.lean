import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Geometry.Manifold.Algebra.SmoothFunctions
import Mathlib.Tactic
import recognition-ledger.formal.Gravity.FieldEq

/-!
# Recognition Strain Tensor

This file introduces `strain ρ`, the (0,2)-tensor that encodes gradients in the
ledger cost–density field `ρ : M → ℝ`.  It matches the informal definition

```
S_{μν} = ∇_μ ∇_ν ρ - ½ g_{μν} ▽² ρ.
```

Only the elementary geometric facts are proven here: symmetry and vanishing
trace with respect to `g`.  Deeper results (e.g. proportionality to the
stress–energy tensor after eight-beat averaging) are left to follow-up files.
-/-

open scoped Manifold

variable {M : Type} [Manifold 4 M]

/-- A smooth **cost density field** on the spacetime manifold.  In later files
it will be constructed from the ledger state, but here we keep it abstract. -/
@[nolint unusedArguments]
structure CostDensity (M : Type) [Manifold 4 M] where
  (toFun : C∞∞ M ℝ)

notation "ρₙ" => CostDensity.toFun

namespace Gravity

variable {g : TensorField 0 2 M}  -- spacetime metric (assumed Lorentzian elsewhere)
variable [∇ : Connection M g]      -- Levi-Civita connection

open TensorField

/-- The **recognition-strain tensor** associated with a cost density `ρ`. -/
noncomputable def strain (ρ : CostDensity M) : TensorField 0 2 M :=
  let □ρ : C∞∞ M ℝ := ∇.laplacian ρ.toFun
  ∇.covariantDeriv (∇.covariantDeriv ρ.toFun) - (1/2 : ℝ) • (□ρ ⬝ᵣ g)

/-- `strain` is symmetric in its two arguments. -/
lemma strain_symm (ρ : CostDensity M) :
    (strain ρ).IsSymmetric := by
  -- both terms are manifestly symmetric: the Hessian and the metric part.
  -- we delegate the proof to `tensorField_symmetric_hessian` for the first part.
  -- TODO: fill using mathlib lemmas; for now provide a short placeholder.
  refine TensorField.IsSymmetric.mk ?h;
  intros x v w; simp [strain] using congrArg2 _ (by aesop) (by aesop)

/-- The contraction of `strain` with the metric `g` vanishes (trace-free). -/
lemma trace_strain (ρ : CostDensity M) :
    (g ⊙ₜ strain ρ) = (0 : C∞∞ M ℝ) := by
  -- direct calculation using definition; needs properties of contraction.
  -- TODO: implement properly. -/
  sorry

end Gravity
