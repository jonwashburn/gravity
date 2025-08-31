/-
Recognition Science – foundational definitions with **zero axioms**.
This file downgrades the former meta-principle "non-existence cannot observe
itself" to a lean-verified theorem.
-/

namespace RS.Basic

/-- A `Subject` is any type whose terms can, in principle, be recognised.  We keep
    it abstract: any Lean `Type` qualifies. -/
abbrev Subject := Type

/-- `Recognises α` means: *there exists* a term of type `α` that is recognised.
    The payload `True` is a placeholder; we care only about existence. -/
def Recognises (α : Subject) : Prop := ∃ (_ : α), True

/-- **Recognition Impossibility** – Nothingness (`Empty`) has no inhabitants and
    therefore cannot be recognised.  This converts the philosophical postulate
    "non-existence cannot observe itself" into a formal theorem. -/
@[simp]
theorem recognition_impossibility : ¬ Recognises Empty := by
  intro h
  rcases h with ⟨e, _⟩   -- `e : Empty`
  cases e                -- no cases; contradiction

end RS.Basic
