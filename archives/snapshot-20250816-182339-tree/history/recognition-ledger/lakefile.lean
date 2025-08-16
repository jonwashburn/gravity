import Lake
open Lake DSL

package «RecognitionScience» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.8.0"

-- The library source is in the `formal/` folder so modules appear as
-- `RecognitionScience.<FileName>` rather than `RecognitionScience.formal.*`.
lean_lib «RecognitionScience» where
  srcDir := "formal"
