/-
Lattice Geometry for Recognition Lengths
=======================================

In a four–dimensional cubic lattice with site–occupation probability
`f_occupancy` the mean nearest–neighbour spacing grows like
`f_occupancy^{-1/4}`.  Combined with the microscopic recognition length
`λ_rec` this gives the effective recognition scale used in macroscopic
(form‐factor) formulas.
-/

import RecognitionScience.ScaleConsistency

namespace RecognitionScience

/--  Fundamental relation between microscopic and effective recognition
lengths on a sparse 4-D lattice.  *To be proven in full lattice-geometry
module* but recorded here as an axiom to unblock other proofs. -/
axiom lambda_eff_scaling :
  λ_eff.value = λ_rec.value * f_occupancy ^ (- (1 : ℝ) / 4)

end RecognitionScience
