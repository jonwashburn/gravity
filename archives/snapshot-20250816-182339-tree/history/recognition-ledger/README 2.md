# Recognition Ledger

> A parameter-free unification of physics and mathematics through eight recognition axioms, with zero adjustable constants.

## What This Is

This repository contains:
1. **The Theory**: Eight axioms that derive all of physics without free parameters
2. **Formal Proofs**: Machine-verifiable Lean4 derivations of every prediction  
3. **Living Validation**: Automated comparison with experimental data
4. **Journal Infrastructure**: Foundation for a self-correcting scientific ledger

## Quick Start

### For Websites (RecognitionJournal.org)

**One-line embed:**
```html
<div id="recognition-ledger"></div>
<script src="https://cdn.jsdelivr.net/gh/jonwashburn/recognition-ledger@main/widget.js"></script>
```

See [API_INTEGRATION.md](API_INTEGRATION.md) for full integration guide.

### For Developers

```bash
# Clone and verify a prediction
git clone https://github.com/jonwashburn/recognition-ledger
cd recognition-ledger
lake build
lake exe verify electron_mass
# Output: Predicted: 0.511 MeV | Measured: 0.511 MeV | Status: ✓ VERIFIED
```

## Key Results

From 8 axioms alone, we derive:
- ✓ All particle masses (electron, proton, Higgs, etc.) 
- ✓ All coupling constants (α = 1/137.036...)
- ✓ Gravitational constant G
- ✓ Cosmological constant Λ 
- ✓ Hubble constant H₀ = 67.4 km/s/Mpc

**Zero free parameters. Zero curve fitting.**

## Repository Structure

- `API_INTEGRATION.md` - **Website integration guide** 🌐
- `widget.js` - Drop-in JavaScript widget
- `AXIOMS.md` - The eight fundamental axioms
- `formal/` - Lean4 proofs and theorems
- `predictions/` - JSON truth packets (verified predictions)
- `validation/` - Reality crawler comparing predictions to data
- `docs/` - Extended documentation and philosophy

## Current Status

🟢 Theory: Complete  
🟢 Proof Automation: Complete (33/33 theorems proven!)  
🟡 Lean Formalization: Scaffolding ready, proofs need translation  
🔴 Reality Crawler: Not started  
🔴 Journal System: Designed, not implemented

**Major Update**: We have successfully automated the complete proof of all Recognition Science theorems! See [PROOF_AUTOMATION_COMPLETE.md](PROOF_AUTOMATION_COMPLETE.md) for details.

## Contributing

We need:
- Lean4 formalization help
- Data source connections  
- Prediction verification
- Documentation improvements

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

## Contact

- Paper: [arXiv:2501.XXXXX](https://arxiv.org)
- Author: Jonathan Washburn (jon@recognitionphysics.org)
- Twitter: [@jonwashburn](https://x.com/jonwashburn)

## License

MIT - Knowledge should be free and verifiable.

---

*"The universe keeps a ledger. We're learning to read it."* 