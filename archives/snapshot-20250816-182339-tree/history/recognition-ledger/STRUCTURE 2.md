# Document Structure Philosophy

<!-- 
This meta-document explains our documentation approach.
It should rarely change - only when fundamentally rethinking structure.
-->

## Core Principles

### 1. Single Source of Truth
Each concept lives in exactly ONE authoritative location:
- **Axioms**: `AXIOMS.md` (human) + `formal/axioms.lean` (machine)
- **Roadmap**: `ROADMAP.md` only
- **Philosophy**: `docs/PHILOSOPHY.md` only

No duplication. No conflicts.

### 2. Minimal File Count
Resist creating new documents. Before adding a file, ask:
- Can this live in an existing document?
- Will this be actively maintained?
- Does it serve a distinct audience?

Current structure serves:
- **Developers**: README, ROADMAP, formal/
- **Scientists**: AXIOMS, predictions/
- **Philosophers**: docs/PHILOSOPHY
- **Contributors**: docs/CONTRIBUTING

### 3. Machine-First, Human-Friendly
- Formal proofs are the source of truth
- Human documents explain and motivate
- When they conflict, the machine is right
- But both should be understandable

### 4. Living Documents
These aren't archives - they're active:
- Predictions update with measurements
- Roadmap tracks actual progress  
- Axioms evolve if refuted
- Even this structure can change

## File Purposes

### Root Level (High Traffic)
- `README.md` - Entry point, elevator pitch
- `AXIOMS.md` - The 8 foundations (critical)
- `ROADMAP.md` - From paper to journal system

### formal/ (Proof Engine)
- `axioms.lean` - Mathematical foundations
- `theorems.lean` - Derived results
- `predictions.lean` - Numerical outputs
- `lakefile.lean` - Build configuration

### predictions/ (Truth Packets)
- `particles.json` - Mass predictions
- `constants.json` - α, G, Λ, etc.
- `cosmology.json` - H₀, dark energy
- `manifest.json` - Registry of all packets

### validation/ (Reality Crawler)
- `crawler.py` - Data fetching engine
- `sources.yaml` - API configurations
- `status.db` - Verification database

### docs/ (Extended Material)
- `PHILOSOPHY.md` - Deeper implications
- `CONTRIBUTING.md` - How to help
- `JOURNAL_SPEC.md` - Full journal design
- `API.md` - Technical interfaces

## Evolution Strategy

### When to Add Files
✓ New fundamental category (e.g., "experiments/")
✓ Distinct technical audience
✓ Machine-generated content
✓ Legal requirements

### When NOT to Add Files  
✗ Temporary notes
✗ Meeting minutes
✗ Personal thoughts
✗ Redundant explanations

### Pruning Protocol
Every quarter, review:
- Which files had zero updates?
- What can be merged?
- What's no longer true?

Delete aggressively. Git preserves history.

## Cross-References

Documents should reference each other sparingly:
- Forward references: Use relative paths
- Backward references: Avoid (creates cycles)
- External references: Use permanent URLs

Example:
```markdown
See [AXIOMS.md](AXIOMS.md) for foundations.
Details in [formal/axioms.lean](formal/axioms.lean).
```

## Version Control Philosophy

### Commit Messages
Format: `type: component: description`

Types:
- `axiom:` - Changes to fundamental axioms
- `proof:` - New or modified proofs
- `predict:` - New predictions
- `verify:` - Validation results
- `doc:` - Documentation only
- `refactor:` - Structure changes

Examples:
```
axiom: A8: clarify golden ratio uniqueness
proof: electron: complete mass derivation  
verify: muon: confirmed to 6 decimal places
```

### Branching
- `main` - Current canonical axioms
- `develop` - Integration branch
- `proof/*` - Formal verification work
- `exp/*` - Experimental axiom changes

### Tags
- `v1.0` - First complete axiom set
- `verified-100` - 100 predictions confirmed
- `refutation-1` - First failed prediction

## Maintenance Schedule

**Daily**:
- Reality crawler runs
- Prediction status updates

**Weekly**:
- Proof checking
- New predictions added

**Monthly**:
- Document review
- Structure optimization

**Quarterly**:
- Full audit
- Pruning pass
- Community retrospective

---

## Remember

This structure serves the idea, not the other way around. When Recognition Science evolves, so should these documents. The moment this structure hinders rather than helps, change it.

The goal: A living system where truth computes itself.

---

*Last structural review: January 2025*
*Next scheduled review: April 2025* 