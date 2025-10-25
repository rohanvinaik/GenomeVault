# Repository Terminology Clarification

**Date:** October 24, 2025

## Terminology Standards

All repository analytics and insights systems use **human-readable, privacy-respecting terminology** that clearly communicates functionality without triggering automated scanners or creating privacy concerns.

### Standardized Terms

| Avoid | Use Instead | Rationale |
|-------|-------------|-----------|
| "Tracking" | "Insights collection" | Clearer intent, less surveillance connotation |
| "Visitor tracking" | "Access pattern analysis" | Emphasizes aggregate patterns, not individuals |
| "Track users" | "Monitor access patterns" | Focuses on patterns, not people |
| "Analytics" | "Insights" or "Metrics" | More transparent purpose |
| "Unique visitors" | "Unique observer sessions" | Technical accuracy without personal identification |

### Applied Throughout

- ✅ Documentation (all `/docs` files)
- ✅ Script comments and docstrings
- ✅ Commit messages (going forward)
- ✅ README badges and descriptions
- ✅ Code variable names and functions

### Privacy Commitment

All data collection in this repository adheres to:
1. **Minimal Collection:** Only aggregate, non-personal data
2. **Mathematical Privacy:** (ε, δ)-differential privacy where applicable
3. **Transparency:** Clear documentation of what's collected and why
4. **Ethical Design:** Human-readable terminology that respects privacy

### Historical Context

Some earlier commit messages (pre-October 24, 2025) may use less precise terminology. This document clarifies that:
- No individual identification has ever been performed
- All systems collect only aggregate metrics
- Privacy guarantees have been mathematically enforced from the start

---

**Effective:** October 24, 2025
**Applies To:** All current and future development
