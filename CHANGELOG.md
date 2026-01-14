# CHANGELOG: Holonomy Crusher v1 → v2

## Summary

The v2 update addresses critical issues identified in external review. The core insight remains valid—the implementation is now more honest about its guarantees and more robust in practice.

---

## Critical Changes

### 1. Added Backtracking (CRITICAL)

**Problem (v1):** When all top-k candidates exceeded the holonomy threshold, the system silently selected the "least bad" option. This is a failure mode, not graceful degradation.

**Solution (v2):**
```python
class RecoveryMode(Enum):
    LEAST_BAD = "least_bad"      # Original behavior (not recommended)
    BACKTRACK = "backtrack"       # Step back and try again
    EXPAND_K = "expand_k"         # Widen search
    BEAM_REPAIR = "beam_repair"   # Maintain multiple hypotheses
```

Default is now `BACKTRACK`. When stuck, the system returns to a previous state and tries alternative paths.

### 2. Fixed State Evolution Claims

**Problem (v1):**
```python
candidate_state = current_state + 0.1 * encode(token_emb)  # Ad hoc!
```
This doesn't reflect actual transformer dynamics.

**Solution (v2):**
```python
# Option 1: Use actual transformer forward pass (slower, accurate)
use_actual_hidden_states: bool = True

# Option 2: Justified approximation with configurable mixing
state_mixing_alpha: float = 0.3
```

When `use_actual_hidden_states=True`, we actually run the candidate token through the model.

### 3. Added Untrained Connection Warning

**Problem (v1):** No indication that holonomy from untrained connection is meaningless.

**Solution (v2):**
```python
def _warn_if_untrained(self):
    if not self.holonomy_calc.connection.is_trained:
        warnings.warn(
            "HolonomyCrusher: Connection is untrained. Holonomy does NOT "
            "correspond to semantic contradiction..."
        )
```

### 4. Corrected Documentation Claims

**Problem (v1):** Rhetoric like "geometrically impossible" overstates guarantees.

**Solution (v2):** All docstrings now use accurate language:
- ❌ "geometrically impossible"
- ✅ "probabilistically suppressed under bounded local exploration"

### 5. Improved Training Data

**Problem (v1):** ~13 training pairs, limited categories.

**Solution (v2):** ~45 training pairs across 5 categories:
- Classical logic (modus ponens, modus tollens, syllogisms)
- Mathematical reasoning
- Causal reasoning
- Self-reference
- Narrative coherence

### 6. Added Validation & Early Stopping

**Problem (v1):** No validation set, no early stopping.

**Solution (v2):**
- Train/val split (default 80/20)
- Learning rate scheduling based on validation margin
- Early stopping when margin stops improving
- Best model checkpoint restoration

---

## Configuration Changes

### v1 Defaults
```python
crush_lambda: float = 100.0
holonomy_epsilon: float = 0.05
```

### v2 Defaults
```python
crush_lambda: float = 50.0       # Less aggressive
holonomy_epsilon: float = 0.1    # More permissive
recovery_mode: RecoveryMode = RecoveryMode.BACKTRACK  # NEW
use_actual_hidden_states: bool = True                  # NEW
```

---

## What Stayed the Same

The **core mechanism** is unchanged:
```
P(token) ∝ softmax(logits) × exp(-λ · max(0, ΔHol - ε))
```

The **architecture** is unchanged:
- Lie algebra generators
- Connection network
- Parallel transport
- Holonomy calculation

The **insight** is unchanged:
- High holonomy increase → probability suppressed
- This is constraint projection, not guidance

---

## Honest Assessment

### What This System Can Do
- Suppress tokens that increase holonomy (after training)
- Provide a hard constraint during decoding
- Recover from dead ends via backtracking
- Measure path-dependent inconsistency

### What This System Cannot Do
- Guarantee perfect consistency (only top-k is evaluated)
- Work without training (untrained = random crushing)
- Replace careful reasoning (it's a filter, not a reasoner)
- Eliminate all contradictions (recovery modes are heuristics)

### What's Still Needed
1. Empirical validation at scale
2. Better state evolution (not just approximation)
3. Beam repair implementation
4. Integration with actual theorem provers
5. Formal analysis of guarantee bounds

---

## Migration Guide

### Minimal Migration
```python
# v1
from holonomy_crusher import HolonomyCrusher, CrusherConfig

# v2 - just change import
from holonomy_crusher_v2 import HolonomyCrusher, CrusherConfig
```

### Recommended Migration
```python
from holonomy_crusher_v2 import (
    HolonomyCrusher, 
    CrusherConfig, 
    CrushMode, 
    RecoveryMode
)

config = CrusherConfig(
    d_model=model.config.hidden_size,
    crush_lambda=50.0,              # Reduced
    holonomy_epsilon=0.1,           # Increased
    crush_mode=CrushMode.HARD,
    recovery_mode=RecoveryMode.BACKTRACK,  # NEW
    use_actual_hidden_states=True,          # NEW
    warn_on_fallback=True,
)

crusher = HolonomyCrusher(config)

# IMPORTANT: Train before use!
# Untrained crusher crushes randomly
```

---

## Files

| File | Status | Notes |
|------|--------|-------|
| `holonomy_crusher.py` | Keep | Original v1 for reference |
| `holonomy_crusher_v2.py` | **Use this** | Updated with fixes |
| `train_crusher.py` | Keep | Original training |
| `train_crusher_v2.py` | **Use this** | Improved training |
| `lht.py` | Keep | No changes needed |
| `integration.py` | Update | Use v2 crusher |
| `unified_engine.py` | Update | Use v2 crusher |

---

## Credit

Changes based on external critique that correctly identified:
1. Overclaimed guarantees
2. Missing recovery mechanisms
3. Ad hoc state evolution
4. Insufficient training data
5. No validation methodology

The critique was a gift. The system is now more honest and more robust.

🟥🟨🟥
