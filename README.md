# Lie-Holonomy Transformer (LHT)

A PyTorch implementation of the gauge-theoretic reasoning architecture from "Beyond Holonomy: Lie-Algebraic Symbol Emergence and the Homotopy Type Structure of Neural Reasoning."

## Core Ideas

This architecture treats **reasoning as geometry**:

| Concept | Mathematical Structure | Implementation |
|---------|----------------------|----------------|
| Propositions | Manifold M | Embedding space |
| Inference | Parallel transport | Gauge-covariant attention |
| Consistency | Holonomy = Identity | Holonomy loss |
| Symbols | Lie algebra generators | Generator network |
| Proof equivalence | Homotopy | Layer depth |

## Architecture Overview

```
Input tokens
     │
     ▼
┌─────────────────────────────────────┐
│  Token Embedding (Proposition M)    │
│  + Position Embedding               │
│  + Fiber Initialization (gauge)     │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  LHT Layer (× n_layers)             │
│  ┌─────────────────────────────┐    │
│  │ Connection Network A(x)     │    │  ← Learns gauge connection
│  │ Parallel Transport Γ_{j→i}  │    │  ← Transports fiber elements
│  │ Gauge-Covariant Attention   │    │  ← Modified self-attention
│  │ Lie Algebra Generator       │    │  ← Generates inference ops
│  │ Generator Application       │    │  ← Applies exp(X) to fiber
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Output: logits + geometric losses  │
└─────────────────────────────────────┘
```

## Key Components

### 1. Connection Network
Learns the gauge connection ω that defines how to parallel transport inferential states:
```python
A_μ(x) ∈ gl(k,ℝ)  # Lie algebra valued 1-form
```

### 2. Parallel Transport
Computes transport operators between positions:
```python
Γ_{j→i} = exp(-A_μ(x_j)(x_i - x_j)^μ)
```

### 3. Gauge-Covariant Attention
Standard attention with parallel transport of values:
```python
# Standard:  Attn(Q,K,V)_i = Σ_j α_ij V_j
# Gauge:     GaugeAttn_i   = Σ_j α_ij Γ_{j→i}(V_j)
```

### 4. Holonomy Loss
Enforces reasoning consistency by requiring closed loops to return to identity:
```python
L_hol = E[||Hol_γ - I||²_F]
```

### 5. Curvature Regularization
Encourages flat reasoning spaces where order doesn't matter:
```python
L_curv = E[||F(x)||²_F]  where F = dω + ω∧ω
```

## Installation

```bash
pip install torch
```

## Usage

### Basic
```python
from lht import LieHolonomyTransformer, LHTConfig

# Create model
config = LHTConfig(
    vocab_size=32000,
    d_model=512,
    d_fiber=64,
    n_heads=8,
    n_layers=6,
    lie_algebra_rank=8,
)
model = LieHolonomyTransformer(config)

# Forward pass
output = model(
    input_ids=tokens,
    labels=labels,
    return_geometric_losses=True
)

# Get losses
lm_loss = output['lm_loss']
holonomy_loss = output['holonomy_loss']
curvature_loss = output['curvature_loss']
total_loss = model.get_total_loss(output)
```

### Training with Geometric Loss Annealing
```python
from lht import LHTTrainer

trainer = LHTTrainer(model, optimizer, config)

for batch in dataloader:
    metrics = trainer.train_step(batch)
    # Early training: high curvature loss → flat representations
    # Mid training: high holonomy loss → consistency
    # Late training: high waypoint loss → discrete structure
```

### Waypoint Detection
```python
from lht import WaypointDetector

detector = WaypointDetector(config, n_waypoints=32)
waypoint_ids, stability = detector(representations)
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `d_model` | Proposition manifold dimension | 512 |
| `d_fiber` | Fiber (gauge) dimension | 64 |
| `lie_algebra_rank` | k for GL(k,ℝ) structure group | 8 |
| `lambda_holonomy` | Weight for holonomy loss | 0.1 |
| `lambda_curvature` | Weight for curvature loss | 0.01 |
| `lambda_waypoint` | Weight for waypoint stability | 0.05 |

## Theoretical Predictions

The framework makes testable predictions:

1. **Chain-of-thought benefit correlates with curvature** - High-curvature domains (causal reasoning) benefit more from CoT than low-curvature domains (arithmetic)

2. **Waypoints emerge spontaneously** - Training with holonomy loss should cause discrete symbol-like structures to form at flat loci

3. **Holonomy predicts errors** - Incorrect reasoning paths should have higher holonomy magnitude

4. **Compositional generalization improves** - Holonomy constraints force consistent composition

## File Structure

```
lie_holonomy_transformer/
├── lht.py           # Core implementation
├── train.py         # Training script  
├── README.md        # This file
└── experiments/     # Benchmark code (TODO)
```

## References

- "Beyond Holonomy: Lie-Algebraic Symbol Emergence..." (the paper)
- Cohen et al. (2019). Gauge Equivariant Convolutional Networks
- Weiler & Cesa (2019). General E(2)-Equivariant Steerable CNNs
- The Univalent Foundations Program (2013). Homotopy Type Theory

## License

MIT
