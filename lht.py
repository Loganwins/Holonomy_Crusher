"""
Lie-Holonomy Transformer (LHT)
==============================
Implementation based on "Beyond Holonomy: Lie-Algebraic Symbol Emergence 
and the Homotopy Type Structure of Neural Reasoning"

Core ideas:
- Symbols are Lie algebra generators
- Reasoning is parallel transport in a principal bundle
- Consistency = holonomy-freedom
- Discrete structure emerges from flat loci

Author: Built collaboratively
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class LHTConfig:
    """Configuration for Lie-Holonomy Transformer."""
    vocab_size: int = 32000
    d_model: int = 512          # Embedding dimension (proposition manifold)
    d_fiber: int = 64           # Fiber dimension (gauge degrees of freedom)
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.1
    
    # Lie algebra structure
    lie_algebra_dim: int = 64   # Dimension of Lie algebra (k×k matrices, k=8)
    lie_algebra_rank: int = 8   # k for gl(k,R) structure group
    
    # Geometric regularization weights
    lambda_holonomy: float = 0.1
    lambda_curvature: float = 0.01
    lambda_waypoint: float = 0.05
    lambda_gauge: float = 0.01


# =============================================================================
# CORE GEOMETRIC COMPONENTS
# =============================================================================

class ConnectionNetwork(nn.Module):
    """
    Learns the connection 1-form ω on the principal bundle.
    
    The connection tells us how to parallel transport inferential states
    along paths in the proposition manifold.
    
    Output: A_μ(x) ∈ gl(k,R) for each position x
    """
    
    def __init__(self, config: LHTConfig):
        super().__init__()
        self.config = config
        k = config.lie_algebra_rank
        
        # Network that outputs Lie algebra elements
        self.connection_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.LayerNorm(config.d_ff),
            nn.Linear(config.d_ff, config.d_model * k * k),
        )
        
        # For each dimension of tangent space, we need a k×k matrix
        self.k = k
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Positions in proposition manifold [batch, seq_len, d_model]
            
        Returns:
            A: Connection coefficients [batch, seq_len, d_model, k, k]
               A_μ(x) gives the Lie algebra element for direction μ at position x
        """
        batch, seq_len, d_model = x.shape
        
        # Get raw connection values
        A_flat = self.connection_mlp(x)  # [batch, seq_len, d_model * k * k]
        
        # Reshape to [batch, seq_len, d_model, k, k]
        A = A_flat.view(batch, seq_len, d_model, self.k, self.k)
        
        # Optional: Project to so(k) for orthogonal structure (antisymmetric)
        # A = 0.5 * (A - A.transpose(-1, -2))
        
        return A


class ParallelTransport(nn.Module):
    """
    Computes parallel transport operators Γ_{j→i} between positions.
    
    Γ_{j→i} = exp(-A_μ(x_j)(x_i - x_j)^μ)
    
    This tells us how to transport fiber elements from position j to position i.
    """
    
    def __init__(self, config: LHTConfig):
        super().__init__()
        self.config = config
        self.k = config.lie_algebra_rank
        
    def forward(
        self, 
        x: torch.Tensor, 
        A: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Positions [batch, seq_len, d_model]
            A: Connection [batch, seq_len, d_model, k, k]
            
        Returns:
            Gamma: Transport operators [batch, seq_len, seq_len, k, k]
                   Gamma[b, i, j] transports from j to i
        """
        batch, seq_len, d_model = x.shape
        k = self.k
        
        # Compute displacement vectors: x_i - x_j
        # [batch, seq_len, 1, d_model] - [batch, 1, seq_len, d_model]
        x_i = x.unsqueeze(2)  # [batch, seq_len, 1, d_model]
        x_j = x.unsqueeze(1)  # [batch, 1, seq_len, d_model]
        displacement = x_i - x_j  # [batch, seq_len, seq_len, d_model]
        
        # Contract A_μ with displacement to get Lie algebra element
        # A[j] has shape [d_model, k, k], displacement[i,j] has shape [d_model]
        # Result: sum over μ of A_μ(x_j) * (x_i - x_j)^μ
        
        # A: [batch, seq_len, d_model, k, k] -> [batch, 1, seq_len, d_model, k, k]
        A_expanded = A.unsqueeze(1)
        
        # displacement: [batch, seq_len, seq_len, d_model] -> [..., d_model, 1, 1]
        disp_expanded = displacement.unsqueeze(-1).unsqueeze(-1)
        
        # Contract: sum over d_model dimension
        # [batch, seq_len, seq_len, d_model, k, k] * [batch, seq_len, seq_len, d_model, 1, 1]
        lie_element = (A_expanded * disp_expanded).sum(dim=3)  # [batch, seq_len, seq_len, k, k]
        
        # Compute matrix exponential: Γ = exp(-lie_element)
        # For efficiency, use truncated Taylor series or Padé approximation
        Gamma = self._matrix_exp(-lie_element)
        
        return Gamma
    
    def _matrix_exp(self, M: torch.Tensor, terms: int = 6) -> torch.Tensor:
        """
        Approximate matrix exponential using Taylor series.
        exp(M) ≈ I + M + M²/2! + M³/3! + ...
        """
        batch, s1, s2, k, _ = M.shape
        
        # Identity matrix
        I = torch.eye(k, device=M.device, dtype=M.dtype)
        I = I.view(1, 1, 1, k, k).expand(batch, s1, s2, k, k)
        
        result = I.clone()
        M_power = I.clone()
        
        for n in range(1, terms + 1):
            M_power = torch.matmul(M_power, M) / n
            result = result + M_power
            
        return result


class CurvatureComputation(nn.Module):
    """
    Computes the curvature 2-form F = dω + ω ∧ ω.
    
    Curvature measures path-dependence of parallel transport.
    F = 0 means reasoning order doesn't matter (flat).
    """
    
    def __init__(self, config: LHTConfig):
        super().__init__()
        self.config = config
        
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Compute curvature at each point.
        
        F_μν = ∂_μ A_ν - ∂_ν A_μ + [A_μ, A_ν]
        
        Args:
            x: Positions [batch, seq_len, d_model]  
            A: Connection [batch, seq_len, d_model, k, k]
            
        Returns:
            F: Curvature magnitude at each position [batch, seq_len]
        """
        batch, seq_len, d_model, k, _ = A.shape
        
        # Approximate derivatives using finite differences
        # For simplicity, compute curvature via the commutator term [A_μ, A_ν]
        
        # Sample random directions μ, ν
        # Commutator: [A_μ, A_ν] = A_μ A_ν - A_ν A_μ
        
        # Average curvature over all pairs of directions
        F_magnitude = torch.zeros(batch, seq_len, device=x.device)
        
        n_samples = min(16, d_model)  # Sample pairs for efficiency
        indices = torch.randperm(d_model)[:n_samples]
        
        for i in range(0, n_samples - 1, 2):
            mu, nu = indices[i].item(), indices[i + 1].item()
            A_mu = A[:, :, mu, :, :]  # [batch, seq_len, k, k]
            A_nu = A[:, :, nu, :, :]
            
            # Commutator
            commutator = torch.matmul(A_mu, A_nu) - torch.matmul(A_nu, A_mu)
            
            # Frobenius norm
            F_magnitude += torch.norm(commutator, dim=(-2, -1))
            
        F_magnitude /= (n_samples // 2)
        
        return F_magnitude


class HolonomyComputation(nn.Module):
    """
    Computes holonomy around closed loops.
    
    Hol_γ = P exp(-∮_γ ω)
    
    For consistent reasoning, Hol_γ should equal identity for contractible loops.
    """
    
    def __init__(self, config: LHTConfig):
        super().__init__()
        self.config = config
        self.k = config.lie_algebra_rank
        
    def forward(
        self, 
        Gamma: torch.Tensor,
        loop_indices: List[List[int]]
    ) -> torch.Tensor:
        """
        Compute holonomy around specified loops.
        
        Args:
            Gamma: Transport operators [batch, seq_len, seq_len, k, k]
            loop_indices: List of loops, each loop is a list of position indices
                          forming a closed path [i_0, i_1, ..., i_n, i_0]
                          
        Returns:
            holonomy: Holonomy matrices for each loop [batch, n_loops, k, k]
        """
        batch = Gamma.shape[0]
        n_loops = len(loop_indices)
        
        holonomies = []
        
        for loop in loop_indices:
            # Compose transport operators around the loop
            # Hol = Γ_{n→0} ∘ Γ_{n-1→n} ∘ ... ∘ Γ_{0→1}
            
            hol = torch.eye(self.k, device=Gamma.device, dtype=Gamma.dtype)
            hol = hol.unsqueeze(0).expand(batch, self.k, self.k).clone()
            
            for idx in range(len(loop) - 1):
                i, j = loop[idx + 1], loop[idx]
                # Transport from j to i
                hol = torch.matmul(Gamma[:, i, j], hol)
            
            holonomies.append(hol)
            
        return torch.stack(holonomies, dim=1)  # [batch, n_loops, k, k]


# =============================================================================
# GAUGE-COVARIANT ATTENTION
# =============================================================================

class GaugeCovariantAttention(nn.Module):
    """
    Attention mechanism with parallel transport.
    
    Standard attention: Attn(Q,K,V)_i = Σ_j α_ij V_j
    Gauge attention:    GaugeAttn(Q,K,V)_i = Σ_j α_ij Γ_{j→i}(V_j)
    
    Values are parallel transported before aggregation.
    """
    
    def __init__(self, config: LHTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.k = config.lie_algebra_rank
        
        # Projections for Q, K, V
        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_o = nn.Linear(config.d_model, config.d_model)
        
        # Fiber projections
        self.fiber_proj = nn.Linear(config.d_fiber, config.d_model)
        self.fiber_out = nn.Linear(config.d_model, config.d_fiber)
        
        # Connection network for this attention layer
        self.connection = ConnectionNetwork(config)
        self.transport = ParallelTransport(config)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.d_head)
        
    def forward(
        self, 
        x: torch.Tensor,
        u: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_transport: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Proposition manifold positions [batch, seq_len, d_model]
            u: Fiber coordinates [batch, seq_len, d_fiber]
            mask: Attention mask [batch, seq_len, seq_len]
            
        Returns:
            x_out: Updated positions [batch, seq_len, d_model]
            u_out: Updated fiber coordinates [batch, seq_len, d_fiber]
            Gamma: (optional) Transport operators for holonomy computation
        """
        batch, seq_len, _ = x.shape
        
        # Compute connection and transport operators
        A = self.connection(x)  # [batch, seq_len, d_model, k, k]
        Gamma = self.transport(x, A)  # [batch, seq_len, seq_len, k, k]
        
        # Standard Q, K, V projections
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Standard attention output for position update
        x_attn = torch.matmul(attn_weights, V)
        x_attn = x_attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        x_out = x + self.W_o(x_attn)
        
        # Gauge-covariant fiber update with parallel transport
        # Project fiber to transportable dimension
        u_proj = self.fiber_proj(u)  # [batch, seq_len, d_model]
        
        # Reshape for transport: need to match k dimension
        # We'll transport chunks of the projected fiber
        chunk_size = self.config.d_model // self.k
        u_chunks = u_proj.view(batch, seq_len, self.k, chunk_size)
        
        # Apply parallel transport: for each target i, transport all sources j
        # u_transported[i] = Σ_j α_ij Γ_{j→i} u[j]
        
        # Average attention weights across heads for fiber transport
        attn_avg = attn_weights.mean(dim=1)  # [batch, seq_len, seq_len]
        
        # Transport each fiber
        u_transported = torch.zeros_like(u_chunks)
        
        for i in range(seq_len):
            for j in range(seq_len):
                # Γ_{j→i}: [batch, k, k]
                # u_chunks[:, j]: [batch, k, chunk_size]
                transported = torch.matmul(
                    Gamma[:, i, j],  # [batch, k, k]
                    u_chunks[:, j]   # [batch, k, chunk_size]
                )
                u_transported[:, i] += attn_avg[:, i, j].unsqueeze(-1).unsqueeze(-1) * transported
        
        # Reshape and project back
        u_transported = u_transported.view(batch, seq_len, -1)
        u_out = u + self.fiber_out(u_transported)
        
        if return_transport:
            return x_out, u_out, Gamma
        return x_out, u_out, None


# =============================================================================
# LIE ALGEBRA GENERATOR NETWORK
# =============================================================================

class LieAlgebraGenerator(nn.Module):
    """
    Generates elements of the Lie algebra g corresponding to inference operations.
    
    Different generators correspond to different types of inference:
    - Conjunction introduction
    - Modus ponens
    - Universal instantiation
    - etc.
    
    The Lie bracket [X_i, X_j] encodes how operations interact.
    """
    
    def __init__(self, config: LHTConfig):
        super().__init__()
        self.config = config
        self.k = config.lie_algebra_rank
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(config.d_model + config.d_fiber, config.d_ff),
            nn.GELU(),
            nn.LayerNorm(config.d_ff),
            nn.Linear(config.d_ff, config.d_ff),
            nn.GELU(),
        )
        
        # Generator head: outputs parameters for distribution over g
        self.generator_mean = nn.Linear(config.d_ff, self.k * self.k)
        self.generator_logvar = nn.Linear(config.d_ff, self.k * self.k)
        
    def forward(
        self, 
        x: torch.Tensor, 
        u: torch.Tensor,
        sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Lie algebra elements from context.
        
        Args:
            x: Position in proposition manifold [batch, seq_len, d_model]
            u: Fiber coordinates [batch, seq_len, d_fiber]
            sample: Whether to sample or return mean
            
        Returns:
            X: Lie algebra elements [batch, seq_len, k, k]
            kl_loss: KL divergence for VAE-style regularization
        """
        batch, seq_len, _ = x.shape
        
        # Encode context
        context = torch.cat([x, u], dim=-1)
        h = self.context_encoder(context)
        
        # Get distribution parameters
        mean = self.generator_mean(h).view(batch, seq_len, self.k, self.k)
        logvar = self.generator_logvar(h).view(batch, seq_len, self.k, self.k)
        
        if sample and self.training:
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            X = mean + eps * std
        else:
            X = mean
            
        # KL divergence with standard normal
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        return X, kl_loss
    
    def apply_generator(self, u: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Apply Lie algebra element to fiber coordinates.
        
        u' = exp(X) · u
        
        Args:
            u: Fiber coordinates [batch, seq_len, d_fiber]
            X: Lie algebra elements [batch, seq_len, k, k]
            
        Returns:
            u': Transformed fiber coordinates
        """
        batch, seq_len, d_fiber = u.shape
        k = self.k
        
        # Reshape u for matrix multiplication
        chunk_size = d_fiber // k
        u_chunks = u.view(batch, seq_len, k, chunk_size)
        
        # Compute exp(X) using Taylor series
        expX = self._matrix_exp(X)
        
        # Apply transformation
        u_transformed = torch.matmul(expX, u_chunks)
        
        return u_transformed.view(batch, seq_len, d_fiber)
    
    def _matrix_exp(self, M: torch.Tensor, terms: int = 6) -> torch.Tensor:
        """Matrix exponential via Taylor series."""
        batch, seq_len, k, _ = M.shape
        
        I = torch.eye(k, device=M.device, dtype=M.dtype)
        I = I.view(1, 1, k, k).expand(batch, seq_len, k, k)
        
        result = I.clone()
        M_power = I.clone()
        
        for n in range(1, terms + 1):
            M_power = torch.matmul(M_power, M) / n
            result = result + M_power
            
        return result


# =============================================================================
# TRANSFORMER LAYER AND FULL MODEL
# =============================================================================

class LHTLayer(nn.Module):
    """
    Single layer of the Lie-Holonomy Transformer.
    
    Each layer adds a level of categorical structure (n-morphisms).
    """
    
    def __init__(self, config: LHTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Gauge-covariant attention
        self.attention = GaugeCovariantAttention(config)
        
        # Lie algebra generator
        self.generator = LieAlgebraGenerator(config)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.fiber_norm = nn.LayerNorm(config.d_fiber)
        
    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_geometric: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass through one LHT layer.
        
        Args:
            x: Positions [batch, seq_len, d_model]
            u: Fiber coordinates [batch, seq_len, d_fiber]
            mask: Attention mask
            return_geometric: Whether to return geometric info for loss
            
        Returns:
            x_out, u_out: Updated representations
            geo_info: Dict with Gamma, curvature, etc. for losses
        """
        geo_info = {}
        
        # 1. Gauge-covariant attention
        x_normed = self.norm1(x)
        x_attn, u_attn, Gamma = self.attention(
            x_normed, u, mask, 
            return_transport=return_geometric
        )
        
        if return_geometric and Gamma is not None:
            geo_info['Gamma'] = Gamma
            geo_info['A'] = self.attention.connection(x_normed)
            
        # 2. Generate inference operation
        X, kl_loss = self.generator(x_attn, u_attn)
        geo_info['generator_kl'] = kl_loss
        
        # 3. Apply generator to fiber
        u_gen = self.generator.apply_generator(u_attn, X)
        u_out = self.fiber_norm(u_gen)
        
        # 4. Feed-forward for position
        x_out = x_attn + self.ff(self.norm2(x_attn))
        
        return x_out, u_out, geo_info


class LieHolonomyTransformer(nn.Module):
    """
    The complete Lie-Holonomy Transformer.
    
    Architecture determined by the geometry:
    - Embedding layer constructs proposition manifold M
    - Gauge augmentation adds fiber coordinates
    - Layers compute parallel transport and inference operations
    - Geometric losses enforce consistency
    """
    
    def __init__(self, config: LHTConfig):
        super().__init__()
        self.config = config
        
        # Token embedding (proposition manifold)
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Fiber initialization
        self.fiber_init = nn.Parameter(torch.randn(1, 1, config.d_fiber) * 0.02)
        
        # Learned Riemannian metric (optional, for curved semantics)
        self.metric = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            LHTLayer(config, i) for i in range(config.n_layers)
        ])
        
        # Curvature computation
        self.curvature = CurvatureComputation(config)
        self.holonomy = HolonomyComputation(config)
        
        # Output head
        self.out_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embed.weight
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_geometric_losses: bool = True
    ) -> dict:
        """
        Forward pass with geometric loss computation.
        
        Args:
            input_ids: Token ids [batch, seq_len]
            attention_mask: Mask for padding
            labels: Target ids for LM loss
            return_geometric_losses: Whether to compute geometric losses
            
        Returns:
            dict with logits, loss, and geometric metrics
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Embed tokens into proposition manifold
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        
        # 2. Initialize fiber coordinates
        u = self.fiber_init.expand(batch, seq_len, -1).clone()
        
        # 3. Create causal mask
        if attention_mask is None:
            attention_mask = torch.ones(batch, seq_len, device=device)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = attention_mask.unsqueeze(1) * causal_mask.unsqueeze(0)
        
        # 4. Pass through layers, collecting geometric info
        all_geo_info = []
        all_Gammas = []
        
        for layer in self.layers:
            x, u, geo_info = layer(x, u, mask, return_geometric=return_geometric_losses)
            all_geo_info.append(geo_info)
            if 'Gamma' in geo_info:
                all_Gammas.append(geo_info['Gamma'])
        
        # 5. Output projection
        x = self.out_norm(x)
        logits = self.lm_head(x)
        
        # 6. Compute losses
        output = {'logits': logits}
        
        # Language modeling loss
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            output['lm_loss'] = lm_loss
        
        # Geometric losses
        if return_geometric_losses and len(all_Gammas) > 0:
            geo_losses = self._compute_geometric_losses(
                x, all_Gammas, all_geo_info
            )
            output.update(geo_losses)
            
        return output
    
    def _compute_geometric_losses(
        self,
        x: torch.Tensor,
        Gammas: List[torch.Tensor],
        geo_infos: List[dict]
    ) -> dict:
        """Compute holonomy, curvature, and other geometric losses."""
        
        losses = {}
        batch, seq_len, _ = x.shape
        
        # 1. Holonomy loss: Sample triangular loops
        if seq_len >= 3 and len(Gammas) > 0:
            Gamma = Gammas[-1]  # Use last layer's transport
            
            # Sample triangular loops: [i, j, k, i]
            n_loops = min(10, seq_len // 3)
            loops = []
            for _ in range(n_loops):
                indices = torch.randperm(seq_len)[:3].tolist()
                loops.append(indices + [indices[0]])  # Close the loop
            
            holonomies = self.holonomy(Gamma, loops)
            
            # Holonomy should be identity
            I = torch.eye(self.config.lie_algebra_rank, device=x.device)
            I = I.view(1, 1, *I.shape).expand(batch, n_loops, -1, -1)
            
            hol_loss = F.mse_loss(holonomies, I)
            losses['holonomy_loss'] = hol_loss * self.config.lambda_holonomy
        
        # 2. Curvature loss: Prefer flat spaces
        curvature_total = 0.0
        for info in geo_infos:
            if 'A' in info:
                curv = self.curvature(x, info['A'])
                curvature_total += curv.mean()
        
        losses['curvature_loss'] = curvature_total * self.config.lambda_curvature
        
        # 3. Generator KL loss
        kl_total = sum(info.get('generator_kl', 0.0) for info in geo_infos)
        losses['generator_kl_loss'] = kl_total * 0.01
        
        # 4. Total geometric loss
        losses['geometric_loss'] = sum(losses.values())
        
        return losses
    
    def get_total_loss(self, output: dict) -> torch.Tensor:
        """Combine all losses."""
        total = output.get('lm_loss', 0.0)
        total = total + output.get('geometric_loss', 0.0)
        return total


# =============================================================================
# WAYPOINT DETECTION
# =============================================================================

class WaypointDetector(nn.Module):
    """
    Detects emergent waypoints - stable points where certain transformations 
    leave the state invariant.
    
    Waypoints correspond to logical constants and anchor reasoning.
    """
    
    def __init__(self, config: LHTConfig, n_waypoints: int = 32):
        super().__init__()
        self.n_waypoints = n_waypoints
        
        # Learnable waypoint embeddings
        self.waypoint_embeds = nn.Parameter(
            torch.randn(n_waypoints, config.d_model) * 0.02
        )
        
        # Stability predictor
        self.stability_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find closest waypoints and stability scores.
        
        Args:
            x: Representations [batch, seq_len, d_model]
            
        Returns:
            waypoint_ids: Closest waypoint indices [batch, seq_len]
            stability: Stability scores [batch, seq_len]
        """
        # Distance to each waypoint
        # x: [batch, seq_len, d_model]
        # waypoints: [n_waypoints, d_model]
        
        dists = torch.cdist(x, self.waypoint_embeds.unsqueeze(0).expand(x.shape[0], -1, -1))
        waypoint_ids = dists.argmin(dim=-1)
        
        # Stability score
        stability = self.stability_net(x).squeeze(-1)
        
        return waypoint_ids, stability
    
    def waypoint_stability_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encourage waypoints to be attractors.
        
        Loss: -log det(I - J) where J is Jacobian of dynamics at waypoints
        
        Simplified: Encourage low gradient magnitude near waypoints.
        """
        _, stability = self.forward(x)
        
        # Points near waypoints should have high stability
        # This is a simplified version
        return -torch.mean(torch.log(stability + 1e-6))


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class LHTTrainer:
    """Training loop with geometric loss annealing."""
    
    def __init__(
        self,
        model: LieHolonomyTransformer,
        optimizer: torch.optim.Optimizer,
        config: LHTConfig
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.step = 0
        
    def get_annealed_lambdas(self) -> dict:
        """
        Anneal geometric loss weights during training.
        
        Early: High curvature loss (encourage flat representations)
        Mid: Increase holonomy loss (enforce consistency)
        Late: Increase waypoint loss (crystallize discrete structure)
        """
        progress = min(1.0, self.step / 10000)
        
        return {
            'lambda_curvature': self.config.lambda_curvature * (1.0 - 0.5 * progress),
            'lambda_holonomy': self.config.lambda_holonomy * (0.5 + 0.5 * progress),
            'lambda_waypoint': self.config.lambda_waypoint * progress,
        }
    
    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Update lambdas
        lambdas = self.get_annealed_lambdas()
        self.config.lambda_curvature = lambdas['lambda_curvature']
        self.config.lambda_holonomy = lambdas['lambda_holonomy']
        self.config.lambda_waypoint = lambdas['lambda_waypoint']
        
        # Forward pass
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch.get('labels'),
            return_geometric_losses=True
        )
        
        # Compute total loss
        loss = self.model.get_total_loss(output)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.step += 1
        
        return {
            'loss': loss.item(),
            'lm_loss': output.get('lm_loss', torch.tensor(0.0)).item(),
            'holonomy_loss': output.get('holonomy_loss', torch.tensor(0.0)).item(),
            'curvature_loss': output.get('curvature_loss', torch.tensor(0.0)).item(),
            'step': self.step,
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Quick demonstration of the LHT architecture."""
    print("=" * 60)
    print("Lie-Holonomy Transformer - Architecture Demo")
    print("=" * 60)
    
    # Create config
    config = LHTConfig(
        vocab_size=1000,
        d_model=256,
        d_fiber=32,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=128,
        lie_algebra_rank=4,
    )
    
    print(f"\nConfig:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_fiber: {config.d_fiber}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  lie_algebra_rank: {config.lie_algebra_rank}")
    
    # Create model
    model = LieHolonomyTransformer(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Create dummy input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    output = model(input_ids=input_ids, labels=labels, return_geometric_losses=True)
    
    print(f"\nOutput logits shape: {output['logits'].shape}")
    print(f"LM Loss: {output.get('lm_loss', 'N/A'):.4f}")
    print(f"Holonomy Loss: {output.get('holonomy_loss', 'N/A'):.4f}")
    print(f"Curvature Loss: {output.get('curvature_loss', 'N/A'):.4f}")
    print(f"Total Geometric Loss: {output.get('geometric_loss', 'N/A'):.4f}")
    
    # Waypoint detection demo
    print("\n" + "=" * 60)
    print("Waypoint Detection Demo")
    print("=" * 60)
    
    waypoint_detector = WaypointDetector(config, n_waypoints=16)
    
    # Get representations from model
    with torch.no_grad():
        x = model.token_embed(input_ids) + model.pos_embed(
            torch.arange(seq_len).unsqueeze(0)
        )
    
    waypoint_ids, stability = waypoint_detector(x)
    print(f"Waypoint IDs shape: {waypoint_ids.shape}")
    print(f"Stability scores shape: {stability.shape}")
    print(f"Sample waypoint assignments: {waypoint_ids[0, :8].tolist()}")
    print(f"Sample stability scores: {stability[0, :8].tolist()}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    return model, config


if __name__ == "__main__":
    demo()
