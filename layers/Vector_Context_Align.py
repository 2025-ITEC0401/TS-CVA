"""
Vector-Context Alignment Layer for TS-CVA

This layer aligns the vector modality (TS2Vec) and context modality (LLM)
through cross-attention mechanisms. It enables the model to leverage both
pure time series patterns and contextual information.

Architecture:
- Dual cross-attention: Vector ↔ Context bidirectional alignment
- Gated fusion: Learnable weighting between modalities
- Similarity-based retrieval: Retrieve context relevant to vector patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VectorContextAlignment(nn.Module):
    """
    Bidirectional alignment between Vector and Context modalities

    Args:
        d_vector (int): Dimension of vector modality (TS2Vec output)
        d_context (int): Dimension of context modality (LLM output)
        d_model (int): Hidden dimension for alignment
        n_heads (int): Number of attention heads
        dropout (float): Dropout rate
        fusion_mode (str): 'concat', 'gated', or 'weighted'
    """
    def __init__(
        self,
        d_vector,
        d_context,
        d_model=256,
        n_heads=8,
        dropout=0.1,
        fusion_mode='gated'
    ):
        super().__init__()

        self.d_vector = d_vector
        self.d_context = d_context
        self.d_model = d_model
        self.fusion_mode = fusion_mode

        # Project vector and context to common dimension
        self.vector_proj = nn.Linear(d_vector, d_model)
        self.context_proj = nn.Linear(d_context, d_model)

        # Cross-attention: Vector queries, Context key-values
        self.v2c_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention: Context queries, Vector key-values
        self.c2v_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm_v2c = nn.LayerNorm(d_model)
        self.norm_c2v = nn.LayerNorm(d_model)

        # Fusion layers
        if fusion_mode == 'gated':
            # Gated fusion with learnable gate
            self.gate_v = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            self.gate_c = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
        elif fusion_mode == 'weighted':
            # Learnable weighted fusion
            self.weight_v = nn.Parameter(torch.tensor(0.5))
            self.weight_c = nn.Parameter(torch.tensor(0.5))
        elif fusion_mode == 'concat':
            # Concatenation followed by projection
            self.fusion_proj = nn.Linear(d_model * 2, d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vector_repr, context_repr, return_attention=False):
        """
        Align vector and context modalities

        Args:
            vector_repr: [B, L_v, d_vector] - Vector modality from TS2Vec
            context_repr: [B, L_c, d_context] - Context modality from LLM
            return_attention: Whether to return attention weights
        Returns:
            aligned: [B, L_v, d_model] - Aligned representation
            attn_weights: Optional attention weights
        """
        # Project to common space
        vector = self.vector_proj(vector_repr)  # [B, L_v, d_model]
        context = self.context_proj(context_repr)  # [B, L_c, d_model]

        # Vector-to-Context attention: Enhance vector with context
        v2c_out, v2c_attn = self.v2c_attention(
            query=vector,
            key=context,
            value=context
        )  # [B, L_v, d_model]

        v2c_out = self.norm_v2c(vector + self.dropout(v2c_out))

        # Context-to-Vector attention: Enhance context with vector
        c2v_out, c2v_attn = self.c2v_attention(
            query=context,
            key=vector,
            value=vector
        )  # [B, L_c, d_model]

        c2v_out = self.norm_c2v(context + self.dropout(c2v_out))

        # Pool context-enhanced representation to match vector length
        if c2v_out.size(1) != v2c_out.size(1):
            # Adaptive pooling to match vector length
            c2v_out = c2v_out.transpose(1, 2)  # [B, d_model, L_c]
            c2v_out = F.adaptive_avg_pool1d(c2v_out, v2c_out.size(1))
            c2v_out = c2v_out.transpose(1, 2)  # [B, L_v, d_model]

        # Fusion
        if self.fusion_mode == 'gated':
            # Gated fusion
            concat = torch.cat([v2c_out, c2v_out], dim=-1)
            gate_v = self.gate_v(concat)
            gate_c = self.gate_c(concat)
            fused = gate_v * v2c_out + gate_c * c2v_out
        elif self.fusion_mode == 'weighted':
            # Weighted fusion
            w_v = torch.sigmoid(self.weight_v)
            w_c = torch.sigmoid(self.weight_c)
            fused = w_v * v2c_out + w_c * c2v_out
        elif self.fusion_mode == 'concat':
            # Concatenation
            concat = torch.cat([v2c_out, c2v_out], dim=-1)
            fused = self.fusion_proj(concat)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        # Output projection
        output = self.output_proj(fused)

        if return_attention:
            return output, {'v2c': v2c_attn, 'c2v': c2v_attn}
        return output


class TripleModalAlignment(nn.Module):
    """
    Triple-modal alignment: Time Series ↔ Vector ↔ Context

    Aligns three modalities:
    1. Raw time series features
    2. Vector modality (TS2Vec contrastive features)
    3. Context modality (LLM-empowered features)

    Args:
        d_ts (int): Time series dimension
        d_vector (int): Vector dimension
        d_context (int): Context dimension
        d_model (int): Hidden dimension
        n_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    def __init__(
        self,
        d_ts,
        d_vector,
        d_context,
        d_model=256,
        n_heads=8,
        dropout=0.1
    ):
        super().__init__()

        # Project all modalities to common dimension
        self.ts_proj = nn.Linear(d_ts, d_model)
        self.vector_proj = nn.Linear(d_vector, d_model)
        self.context_proj = nn.Linear(d_context, d_model)

        # Pairwise alignments
        self.ts_vector_align = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ts_context_align = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.vector_context_align = VectorContextAlignment(
            d_vector=d_vector,
            d_context=d_context,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            fusion_mode='gated'
        )

        # Layer norms
        self.norm_ts = nn.LayerNorm(d_model)
        self.norm_vector = nn.LayerNorm(d_model)
        self.norm_context = nn.LayerNorm(d_model)

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, ts_repr, vector_repr, context_repr):
        """
        Args:
            ts_repr: [B, L, d_ts] - Raw time series features
            vector_repr: [B, L, d_vector] - Vector modality
            context_repr: [B, N, d_context] - Context modality
        Returns:
            aligned: [B, L, d_model] - Triple-aligned representation
        """
        # Project to common space
        ts = self.ts_proj(ts_repr)
        vector = self.vector_proj(vector_repr)
        context = self.context_proj(context_repr)

        # Align TS with Vector
        ts_v, _ = self.ts_vector_align(ts, vector, vector)
        ts = self.norm_ts(ts + self.dropout(ts_v))

        # Align TS with Context
        ts_c, _ = self.ts_context_align(ts, context, context)
        ts = self.norm_ts(ts + self.dropout(ts_c))

        # Align Vector with Context (bidirectional)
        vc_aligned = self.vector_context_align(vector_repr, context_repr)

        # Ensure all have same length
        if context.size(1) != ts.size(1):
            context = context.transpose(1, 2)
            context = F.adaptive_avg_pool1d(context, ts.size(1))
            context = context.transpose(1, 2)

        # Concatenate and fuse
        combined = torch.cat([ts, vc_aligned, context], dim=-1)
        fused = self.fusion(combined)

        return fused


if __name__ == "__main__":
    # Test Vector-Context Alignment
    batch_size = 32
    len_v = 96
    len_c = 7
    d_vector = 320
    d_context = 768

    vector_repr = torch.randn(batch_size, len_v, d_vector)
    context_repr = torch.randn(batch_size, len_c, d_context)

    # Test VectorContextAlignment
    align_layer = VectorContextAlignment(
        d_vector=d_vector,
        d_context=d_context,
        d_model=256,
        n_heads=8
    )

    aligned, attn = align_layer(vector_repr, context_repr, return_attention=True)
    print(f"Vector shape: {vector_repr.shape}")
    print(f"Context shape: {context_repr.shape}")
    print(f"Aligned shape: {aligned.shape}")
    print(f"V2C attention shape: {attn['v2c'].shape}")
    print(f"C2V attention shape: {attn['c2v'].shape}")

    # Test TripleModalAlignment
    d_ts = 32
    ts_repr = torch.randn(batch_size, len_v, d_ts)

    triple_align = TripleModalAlignment(
        d_ts=d_ts,
        d_vector=d_vector,
        d_context=d_context,
        d_model=256
    )

    triple_aligned = triple_align(ts_repr, vector_repr, context_repr)
    print(f"\nTriple-aligned shape: {triple_aligned.shape}")
