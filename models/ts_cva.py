"""
TS-CVA: TimeSeries-Context Vector Modality Alignment

This module implements the TS-CVA encoder that combines:
1. TS2Vec Encoder: Dilated Convolutional Network for time series representation
2. Cross-Modal Alignment: Aligns time series embeddings with LLM-based context embeddings

Architecture:
    Input Time Series [B, T, N] 
        → TS2Vec Encoder → [B, T, D]
        → Cross-Modal Attention (with LLM embeddings) → [B, T, D]
        → Final Representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from .dilated_conv import DilatedConvEncoder
from .encoder import generate_continuous_mask, generate_binomial_mask


class TSCVAEncoder(nn.Module):
    """
    TS-CVA Encoder combining TS2Vec's dilated convolution with cross-modal attention.
    
    Args:
        input_dims: Number of input features (variables)
        output_dims: Output representation dimension
        hidden_dims: Hidden dimension for the encoder
        depth: Number of dilated conv blocks
        num_heads: Number of attention heads for cross-modal alignment
        d_llm: Dimension of LLM embeddings (default: 768 for BERT-base)
        dropout: Dropout rate
        mask_mode: Masking strategy ('binomial', 'continuous', 'all_true')
        use_cross_modal: Whether to use cross-modal alignment
    """
    
    def __init__(
        self,
        input_dims: int,
        output_dims: int = 320,
        hidden_dims: int = 64,
        depth: int = 10,
        num_heads: int = 8,
        d_llm: int = 768,
        d_ff: int = 256,
        dropout: float = 0.1,
        mask_mode: str = 'binomial',
        use_cross_modal: bool = True
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.use_cross_modal = use_cross_modal
        
        # TS2Vec Encoder Components
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=dropout)
        
        if use_cross_modal:
            # LLM Embedding Projection
            self.llm_projection = nn.Linear(d_llm, output_dims)
            
            # Prompt Encoder (for LLM embeddings)
            self.prompt_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=output_dims,
                    nhead=num_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True
                ),
                num_layers=1
            )
            
            # Cross-Modal Attention
            # Q: Time series encoding, K/V: LLM embeddings
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=output_dims,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.cross_norm = nn.LayerNorm(output_dims)
            self.cross_dropout = nn.Dropout(dropout)
            
            # Feed-forward after cross attention
            self.cross_ff = nn.Sequential(
                nn.Linear(output_dims, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, output_dims)
            )
            self.ff_norm = nn.LayerNorm(output_dims)
            self.ff_dropout = nn.Dropout(dropout)
            
            # Fusion gate for combining TS and cross-modal features
            self.fusion_gate = nn.Sequential(
                nn.Linear(output_dims * 2, output_dims),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        x: torch.Tensor,
        llm_embeddings: Optional[torch.Tensor] = None,
        mask: Optional[str] = None
    ) -> torch.Tensor:
        """
        Forward pass of TS-CVA Encoder.
        
        Args:
            x: Input time series [B, T, N] (Batch, Time, Features)
            llm_embeddings: LLM embeddings [B, E, D_llm] (Batch, Embedding_len, LLM_dim)
            mask: Masking mode override
            
        Returns:
            representations: [B, T, output_dims]
        """
        # Handle NaN values
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        
        # Input projection
        x = self.input_fc(x)  # [B, T, hidden_dims]
        
        # Generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask_tensor = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask_tensor = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask_tensor = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask_tensor = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask_tensor = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask_tensor[:, -1] = False
        else:
            mask_tensor = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        
        mask_tensor &= nan_mask
        x[~mask_tensor] = 0
        
        # TS2Vec Dilated Conv Encoder
        x = x.transpose(1, 2)  # [B, hidden_dims, T]
        ts_repr = self.feature_extractor(x)  # [B, output_dims, T]
        ts_repr = ts_repr.transpose(1, 2)  # [B, T, output_dims]
        ts_repr = self.repr_dropout(ts_repr)
        
        # Cross-Modal Alignment (if LLM embeddings provided)
        if self.use_cross_modal and llm_embeddings is not None:
            # Handle different LLM embedding shapes
            # Expected: [B, E, D_llm] or [B, E, D_llm, C] (per-channel embeddings)
            if llm_embeddings.dim() == 4:
                # Per-channel embeddings: [B, E, D_llm, C]
                # Average across channels to get [B, E, D_llm]
                llm_embeddings = llm_embeddings.mean(dim=-1)
            
            # Project LLM embeddings to same dimension
            llm_proj = self.llm_projection(llm_embeddings)  # [B, E, output_dims]
            
            # Encode LLM embeddings
            llm_encoded = self.prompt_encoder(llm_proj)  # [B, E, output_dims]
            
            # Cross-Modal Attention: Q=ts_repr, K=V=llm_encoded
            cross_out, _ = self.cross_attention(
                query=ts_repr,
                key=llm_encoded,
                value=llm_encoded
            )
            cross_out = self.cross_norm(ts_repr + self.cross_dropout(cross_out))
            
            # Feed-forward
            ff_out = self.cross_ff(cross_out)
            cross_out = self.ff_norm(cross_out + self.ff_dropout(ff_out))
            
            # Gated fusion
            gate = self.fusion_gate(torch.cat([ts_repr, cross_out], dim=-1))
            output = gate * cross_out + (1 - gate) * ts_repr
        else:
            output = ts_repr
        
        return output


class TSCVA(nn.Module):
    """
    TS-CVA Model wrapper for training with hierarchical contrastive loss.
    
    This model wraps TSCVAEncoder and provides training interface compatible
    with the original TS2Vec training pipeline.
    """
    
    def __init__(
        self,
        input_dims: int,
        output_dims: int = 320,
        hidden_dims: int = 64,
        depth: int = 10,
        num_heads: int = 8,
        d_llm: int = 768,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_cross_modal: bool = True
    ):
        super().__init__()
        
        self.encoder = TSCVAEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            num_heads=num_heads,
            d_llm=d_llm,
            d_ff=d_ff,
            dropout=dropout,
            use_cross_modal=use_cross_modal
        )
        
        self.output_dims = output_dims
        self.use_cross_modal = use_cross_modal
    
    def forward(
        self,
        x: torch.Tensor,
        llm_embeddings: Optional[torch.Tensor] = None,
        mask: Optional[str] = None
    ) -> torch.Tensor:
        """Forward pass."""
        return self.encoder(x, llm_embeddings, mask)
    
    def get_ts_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get pure time series representation without cross-modal alignment."""
        return self.encoder(x, llm_embeddings=None, mask='all_true')


# Contrastive loss for cross-modal alignment
def cross_modal_contrastive_loss(
    ts_repr: torch.Tensor,
    llm_repr: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Cross-modal contrastive loss to align time series and LLM embeddings.
    
    Args:
        ts_repr: Time series representations [B, D]
        llm_repr: LLM representations [B, D]
        temperature: Temperature for softmax
        
    Returns:
        loss: Contrastive loss value
    """
    B = ts_repr.size(0)
    if B == 1:
        return ts_repr.new_tensor(0.)
    
    # Normalize representations
    ts_repr = F.normalize(ts_repr, dim=-1)
    llm_repr = F.normalize(llm_repr, dim=-1)
    
    # Compute similarity
    logits = torch.matmul(ts_repr, llm_repr.T) / temperature  # [B, B]
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(B, device=ts_repr.device)
    
    # Cross entropy loss (bidirectional)
    loss_ts_to_llm = F.cross_entropy(logits, labels)
    loss_llm_to_ts = F.cross_entropy(logits.T, labels)
    
    return (loss_ts_to_llm + loss_llm_to_ts) / 2
