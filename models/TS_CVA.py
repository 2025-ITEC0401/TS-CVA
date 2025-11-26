"""
TS-CVA: Time Series Forecasting via Cross-modal Variable Alignment

Extends TimeCMA with vector modality from TS2Vec for enhanced forecasting performance.

Architecture:
1. Time Series Branch: Raw time series encoding (from TimeCMA)
2. Vector Modality Branch: TS2Vec contrastive learning encoder
3. Context Modality Branch: LLM-empowered encoder (from TimeCMA)
4. Triple-Modal Alignment: Aligns all three modalities
5. Decoder: Transformer decoder for forecasting
6. Projection: Linear projection to prediction horizon

Key Features:
- Multi-task learning: Contrastive loss + Forecasting loss
- Pre-training option: TS2Vec pre-training before end-to-end training
- Flexible fusion strategies: Gated, weighted, or concatenation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal
from layers.Vector_Context_Align import VectorContextAlignment, TripleModalAlignment
from models.TS2Vec import TS2VecEncoder


class TS_CVA(nn.Module):
    """
    TS-CVA: Cross-modal Variable Alignment for Time Series Forecasting

    Args:
        device (str): Device to use
        channel (int): Hidden dimension for time series branch
        num_nodes (int): Number of variables
        seq_len (int): Input sequence length
        pred_len (int): Prediction horizon
        dropout_n (float): Dropout rate
        d_llm (int): LLM embedding dimension
        d_vector (int): TS2Vec output dimension
        e_layer (int): Number of encoder layers
        d_layer (int): Number of decoder layers
        d_ff (int): Feed-forward dimension
        head (int): Number of attention heads
        use_triple_align (bool): Whether to use triple-modal alignment
        fusion_mode (str): Fusion strategy ('gated', 'weighted', 'concat')
        contrastive_weight (float): Weight for contrastive loss
    """
    def __init__(
        self,
        device="cuda:0",
        channel=32,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.1,
        d_llm=768,
        d_vector=320,
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8,
        use_triple_align=True,
        fusion_mode='gated',
        contrastive_weight=0.3
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n = dropout_n
        self.d_llm = d_llm
        self.d_vector = d_vector
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head
        self.use_triple_align = use_triple_align
        self.contrastive_weight = contrastive_weight

        # Normalization (RevIN)
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # === Time Series Branch (from TimeCMA) ===
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(
            self.ts_encoder_layer,
            num_layers=self.e_layer
        ).to(self.device)

        # === Vector Modality Branch (TS2Vec) ===
        self.vector_encoder = TS2VecEncoder(
            input_dims=self.num_nodes,
            output_dims=self.d_vector,
            hidden_dims=64,
            depth=10,
            mask_mode='binomial'
        ).to(self.device)

        # === Context Modality Branch (LLM from TimeCMA) ===
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_encoder_layer,
            num_layers=self.e_layer
        ).to(self.device)

        # === Cross-Modal Alignment ===
        if use_triple_align:
            # Triple-modal alignment: TS ↔ Vector ↔ Context
            self.alignment = TripleModalAlignment(
                d_ts=self.channel,
                d_vector=self.d_vector,
                d_context=self.d_llm,
                d_model=self.channel,
                n_heads=self.head,
                dropout=self.dropout_n
            ).to(self.device)
        else:
            # Vector-Context alignment only
            self.vc_alignment = VectorContextAlignment(
                d_vector=self.d_vector,
                d_context=self.d_llm,
                d_model=self.channel,
                n_heads=self.head,
                dropout=self.dropout_n,
                fusion_mode=fusion_mode
            ).to(self.device)

            # TS-Aligned cross attention (from TimeCMA)
            self.ts_cross = CrossModal(
                d_model=self.num_nodes,
                n_heads=1,
                d_ff=self.d_ff,
                norm='LayerNorm',
                attn_dropout=self.dropout_n,
                dropout=self.dropout_n,
                pre_norm=True,
                activation="gelu",
                res_attention=True,
                n_layers=1,
                store_attn=False
            ).to(self.device)

        # === Decoder ===
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=self.d_layer
        ).to(self.device)

        # === Projection to Prediction Horizon ===
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def forward(self, input_data, input_data_mark, embeddings, return_contrastive=False):
        """
        Forward pass

        Args:
            input_data: [B, L, N] - Input time series
            input_data_mark: [B, L, D] - Time features (not used currently)
            embeddings: [B, E, N, 1] - LLM prompt embeddings
            return_contrastive: Whether to return contrastive representations
        Returns:
            prediction: [B, pred_len, N] - Forecasted values
            contrastive_repr: Optional [B, C, L] - TS2Vec representations for contrastive loss
        """
        input_data = input_data.float()
        embeddings = embeddings.float()

        B, L, N = input_data.shape

        # === Normalization (RevIN) ===
        input_data_norm = self.normalize_layers(input_data, 'norm')

        # === 1. Time Series Branch ===
        ts_data = input_data_norm.permute(0, 2, 1)  # [B, N, L]
        ts_features = self.length_to_feature(ts_data)  # [B, N, C]
        ts_encoded = self.ts_encoder(ts_features)  # [B, N, C]
        ts_encoded = ts_encoded.permute(0, 2, 1)  # [B, C, N]

        # === 2. Vector Modality Branch (TS2Vec) ===
        vector_repr = self.vector_encoder(input_data_norm)  # [B, C_vec, L]

        # Adaptive pooling to match num_nodes for alignment: [B, C_vec, N]
        vector_repr_pooled = F.adaptive_avg_pool1d(vector_repr, self.num_nodes)

        # Transpose for alignment: [B, N, C_vec]
        vector_repr_T = vector_repr_pooled.transpose(1, 2)

        # === 3. Context Modality Branch (LLM) ===
        embeddings = embeddings.squeeze(-1)  # [B, E, N]
        embeddings = embeddings.permute(0, 2, 1)  # [B, N, E]
        context_encoded = self.prompt_encoder(embeddings)  # [B, N, E]
        context_encoded = context_encoded.permute(0, 2, 1)  # [B, E, N]

        # === 4. Cross-Modal Alignment ===
        if self.use_triple_align:
            # Triple-modal alignment
            # Prepare inputs: [B, L, C] format
            ts_for_align = ts_encoded.transpose(1, 2)  # [B, N, C] - treating N as sequence
            context_for_align = context_encoded.transpose(1, 2)  # [B, N, E]

            # Align (will handle length mismatch internally)
            aligned = self.alignment(
                ts_for_align,
                vector_repr_T,
                context_for_align
            )  # [B, L or N, C]

            # Reshape back to [B, C, N] for decoder
            if aligned.size(1) == L:
                # Pooling to N
                aligned = aligned.transpose(1, 2)  # [B, C, L]
                aligned = F.adaptive_avg_pool1d(aligned, N)  # [B, C, N]
            else:
                aligned = aligned.transpose(1, 2)  # [B, C, N]

            cross_out = aligned.permute(0, 2, 1)  # [B, N, C]

        else:
            # Vector-Context alignment
            vc_aligned = self.vc_alignment(
                vector_repr_T,  # [B, L, C_vec]
                context_encoded.transpose(1, 2)  # [B, N, E]
            )  # [B, L, C]

            # Pool to [B, N, C]
            vc_aligned = vc_aligned.transpose(1, 2)  # [B, C, L]
            vc_aligned = F.adaptive_avg_pool1d(vc_aligned, N)  # [B, C, N]
            vc_aligned = vc_aligned.permute(0, 2, 1)  # [B, N, C]

            # Cross with TS
            ts_enc_cross = ts_encoded  # [B, C, N]
            vc_enc = vc_aligned.permute(0, 2, 1)  # [B, C, N]

            cross_out = self.ts_cross(ts_enc_cross, vc_enc, vc_enc)  # [B, C, N]
            cross_out = cross_out.permute(0, 2, 1)  # [B, N, C]

        # === 5. Decoder ===
        dec_out = self.decoder(cross_out, cross_out)  # [B, N, C]

        # === 6. Projection ===
        dec_out = self.c_to_length(dec_out)  # [B, N, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, N]

        # === Denormalization ===
        output = self.normalize_layers(dec_out, 'denorm')

        if return_contrastive:
            return output, vector_repr
        return output

    def param_num(self):
        """Count total parameters"""
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test TS-CVA model
    device = "cpu"  # Use CPU for testing
    batch_size = 4
    seq_len = 96
    pred_len = 96
    num_nodes = 7
    d_llm = 768

    # Create sample data
    input_data = torch.randn(batch_size, seq_len, num_nodes)
    input_data_mark = torch.randn(batch_size, seq_len, 4)
    embeddings = torch.randn(batch_size, d_llm, num_nodes, 1)

    # Initialize model
    model = TS_CVA(
        device=device,
        channel=32,
        num_nodes=num_nodes,
        seq_len=seq_len,
        pred_len=pred_len,
        dropout_n=0.1,
        d_llm=d_llm,
        d_vector=320,
        e_layer=2,
        d_layer=1,
        d_ff=64,
        head=8,
        use_triple_align=True
    )

    print(f"Model initialized on {device}")
    print(f"Total parameters: {model.param_num():,}")
    print(f"Trainable parameters: {model.count_trainable_params():,}")

    # Forward pass
    output, vector_repr = model(
        input_data,
        input_data_mark,
        embeddings,
        return_contrastive=True
    )

    print(f"\nInput shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Vector representation shape: {vector_repr.shape}")
    print(f"\nExpected output: [{batch_size}, {pred_len}, {num_nodes}]")
    print(f"Actual output: {list(output.shape)}")
