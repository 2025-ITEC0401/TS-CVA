"""
TS2Vec Encoder Implementation
Time Series Representation Learning via Temporal and Instance Contrastive Learning

Based on: Zhihan Yue et al. "TS2Vec: Towards Universal Representation of Time Series" (2022)
Architecture: Dilated Convolutional Network with hierarchical temporal pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SamePadConv(nn.Module):
    """1D Convolution with same padding to preserve temporal dimension"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out


class ConvBlock(nn.Module):
    """Residual convolutional block with batch normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.bn = nn.BatchNorm1d(out_channels)
        self.final = final

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        if not self.final:
            x = self.bn(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    """
    Dilated Convolutional Encoder for TS2Vec

    Uses exponentially increasing dilation rates to capture multi-scale temporal patterns.
    Each layer doubles the receptive field while maintaining temporal resolution.

    Args:
        in_channels (int): Number of input variables/features
        channels (list): Hidden dimensions for each layer [64, 128, 256, ...]
        kernel_size (int): Convolutional kernel size (default: 3)
    """
    def __init__(self, in_channels, channels, kernel_size=3):
        super().__init__()
        self.input_fc = nn.Linear(in_channels, channels[0])

        layers = []
        for i in range(len(channels) - 1):
            dilation = 2 ** i
            layers.append(
                ConvBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilation,
                    final=(i == len(channels) - 2)
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [B, L, N] - Batch, Length, Num_vars
        Returns:
            out: [B, C, L] - Batch, Channels, Length
        """
        x = self.input_fc(x)  # [B, L, N] -> [B, L, C]
        x = x.transpose(1, 2)  # [B, L, C] -> [B, C, L]
        x = self.network(x)  # [B, C, L]
        return x


class TS2VecEncoder(nn.Module):
    """
    Complete TS2Vec Encoder with temporal pooling for multi-scale representations

    This encoder produces hierarchical representations at different temporal resolutions,
    which is crucial for capturing both fine-grained and coarse-grained patterns.

    Args:
        input_dims (int): Number of input variables
        output_dims (int): Output representation dimension
        hidden_dims (int): Hidden dimension (default: 64)
        depth (int): Number of convolutional layers (default: 10)
        mask_mode (str): Masking strategy - 'binomial', 'continuous', or 'all_true'
    """
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        mask_mode='binomial'
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode

        # Build channel dimensions [hidden, hidden, ..., output]
        channels = [hidden_dims] * depth + [output_dims]

        self.encoder = DilatedConvEncoder(
            input_dims,
            channels,
            kernel_size=3
        )

        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, N] - Batch, Length, Num_vars
            mask: [B, L] - Optional mask for temporal masking
        Returns:
            out: [B, C, L] - Encoded representations
        """
        if mask is None:
            if self.training:
                mask = self.generate_mask(x)
            else:
                mask = torch.ones_like(x[:, :, 0], dtype=torch.bool)

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Encode
        out = self.encoder(x)
        out = self.repr_dropout(out)

        return out

    def generate_mask(self, x):
        """Generate random temporal mask for contrastive learning"""
        B, L, N = x.shape

        if self.mask_mode == 'binomial':
            # Random binary mask (each timestep has 50% chance of being masked)
            mask = torch.rand(B, L, device=x.device) > 0.5
        elif self.mask_mode == 'continuous':
            # Continuous random masking (random consecutive segments)
            mask = torch.ones(B, L, device=x.device, dtype=torch.bool)
            for i in range(B):
                mask_start = np.random.randint(0, L)
                mask_len = np.random.randint(1, L // 4)
                mask[i, mask_start:min(mask_start + mask_len, L)] = False
        elif self.mask_mode == 'all_true':
            mask = torch.ones(B, L, device=x.device, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown mask mode: {self.mask_mode}")

        return mask

    def encode(self, x, mask=None, encoding_window=None):
        """
        Encode and return instance-level representation

        Args:
            x: [B, L, N]
            mask: Optional temporal mask
            encoding_window: If 'full_series', use entire series; else use specific window
        Returns:
            repr: [B, output_dims] - Instance-level representation
        """
        out = self.forward(x, mask)  # [B, C, L]

        if encoding_window == 'full_series':
            # Max pooling across entire temporal dimension
            repr = F.max_pool1d(
                out,
                kernel_size=out.size(2)
            ).squeeze(-1)  # [B, C]
        else:
            # Return all temporal representations
            repr = out.transpose(1, 2)  # [B, L, C]

        return repr


class TS2VecProjectionHead(nn.Module):
    """
    Projection head for contrastive learning
    Maps encoder output to a lower-dimensional space for computing contrastive loss
    """
    def __init__(self, input_dims, hidden_dims=128, output_dims=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C] or [B, L, C]
        Returns:
            out: [B, D] or [B, L, D] - Projected representation
        """
        if x.dim() == 3:
            B, L, C = x.shape
            x = x.reshape(B * L, C)
            out = self.net(x)
            out = out.reshape(B, L, -1)
        else:
            out = self.net(x)
        return out


if __name__ == "__main__":
    # Test TS2Vec encoder
    batch_size = 32
    seq_len = 96
    num_vars = 7

    # Create sample data
    x = torch.randn(batch_size, seq_len, num_vars)

    # Initialize encoder
    encoder = TS2VecEncoder(
        input_dims=num_vars,
        output_dims=320,
        hidden_dims=64,
        depth=10
    )

    # Forward pass
    out = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # Expected: [32, 320, 96]

    # Instance-level encoding
    instance_repr = encoder.encode(x, encoding_window='full_series')
    print(f"Instance representation shape: {instance_repr.shape}")  # Expected: [32, 320]
