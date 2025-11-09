"""
Contrastive Loss Functions for TS2Vec

Implements hierarchical contrastive learning:
- Temporal Contrastive Loss: Aligns representations from different temporal contexts
- Instance Contrastive Loss: Aligns representations from augmented views

Based on InfoNCE (Noise Contrastive Estimation) framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def take_per_row(A, indx, num_elem):
    """
    Sample elements per row from a matrix A based on indices

    Args:
        A: [B, N, C] - Batch, Num_elements, Channels
        indx: [B, num_elem] - Indices to sample
        num_elem: Number of elements to sample per row
    Returns:
        sampled: [B, num_elem, C]
    """
    all_indx = indx[:, None, :] + torch.arange(A.shape[1], device=A.device)[None, :, None]
    return A[torch.arange(all_indx.shape[0])[:, None, None], all_indx]


class HierarchicalContrastiveLoss(nn.Module):
    """
    Hierarchical Contrastive Loss for TS2Vec

    This loss encourages representations from different temporal contexts of the same
    time series to be similar, while being different from other time series.

    Args:
        temperature (float): Temperature parameter for softmax (default: 0.05)
        temporal_unit (int): Size of temporal context window (default: 0 for adaptive)
    """
    def __init__(self, temperature=0.05, temporal_unit=0):
        super().__init__()
        self.temperature = temperature
        self.temporal_unit = temporal_unit

    def forward(self, z1, z2):
        """
        Compute hierarchical contrastive loss

        Args:
            z1: [B, C, L] - First view representations
            z2: [B, C, L] - Second view representations (augmented)
        Returns:
            loss: Scalar contrastive loss
        """
        B, C, L = z1.shape

        if self.temporal_unit == 0:
            temporal_unit = max(1, L // 10)
        else:
            temporal_unit = self.temporal_unit

        z1 = z1.transpose(1, 2)  # [B, L, C]
        z2 = z2.transpose(1, 2)  # [B, L, C]

        loss = 0
        d = 0

        # Iterate over temporal hierarchy (multi-scale)
        while L > 1:
            if self.temporal_unit > 0:
                if L <= temporal_unit:
                    break
            else:
                temporal_unit = max(1, L // 10)

            # Sample random temporal contexts
            # For each instance, sample two overlapping temporal segments
            loss += self._instance_contrastive_loss(z1, z2)

            # Temporal max pooling to create coarser representations
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)

            L = z1.size(1)
            d += 1

        if d == 0:
            return self._instance_contrastive_loss(z1, z2)
        return loss / d

    def _instance_contrastive_loss(self, z1, z2):
        """
        Instance-level contrastive loss using InfoNCE

        Args:
            z1: [B, L, C]
            z2: [B, L, C]
        Returns:
            loss: Scalar
        """
        B, L, C = z1.shape

        # Flatten temporal dimension
        z1_flat = z1.reshape(B * L, C)  # [B*L, C]
        z2_flat = z2.reshape(B * L, C)  # [B*L, C]

        # Normalize
        z1_norm = F.normalize(z1_flat, dim=1)
        z2_norm = F.normalize(z2_flat, dim=1)

        # Compute similarity matrix
        # Positive pairs: z1[i] and z2[i] from same instance and same timestep
        # Negative pairs: all others in the batch
        logits = torch.mm(z1_norm, z2_norm.T) / self.temperature  # [B*L, B*L]

        # Create labels: diagonal elements are positive pairs
        labels = torch.arange(B * L, device=z1.device)

        # InfoNCE loss
        loss = F.cross_entropy(logits, labels)

        return loss


class TemporalContrastiveLoss(nn.Module):
    """
    Temporal Contrastive Loss

    Encourages representations from different parts of the same time series
    to be similar (temporal consistency)

    Args:
        temperature (float): Temperature for softmax
    """
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, crop_l=None):
        """
        Args:
            z: [B, C, L] - Encoded representations
            crop_l: Crop length for temporal context sampling
        Returns:
            loss: Scalar
        """
        B, C, L = z.shape

        if crop_l is None:
            crop_l = max(1, L // 4)

        z = z.transpose(1, 2)  # [B, L, C]

        # Sample two random crops from each instance
        crop1_start = torch.randint(0, max(1, L - crop_l + 1), (B,))
        crop2_start = torch.randint(0, max(1, L - crop_l + 1), (B,))

        # Extract crops
        crop1 = []
        crop2 = []
        for i in range(B):
            crop1.append(z[i, crop1_start[i]:crop1_start[i] + crop_l].mean(0))
            crop2.append(z[i, crop2_start[i]:crop2_start[i] + crop_l].mean(0))

        crop1 = torch.stack(crop1)  # [B, C]
        crop2 = torch.stack(crop2)  # [B, C]

        # Normalize
        crop1_norm = F.normalize(crop1, dim=1)
        crop2_norm = F.normalize(crop2, dim=1)

        # Compute similarity
        logits = torch.mm(crop1_norm, crop2_norm.T) / self.temperature

        labels = torch.arange(B, device=z.device)

        loss = F.cross_entropy(logits, labels)

        return loss


class TS2VecLoss(nn.Module):
    """
    Combined TS2Vec Loss

    Combines hierarchical contrastive loss and temporal contrastive loss

    Args:
        alpha (float): Weight for hierarchical loss (default: 0.5)
        beta (float): Weight for temporal loss (default: 0.5)
        temperature (float): Temperature parameter
    """
    def __init__(self, alpha=0.5, beta=0.5, temperature=0.05):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.hierarchical_loss = HierarchicalContrastiveLoss(temperature=temperature)
        self.temporal_loss = TemporalContrastiveLoss(temperature=temperature)

    def forward(self, z1, z2):
        """
        Args:
            z1: [B, C, L] - First view
            z2: [B, C, L] - Second view (augmented)
        Returns:
            loss: Scalar combined loss
            loss_dict: Dictionary with individual losses
        """
        hier_loss = self.hierarchical_loss(z1, z2)
        temp_loss = self.temporal_loss(z1)

        total_loss = self.alpha * hier_loss + self.beta * temp_loss

        loss_dict = {
            'total': total_loss.item(),
            'hierarchical': hier_loss.item(),
            'temporal': temp_loss.item()
        }

        return total_loss, loss_dict


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    Alternative contrastive loss used in SimCLR

    Args:
        temperature (float): Temperature parameter
        use_cosine_similarity (bool): Whether to use cosine similarity
    """
    def __init__(self, temperature=0.5, use_cosine_similarity=True):
        super().__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: [B, C] - First view embeddings
            z_j: [B, C] - Second view embeddings
        Returns:
            loss: Scalar
        """
        batch_size = z_i.size(0)

        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)  # [2B, C]

        if self.use_cosine_similarity:
            representations = F.normalize(representations, dim=1)
            similarity_matrix = torch.mm(representations, representations.T)  # [2B, 2B]
        else:
            similarity_matrix = torch.mm(representations, representations.T) / self.temperature

        # Create mask for positive pairs
        # For each z_i[k], the positive pair is z_j[k]
        # Mask diagonal (self-similarity) and positive pairs
        mask = torch.eye(2 * batch_size, device=z_i.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        pos_sim = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ], dim=0).unsqueeze(1)  # [2B, 1]

        # Negative pairs: all others
        neg_sim = similarity_matrix

        # Compute logits
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature  # [2B, 2B+1]

        # Labels: positive pair is always the first element
        labels = torch.zeros(2 * batch_size, device=z_i.device, dtype=torch.long)

        loss = F.cross_entropy(logits, labels)

        return loss


if __name__ == "__main__":
    # Test contrastive losses
    batch_size = 32
    channels = 320
    length = 96

    # Create sample representations
    z1 = torch.randn(batch_size, channels, length)
    z2 = torch.randn(batch_size, channels, length)

    # Test hierarchical loss
    hier_loss = HierarchicalContrastiveLoss()
    loss = hier_loss(z1, z2)
    print(f"Hierarchical Contrastive Loss: {loss.item():.4f}")

    # Test temporal loss
    temp_loss = TemporalContrastiveLoss()
    loss = temp_loss(z1)
    print(f"Temporal Contrastive Loss: {loss.item():.4f}")

    # Test combined loss
    combined_loss = TS2VecLoss()
    loss, loss_dict = combined_loss(z1, z2)
    print(f"Combined Loss: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
