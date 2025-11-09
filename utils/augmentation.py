"""
Data Augmentation for Time Series

Implements various augmentation techniques for contrastive learning on time series data.
These augmentations help the model learn robust and invariant representations.

Reference:
- Um et al. "Data augmentation of wearable sensor data for parkinson's disease monitoring" (2017)
- Rashid & Louis "Times-series data augmentation" (2019)
"""

import torch
import numpy as np
from scipy.interpolate import CubicSpline


class TimeSeriesAugmentation:
    """
    Collection of time series augmentation methods

    Each method takes input [B, L, N] and returns augmented [B, L, N]
    where B=batch, L=length, N=num_variables
    """

    @staticmethod
    def jitter(x, sigma=0.03):
        """
        Add random Gaussian noise to the time series

        Args:
            x: [B, L, N] - Input time series
            sigma: Standard deviation of noise
        Returns:
            augmented: [B, L, N]
        """
        noise = torch.randn_like(x) * sigma
        return x + noise

    @staticmethod
    def scaling(x, sigma=0.1):
        """
        Multiply time series by random scaling factor

        Args:
            x: [B, L, N]
            sigma: Standard deviation of scaling factor
        Returns:
            augmented: [B, L, N]
        """
        B, L, N = x.shape
        factor = torch.randn(B, 1, N, device=x.device) * sigma + 1.0
        return x * factor

    @staticmethod
    def rotation(x):
        """
        Apply random rotation in the feature space (useful for multivariate series)

        Args:
            x: [B, L, N]
        Returns:
            augmented: [B, L, N]
        """
        B, L, N = x.shape
        if N <= 1:
            return x

        # Generate random rotation matrix
        flip = torch.randint(0, 2, (N,), device=x.device) * 2 - 1  # Random {-1, 1}
        rotate_axis = torch.arange(N, device=x.device)
        torch.random.shuffle(rotate_axis)

        return flip * x[:, :, rotate_axis]

    @staticmethod
    def permutation(x, max_segments=5, seg_mode="equal"):
        """
        Randomly permute segments of the time series

        Args:
            x: [B, L, N]
            max_segments: Maximum number of segments to divide the series
            seg_mode: 'equal' or 'random' segment sizes
        Returns:
            augmented: [B, L, N]
        """
        B, L, N = x.shape
        orig_steps = torch.arange(L, device=x.device)

        num_segs = torch.randint(1, max_segments + 1, (1,)).item()

        if seg_mode == "equal":
            # Equal-sized segments
            split_points = torch.linspace(0, L, num_segs + 1, device=x.device).long()
        else:
            # Random-sized segments
            split_points = torch.sort(torch.randint(0, L, (num_segs - 1,), device=x.device))[0]
            split_points = torch.cat([
                torch.tensor([0], device=x.device),
                split_points,
                torch.tensor([L], device=x.device)
            ])

        # Create permutation
        perm = torch.randperm(num_segs, device=x.device)

        # Permute segments
        ret = torch.zeros_like(x)
        for i in range(num_segs):
            start_orig = split_points[perm[i]]
            end_orig = split_points[perm[i] + 1]
            start_new = split_points[i]
            end_new = split_points[i + 1]

            # Handle different segment lengths
            seg_len = min(end_orig - start_orig, end_new - start_new)
            ret[:, start_new:start_new + seg_len] = x[:, start_orig:start_orig + seg_len]

        return ret

    @staticmethod
    def magnitude_warp(x, sigma=0.2, knot=4):
        """
        Warp magnitude using smooth random curves (cubic spline)

        Args:
            x: [B, L, N]
            sigma: Standard deviation of warping curve
            knot: Number of knots for spline
        Returns:
            augmented: [B, L, N]
        """
        B, L, N = x.shape
        device = x.device

        # Generate random warping curve
        orig_steps = np.arange(L)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(B, knot + 2, N))

        # Create warping curves using cubic spline
        warp_steps = np.linspace(0, L - 1, num=knot + 2)

        ret = torch.zeros_like(x)
        for b in range(B):
            for n in range(N):
                # Create smooth warping curve
                warper = CubicSpline(warp_steps, random_warps[b, :, n])
                warp_curve = warper(orig_steps)
                ret[b, :, n] = x[b, :, n] * torch.from_numpy(warp_curve).float().to(device)

        return ret

    @staticmethod
    def time_warp(x, sigma=0.2, knot=4):
        """
        Warp time axis using smooth random curves

        Args:
            x: [B, L, N]
            sigma: Standard deviation of warping
            knot: Number of knots
        Returns:
            augmented: [B, L, N]
        """
        B, L, N = x.shape
        device = x.device

        orig_steps = np.arange(L)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(B, knot + 2))
        warp_steps = np.linspace(0, L - 1, num=knot + 2)

        ret = torch.zeros_like(x)
        for b in range(B):
            # Time warping curve
            time_warp = CubicSpline(warp_steps, warp_steps * random_warps[b])
            warped_time = time_warp(orig_steps)

            # Clamp to valid range
            warped_time = np.clip(warped_time, 0, L - 1)

            # Interpolate
            for n in range(N):
                ret[b, :, n] = torch.from_numpy(
                    np.interp(orig_steps, warped_time, x[b, :, n].cpu().numpy())
                ).float().to(device)

        return ret

    @staticmethod
    def window_slice(x, reduce_ratio=0.9):
        """
        Randomly crop a window from the time series

        Args:
            x: [B, L, N]
            reduce_ratio: Target length / original length
        Returns:
            augmented: [B, L, N] (zero-padded if needed)
        """
        B, L, N = x.shape
        target_len = max(1, int(L * reduce_ratio))

        start = torch.randint(0, L - target_len + 1, (B,))

        ret = torch.zeros_like(x)
        for b in range(B):
            ret[b, :target_len] = x[b, start[b]:start[b] + target_len]

        return ret

    @staticmethod
    def window_warp(x, window_ratio=0.1, scales=[0.5, 2.0]):
        """
        Warp a random window of the time series

        Args:
            x: [B, L, N]
            window_ratio: Size of window to warp
            scales: Range of scaling factors [min, max]
        Returns:
            augmented: [B, L, N]
        """
        B, L, N = x.shape
        device = x.device

        warp_size = max(1, int(L * window_ratio))

        ret = x.clone()
        for b in range(B):
            # Random window
            start = torch.randint(0, L - warp_size + 1, (1,)).item()
            end = start + warp_size

            # Random scale
            scale = np.random.uniform(scales[0], scales[1])

            # Warp window
            window = x[b, start:end]
            warped_len = max(1, int(warp_size * scale))

            # Interpolate to new length
            for n in range(N):
                orig_indices = np.linspace(0, warp_size - 1, warp_size)
                new_indices = np.linspace(0, warp_size - 1, warped_len)
                warped = np.interp(new_indices, orig_indices, window[:, n].cpu().numpy())

                # Replace window (truncate or pad)
                if warped_len < warp_size:
                    ret[b, start:start + warped_len, n] = torch.from_numpy(warped).float().to(device)
                    ret[b, start + warped_len:end, n] = 0
                else:
                    ret[b, start:end, n] = torch.from_numpy(warped[:warp_size]).float().to(device)

        return ret


class RandomAugmentor:
    """
    Randomly apply one or more augmentations

    Args:
        augmentation_list: List of augmentation names to choose from
        n_augmentations: Number of augmentations to apply (default: 1)
        augmentation_prob: Probability of applying each augmentation
    """
    def __init__(self, augmentation_list=None, n_augmentations=1, augmentation_prob=0.5):
        if augmentation_list is None:
            augmentation_list = ['jitter', 'scaling', 'permutation', 'magnitude_warp']

        self.augmentation_list = augmentation_list
        self.n_augmentations = n_augmentations
        self.augmentation_prob = augmentation_prob

        self.augmentations = {
            'jitter': TimeSeriesAugmentation.jitter,
            'scaling': TimeSeriesAugmentation.scaling,
            'rotation': TimeSeriesAugmentation.rotation,
            'permutation': TimeSeriesAugmentation.permutation,
            'magnitude_warp': TimeSeriesAugmentation.magnitude_warp,
            'time_warp': TimeSeriesAugmentation.time_warp,
            'window_slice': TimeSeriesAugmentation.window_slice,
            'window_warp': TimeSeriesAugmentation.window_warp,
        }

    def __call__(self, x):
        """
        Apply random augmentations

        Args:
            x: [B, L, N]
        Returns:
            augmented: [B, L, N]
        """
        # Randomly select augmentations
        selected = np.random.choice(
            self.augmentation_list,
            size=self.n_augmentations,
            replace=False
        )

        augmented = x
        for aug_name in selected:
            if np.random.random() < self.augmentation_prob:
                augmented = self.augmentations[aug_name](augmented)

        return augmented


class StockAugmentor(RandomAugmentor):
    """
    Augmentor specialized for stock price data

    Uses conservative augmentations suitable for financial time series
    """
    def __init__(self, n_augmentations=1, augmentation_prob=0.6):
        # Stock data is sensitive - use conservative augmentations
        augmentation_list = ['jitter', 'scaling', 'window_slice']
        super().__init__(augmentation_list, n_augmentations, augmentation_prob)


if __name__ == "__main__":
    # Test augmentations
    batch_size = 4
    seq_len = 96
    num_vars = 7

    x = torch.randn(batch_size, seq_len, num_vars)

    print("Original shape:", x.shape)
    print("Original mean:", x.mean().item(), "std:", x.std().item())

    # Test individual augmentations
    augmentors = {
        'jitter': lambda: TimeSeriesAugmentation.jitter(x),
        'scaling': lambda: TimeSeriesAugmentation.scaling(x),
        'permutation': lambda: TimeSeriesAugmentation.permutation(x),
        'magnitude_warp': lambda: TimeSeriesAugmentation.magnitude_warp(x),
    }

    for name, aug_fn in augmentors.items():
        augmented = aug_fn()
        print(f"\n{name}:")
        print(f"  Shape: {augmented.shape}")
        print(f"  Mean: {augmented.mean().item():.4f}, Std: {augmented.std().item():.4f}")

    # Test random augmentor
    print("\n\nRandom Augmentor:")
    random_aug = RandomAugmentor(n_augmentations=2)
    augmented = random_aug(x)
    print(f"Shape: {augmented.shape}")
    print(f"Mean: {augmented.mean().item():.4f}, Std: {augmented.std().item():.4f}")
