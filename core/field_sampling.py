"""
Field sampling module for Token Field Network (TFN).
Samples the evolved field at specified positions (e.g., token positions) to update token representations.
Supports differentiable nearest and linear interpolation.
"""
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class FieldSampler(nn.Module):
    """
    Samples a field at given positions using differentiable interpolation.
    Args:
        mode: 'nearest' or 'linear' (default: 'linear')
    """
    def __init__(self, mode: Literal['nearest', 'linear'] = 'linear'):
        super().__init__()
        assert mode in ('nearest', 'linear'), "mode must be 'nearest' or 'linear'"
        self.mode = mode

    def forward(
        self,
        field: torch.Tensor,         # [B, G, D] field values at grid points
        grid_points: torch.Tensor,   # [B, G, P] grid coordinates (P=1 for 1D)
        sample_positions: torch.Tensor  # [B, N, P] positions to sample at
    ) -> torch.Tensor:
        """
        Args:
            field: [B, G, D] - field values at grid points
            grid_points: [B, G, P] - grid coordinates (P=1 for 1D)
            sample_positions: [B, N, P] - positions to sample at
        Returns:
            sampled: [B, N, D] - field values at sample positions
        """
        B, G, D = field.shape
        _, N, P = sample_positions.shape
        assert grid_points.shape == (B, G, P)
        assert P == 1, "Only 1D sampling supported for now"

        # Flatten batch for vectorized ops
        field_flat = field.view(B, G, D)
        grid_flat = grid_points.view(B, G)
        pos_flat = sample_positions.view(B, N)

        # Handle out-of-bounds sampling by clamping positions to grid bounds
        grid_min = grid_flat[:, 0:1]  # [B, 1]
        grid_max = grid_flat[:, -1:]  # [B, 1]
        pos_clamped = torch.clamp(pos_flat, grid_min, grid_max)  # [B, N]

        # For each sample position, find left/right grid indices
        idx_left = torch.searchsorted(grid_flat, pos_clamped, right=True) - 1
        idx_left = idx_left.clamp(0, G - 2)  # [B, N]
        idx_right = idx_left + 1

        g_left = torch.gather(grid_flat, 1, idx_left)  # [B, N]
        g_right = torch.gather(grid_flat, 1, idx_right)  # [B, N]

        f_left = torch.gather(field_flat, 1, idx_left.unsqueeze(-1).expand(-1, -1, D))  # [B, N, D]
        f_right = torch.gather(field_flat, 1, idx_right.unsqueeze(-1).expand(-1, -1, D))  # [B, N, D]

        if self.mode == 'nearest':
            dist_left = (pos_clamped - g_left).abs()
            dist_right = (pos_clamped - g_right).abs()
            use_left = (dist_left <= dist_right).unsqueeze(-1)
            sampled = torch.where(use_left, f_left, f_right)
        else:  # linear
            denom = (g_right - g_left).clamp(min=1e-8)
            w_right = (pos_clamped - g_left) / denom
            w_left = 1.0 - w_right
            sampled = w_left.unsqueeze(-1) * f_left + w_right.unsqueeze(-1) * f_right

        return sampled  # [B, N, D] 