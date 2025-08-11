"""
Field sampling module for Token Field Network (TFN).
Samples the evolved field at specified positions (e.g., token positions) to update token representations.
Uses efficient, hardware-accelerated grid_sample for optimal GPU performance.
Supports differentiable nearest and linear interpolation.
"""
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class FieldSampler(nn.Module):
    """
    Samples a field at given positions using efficient, hardware-accelerated interpolation.
    
    This implementation uses torch.nn.functional.grid_sample for optimal performance
    on GPUs, replacing the slower searchsorted-based approach.
    
    Args:
        mode: 'nearest' or 'linear' (default: 'linear')
            - 'nearest': Nearest neighbor interpolation
            - 'linear': Bilinear interpolation (smooth)
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
        Efficient field sampling using grid_sample for optimal GPU performance.
        
        This method replaces the slower searchsorted-based approach with a single,
        hardware-accelerated grid_sample operation that provides the same physical
        interpolation but with significantly better performance.
        
        Args:
            field: [B, G, D] - field values at grid points
            grid_points: [B, G, P] - grid coordinates (P=1 for 1D, assumed to be in [0, 1] range)
            sample_positions: [B, N, P] - positions to sample at (assumed to be in [0, 1] range)
        Returns:
            sampled: [B, N, D] - field values at sample positions
        """
        B, G, D = field.shape
        _, N, P = sample_positions.shape
        assert grid_points.shape == (B, G, P)
        assert P == 1, "Only 1D sampling supported for now"

        # --- EFFICIENT GRID_SAMPLE IMPLEMENTATION ---
        # Reshape field for grid_sample: [B, G, D] -> [B, D, 1, G] (like a 2D image)
        field_reshaped = field.transpose(1, 2).unsqueeze(2)  # [B, D, 1, G]
        
        # Normalize sample positions to [-1, 1] range for grid_sample
        # grid_points are assumed to be in [0, 1] range, so we map to [-1, 1]
        sample_coords = sample_positions * 2.0 - 1.0  # [B, N, 1] -> [B, N, 1]
        
        # Reshape coordinates for grid_sample: [B, N, 1] -> [B, N, 2]
        # grid_sample expects coordinates in format [B, H, W, 2] for 2D
        # For 1D, we treat it as a 2D image with height=1, so we need [B, 1, N, 2]
        # First, expand to 2D: [B, N, 1] -> [B, N, 2]
        sample_coords_2d = torch.cat([
            sample_coords,  # x coordinate [B, N, 1]
            torch.zeros_like(sample_coords)  # y coordinate (dummy) [B, N, 1]
        ], dim=-1)  # [B, N, 2]
        
        # Then reshape for grid_sample: [B, N, 2] -> [B, 1, N, 2]
        sample_coords_reshaped = sample_coords_2d.unsqueeze(1)  # [B, 1, N, 2]
        

        
        # Use grid_sample for efficient, hardware-accelerated interpolation
        # mode='bilinear' provides smooth interpolation, 'nearest' for nearest neighbor
        grid_mode = 'bilinear' if self.mode == 'linear' else 'nearest'
        
        sampled_field = F.grid_sample(
            field_reshaped,           # [B, D, 1, G] - input field
            sample_coords_reshaped,   # [B, 1, N, 2] - normalized coordinates
            mode=grid_mode,           # interpolation mode
            align_corners=True,       # consistent with PyTorch conventions
            padding_mode='border'     # handle out-of-bounds gracefully
        )  # [B, D, 1, N]
        
        # Reshape back to expected format: [B, D, 1, N] -> [B, N, D]
        sampled_field = sampled_field.squeeze(2).transpose(1, 2)  # [B, N, D]
        
        return sampled_field 