import torch
from typing import Tuple

def mse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Squared Error (MSE) between predictions and targets.
    Args:
        preds: [N, ...] predicted values
        targets: [N, ...] ground truth values
    Returns:
        float: mean squared error
    """
    return torch.mean((preds - targets) ** 2).item()

def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error (MAE) between predictions and targets.
    Args:
        preds: [N, ...] predicted values
        targets: [N, ...] ground truth values
    Returns:
        float: mean absolute error
    """
    return torch.mean(torch.abs(preds - targets)).item()


def relative_l2_error(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the relative L2 error norm.
    
    This metric is commonly used in physics-informed neural networks and PDE solving
    to evaluate the accuracy of solutions. It measures the relative error between
    predicted and target solutions.
    
    Args:
        preds: [N, ...] predicted values
        targets: [N, ...] ground truth values
    Returns:
        float: relative L2 error (mean across samples)
    """
    # Reshape to [N, -1] to handle any input shape
    preds_flat = preds.reshape(preds.shape[0], -1)
    targets_flat = targets.reshape(targets.shape[0], -1)
    
    # Compute L2 norm of the difference
    l2_diff = torch.norm(preds_flat - targets_flat, dim=1)
    
    # Compute L2 norm of the targets
    l2_targets = torch.norm(targets_flat, dim=1)
    
    # Avoid division by zero
    l2_targets = torch.where(l2_targets == 0, torch.ones_like(l2_targets), l2_targets)
    
    # Compute relative L2 error
    relative_error = l2_diff / l2_targets
    
    return torch.mean(relative_error).item()


if __name__ == "__main__":
    p = torch.tensor([1.0, 2.0, 3.0])
    t = torch.tensor([1.5, 2.5, 2.0])
    print("MSE:", mse(p, t))
    print("MAE:", mae(p, t))
    print("Relative L2 Error:", relative_l2_error(p, t)) 