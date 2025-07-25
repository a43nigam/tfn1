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

if __name__ == "__main__":
    p = torch.tensor([1.0, 2.0, 3.0])
    t = torch.tensor([1.5, 2.5, 2.0])
    print("MSE:", mse(p, t))
    print("MAE:", mae(p, t)) 