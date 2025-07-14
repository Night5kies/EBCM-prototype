import torch

def groundedness_energy(positions: torch.Tensor) -> torch.Tensor:
    """
    Penalize objects that are above the ground plane (y > 0).
    """
    y_coords = positions[:, 1]
    # Clamp negative values to zero (below ground allowed), penalize hovering
    return torch.sum(torch.clamp(y_coords, min=0.0))