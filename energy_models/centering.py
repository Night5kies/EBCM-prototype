import torch

def centering_energy(positions: torch.Tensor) -> torch.Tensor:
    """
    Encourage objects to stay near the center of the area (x,z plane).
    """
    center = torch.tensor([0.5, 0.0, 0.5], device=positions.device)
    # Distance from center in 3D
    return torch.sum(torch.norm(positions - center, dim=1))