import torch

def generate_scene(n_objects: int = 5) -> torch.Tensor:
    """
    Generate a random 3D scene with boxes represented by their center positions.
    Positions: Tensor of shape (n_objects, 3) with x,z in [0,1], y in [0,0.5].
    """
    positions = torch.rand(n_objects, 3)
    positions[:, 1] = torch.rand(n_objects) * 0.5  # y-axis (height)
    return positions

