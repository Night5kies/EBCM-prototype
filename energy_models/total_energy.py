import torch
from .groundedness import groundedness_energy
from .non_interpretation import non_interpenetration_energy
from .centering import centering_energy

def total_energy(positions: torch.Tensor) -> torch.Tensor:
    """
    Sum of all defined energy terms for a scene.
    """
    e1 = groundedness_energy(positions)
    e2 = non_interpenetration_energy(positions)
    e3 = centering_energy(positions)
    return e1 + e2 + e3