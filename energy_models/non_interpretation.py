import torch

def non_interpenetration_energy(positions: torch.Tensor) -> torch.Tensor:
    """
    Penalize overlapping objects using a simple repulsive energy.
    Assumes each object is approximated as a sphere with radius r.
    """
    n = positions.shape[0]
    energy = torch.tensor(0.0, device=positions.device)
    r = 0.1
    for i in range(n):
        for j in range(i + 1, n):
            dist = torch.norm(positions[i] - positions[j])
            overlap = torch.clamp((2 * r) - dist, min=0.0)
            energy = energy + overlap**2
    return energy