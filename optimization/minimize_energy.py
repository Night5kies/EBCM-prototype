import torch

def minimize_energy(
    init_positions: torch.Tensor,
    energy_fn,
    steps: int = 100,
    lr: float = 0.05
) -> torch.Tensor:
    """
    Optimize object positions to minimize the given energy function.
    Returns optimized positions tensor.
    """
    positions = init_positions.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([positions], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        energy = energy_fn(positions)
        energy.backward()
        optimizer.step()
    return positions.detach()