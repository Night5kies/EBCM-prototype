import torch
from utils.generate_scene import generate_scene
from optimization.minimize_energy import minimize_energy
from energy_models.total_energy import total_energy
from utils.visualize import visualize_scene


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generate a toy scene
    scene = generate_scene(n_objects=5)

    # Optimize positions to minimize energy
    optimized = minimize_energy(scene, total_energy, steps=100, lr=0.1)

    # Visualize before vs after
    visualize_scene(scene.numpy(), optimized.numpy())

if __name__ == '__main__':
    main()
