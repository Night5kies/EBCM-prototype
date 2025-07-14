import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def visualize_scene(before: 'NPArray', after: 'NPArray'):
    """
    Visualize the 3D scene before and after optimization side by side.
    Positions arrays should be shape (N, 3).
    """
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(before[:, 0], before[:, 2], before[:, 1], s=50)
    ax1.set_title("Before Optimization")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.set_zlabel("Y")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(after[:, 0], after[:, 2], after[:, 1], s=50)
    ax2.set_title("After Optimization")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_zlabel("Y")

    plt.tight_layout()
    plt.show()