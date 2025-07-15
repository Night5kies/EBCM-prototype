import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def visualize_scene(before: 'NPArray', after: 'NPArray'):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(before[:, 0], before[:, 2], before[:, 1], s=50)
    ax1.set_title("Before Optimization")
    ax1.set_xlabel("X"); ax1.set_ylabel("Z"); ax1.set_zlabel("Y")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(after[:, 0], after[:, 2], after[:, 1], s=50)
    ax2.set_title("After Optimization")
    ax2.set_xlabel("X"); ax2.set_ylabel("Z"); ax2.set_zlabel("Y")

    plt.tight_layout(); plt.show()


def visualize_overlay(frame, pixel_coords_before, pixel_coords_after):
    """
    Overlay before/after centroid pixel positions onto the input image.
    pixel_coords_before/after: list of (x, y) tuples (column, row)
    """
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img_rgb)
    # Before: red, After: green
    if pixel_coords_before:
        xs_b, ys_b = zip(*pixel_coords_before)
        ax.scatter(xs_b, ys_b, c='r', marker='x', label='Before')
    if pixel_coords_after:
        xs_a, ys_a = zip(*pixel_coords_after)
        ax.scatter(xs_a, ys_a, c='g', marker='o', label='After')
    ax.legend(); ax.axis('off'); plt.show()