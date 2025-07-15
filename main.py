import torch
import cv2
from utils.scene_loader import load_depth_model, load_detector, extract_positions_from_frame
from optimization.minimize_energy import minimize_energy
from energy_models.total_energy import total_energy
from utils.visualize import visualize_scene


def main(use_real: bool = True, video_source=0):
    torch.manual_seed(42)
    if use_real:
        # Initialize models
        depth_model = load_depth_model()
        detector = load_detector()
        
        # Capture one frame (can be extended to video loop)
        cap = cv2.VideoCapture(video_source)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to read from video source")
        # Extract positions from real frame
        scene = extract_positions_from_frame(frame, depth_model, detector)
    else:
        from utils.generate_scene import generate_scene
        scene = generate_scene(n_objects=5)

    if scene.numel() == 0:
        print("No objects detected. Exiting.")
        return

    optimized = minimize_energy(scene, total_energy, steps=100, lr=0.1)
    visualize_scene(scene.numpy(), optimized.numpy())

if __name__ == '__main__':
    main()