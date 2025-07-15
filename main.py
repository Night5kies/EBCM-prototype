import torch, cv2
from utils.scene_loader import load_depth_model, load_detector, extract_positions_from_frame
from optimization.minimize_energy import minimize_energy
from energy_models.total_energy import total_energy
from utils.visualize import visualize_scene, visualize_overlay

def main(use_real:bool=True, video_source=0):
    torch.manual_seed(42)
    if use_real:
        depth_model, detector = load_depth_model(), load_detector()
        cap = cv2.VideoCapture(video_source); ret,frame = cap.read(); cap.release()
        if not ret: raise RuntimeError("Failed to read frame")
        scene, pixels_before, frame_to_show = extract_positions_from_frame(frame, depth_model, detector)
    else:
        from utils.generate_scene import generate_scene
        scene, pixels_before, frame_to_show = generate_scene(5), [], None
    if scene.numel()==0:
        print("No objects detected."); return
    scene_after = minimize_energy(scene, total_energy, steps=100, lr=0.1)
    pixels_after = pixels_before
    visualize_scene(scene.numpy(), scene_after.numpy())
    if frame_to_show is not None:
        visualize_overlay(frame_to_show, pixels_before, pixels_after)

if __name__=='__main__': main()