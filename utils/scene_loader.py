import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn

FRAME_SIZE = (256, 256)
_MIDAS_MEAN = [0.485, 0.456, 0.406]
_MIDAS_STD  = [0.229, 0.224, 0.225]

def load_depth_model():
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
    model.eval()
    return model

def load_detector():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def extract_positions_from_frame(frame, depth_model, detector, device='cpu'):
    frame_resized = cv2.resize(frame, FRAME_SIZE)
    depth_transform = T.Compose([
        T.ToPILImage(), T.ToTensor(),
        T.Normalize(mean=_MIDAS_MEAN, std=_MIDAS_STD)
    ])
    input_img = depth_transform(frame_resized).to(device)
    with torch.no_grad():
        depth_out = depth_model(input_img.unsqueeze(0))
        depth = depth_out['out'][0] if isinstance(depth_out, dict) else depth_out[0]
        depth = depth.cpu().numpy()
    img_tensor = T.ToTensor()(frame_resized).to(device)
    with torch.no_grad():
        outputs = detector([img_tensor])[0]
    positions, pixel_coords = [], []
    for mask, score in zip(outputs['masks'], outputs['scores']):
        if score < 0.7: continue
        m = mask[0].cpu().numpy() > 0.5
        ys, xs = m.nonzero()
        if xs.size == 0: continue
        z = depth[ys, xs].mean()
        x3 = (xs.mean() - FRAME_SIZE[0]/2) * z / FRAME_SIZE[0]
        y3 = (ys.mean() - FRAME_SIZE[1]/2) * z / FRAME_SIZE[1]
        positions.append([x3, y3, float(z)])
        pixel_coords.append((float(xs.mean()), float(ys.mean())))
    if not positions:
        return torch.empty(0,3), [], frame_resized
    return torch.tensor(positions, dtype=torch.float32), pixel_coords, frame_resized
