# predict.py
"""
Inference script — run diagnosis on a new patient scan.
Uses the latest global model from FL server.
Integrated with FastAPI backend.
"""
import torch
import numpy as np
from PIL import Image
import cv2
from config import cfg
from model import get_model
from dataset import get_transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load model once at startup (singleton pattern)
_model_cache = None

def load_global_model(model_path: str = cfg.global_model_path):
    """Load the latest FL global model. Cached after first load."""
    global _model_cache
    if _model_cache is None:
        model = get_model(cfg.device)
        checkpoint = torch.load(model_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        _model_cache = model
        print(f"[Predict] Global model loaded from {model_path}")
    return _model_cache


@torch.no_grad()
def predict_single(image_path: str) -> dict:
    """
    Run inference on a single chest X-ray image.
    Returns diagnosis, confidence, and class probabilities.
    """
    model = load_global_model()
    transform = get_transforms("test")
    
    # Load and preprocess
    if image_path.endswith('.dcm'):
        import pydicom
        dcm = pydicom.dcmread(image_path)
        arr = dcm.pixel_array.astype(np.float32)
        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
        image = np.stack([arr, arr, arr], axis=-1)
    else:
        image = np.array(Image.open(image_path).convert('RGB'))
    
    transformed = transform(image=image)
    tensor = transformed['image'].unsqueeze(0).to(cfg.device)
    
    # Inference
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = probs.argmax()
    
    return {
        "diagnosis": cfg.class_names[pred_idx],
        "confidence": round(float(probs[pred_idx]) * 100, 1),
        "probabilities": {
            name: round(float(prob) * 100, 1)
            for name, prob in zip(cfg.class_names, probs)
        },
        "model_version": "FL Global Model",
        "privacy_guarantee": f"ε={cfg.dp_target_epsilon}",
    }


def predict_with_gradcam(image_path: str) -> dict:
    """
    Predict + generate Grad-CAM heatmap showing 
    which regions of the X-ray influenced the diagnosis.
    Clinically useful — shows doctors where the AI 'looked'.
    """
    model = load_global_model()
    
    # Hook to capture gradients and activations
    gradients = []
    activations = []
    
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    # Register hooks on last conv layer
    target_layer = model.features[-1][-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass (with gradients this time)
    transform = get_transforms("test")
    image = np.array(Image.open(image_path).convert('RGB'))
    tensor = transform(image=image)['image'].unsqueeze(0).to(cfg.device)
    
    model.zero_grad()
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze()
    pred_idx = probs.argmax()
    
    # Backward pass for predicted class
    logits[0, pred_idx].backward()
    
    fh.remove(); bh.remove()
    
    # Compute Grad-CAM
    grads = gradients[0].cpu().numpy()[0]       # (C, H, W)
    acts  = activations[0].cpu().numpy()[0]     # (C, H, W)
    
    weights = grads.mean(axis=(1, 2))           # Global average pooling
    cam = (weights[:, None, None] * acts).sum(axis=0)
    cam = np.maximum(cam, 0)                    # ReLU
    cam = cv2.resize(cam, (cfg.image_size, cfg.image_size))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Overlay on original image
    original = cv2.resize(image, (cfg.image_size, cfg.image_size))
    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    overlaid = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    
    return {
        "diagnosis": cfg.class_names[pred_idx.item()],
        "confidence": round(float(probs[pred_idx]) * 100, 1),
        "gradcam_image": overlaid,  # numpy array — save or encode
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image to predict on")
    args = parser.parse_args()
    print(predict_single(args.image))
