"""
Genuine Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet18
Trained for Knee Osteoporosis Classification.

Reference:
Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-Based Localization." ICCV 2017.

Target architecture: ResNet18
Target layer: model.layer4[-1] (final BasicBlock before global average pooling)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms

CLASS_NAMES = ['Normal', 'Osteopenia', 'Osteoporosis']
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transform():
    """Exact inference transform matching training validation/test splits."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(image_input):
    """
    Load and preprocess image for ResNet18 inference.
    
    Args:
        image_input: File path (str) or PIL Image.
    Returns:
        input_tensor: (1, 3, 224, 224) torch.FloatTensor
        original_pil: PIL.Image in RGB format
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at: {image_input}")
        original_pil = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        original_pil = image_input.convert('RGB')
    else:
        raise ValueError(f"Expected file path or PIL Image, got {type(image_input)}")

    transform = get_inference_transform()
    input_tensor = transform(original_pil).unsqueeze(0)  # Add batch dim: (1, 3, 224, 224)
    return input_tensor, original_pil


class GradCAM:
    """
    Grad-CAM implementation for PyTorch models.
    Captures forward feature activations and backward gradients at target_layer.
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model: PyTorch model in eval() mode.
            target_layer: nn.Module to hook. Defaults to model.layer4[-1] for ResNet18.
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Default to the final basic block in layer4 for ResNet18
            if hasattr(model, 'layer4'):
                self.target_layer = model.layer4[-1]
            else:
                raise ValueError("Model does not have 'layer4'. Please specify target_layer explicitly.")
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] is the gradient of the loss w.r.t. the layer's output activations
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Compute the Grad-CAM activation map for a given input tensor.
        
        Args:
            input_tensor: (1, 3, 224, 224) input tensor
            target_class: int (0, 1, 2) or None. If None, uses model's predicted class.
            
        Returns:
            dict containing:
                'cam': 2D numpy array (224, 224) in range [0, 1]
                'pred_class_idx': int
                'pred_class_name': str
                'target_class_idx': int
                'target_class_name': str
                'confidence': float (softmax probability of predicted class)
                'target_confidence': float (softmax probability of target class)
                'all_probs': list of floats for all classes
                'raw_logits': list of floats
                'activation_shape': tuple
                'gradient_shape': tuple
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Ensure gradient tracking for backward pass while keeping parameters frozen
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            self.model.zero_grad()

            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred_class = int(torch.argmax(logits, dim=1).item())

            if target_class is None:
                target_class = pred_class
            else:
                target_class = int(target_class)

            # Score for the target class
            score = logits[0, target_class]
            score.backward()

        # Capture hooked tensors
        activations = self.activations.detach()  # Shape: (1, channels, H, W)
        gradients = self.gradients.detach()      # Shape: (1, channels, H, W)

        act_shape = tuple(activations.shape)
        grad_shape = tuple(gradients.shape)

        # Global average pooling of gradients: weights alpha_k^c
        # Shape: (1, channels, 1, 1)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of forward activation maps
        # Shape: (1, 1, H, W)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # Apply ReLU to retain only positive influences on target score
        cam = F.relu(cam)

        # Interpolate to 224x224
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return {
            'cam': cam,
            'pred_class_idx': pred_class,
            'pred_class_name': CLASS_NAMES[pred_class],
            'target_class_idx': target_class,
            'target_class_name': CLASS_NAMES[target_class],
            'confidence': float(probs[pred_class]),
            'target_confidence': float(probs[target_class]),
            'all_probs': [float(p) for p in probs],
            'raw_logits': [float(l) for l in logits.detach().cpu().numpy()[0]],
            'activation_shape': act_shape,
            'gradient_shape': grad_shape,
        }

    @staticmethod
    def overlay_heatmap(original_pil, cam, alpha=0.4, colormap_name='jet'):
        """
        Render a transparent heatmap overlay on the original X-ray.
        
        Args:
            original_pil: PIL.Image (RGB)
            cam: 2D numpy array in [0, 1]
            alpha: float transparency factor for heatmap overlay (0.0 to 1.0)
            colormap_name: str (e.g., 'jet', 'turbo', 'inferno')
        Returns:
            heatmap_pil: PIL.Image of colored heatmap resized to original image dimensions
            overlay_pil: PIL.Image of blended original image + heatmap
        """
        orig_w, orig_h = original_pil.size

        # Resize CAM to match original image dimensions exactly
        cam_image = Image.fromarray((cam * 255).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.BILINEAR)
        cam_resized = np.array(cam_image, dtype=np.float32) / 255.0

        # Apply colormap
        try:
            cmap = matplotlib.colormaps[colormap_name]
        except Exception:
            cmap = cm.get_cmap(colormap_name)
        heatmap_rgba = cmap(cam_resized)  # Shape: (orig_h, orig_w, 4), values in [0, 1]
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_rgb, mode='RGB')

        # Blend with original X-ray
        orig_np = np.array(original_pil, dtype=np.float32)
        heatmap_np = np.array(heatmap_pil, dtype=np.float32)
        blended_np = (1.0 - alpha) * orig_np + alpha * heatmap_np
        blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(blended_np, mode='RGB')

        return heatmap_pil, overlay_pil


def load_trained_resnet18(checkpoint_path=None, device=None):
    """
    Helper to load the trained ResNet18 model and frozen checkpoint.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if checkpoint_path is None:
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        checkpoint_path = os.path.join(workspace_root, 'models', 'checkpoints', 'resnet18', 'best_model.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    from torchvision import models
    import torch.nn as nn

    # Build exact architecture used in training
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model, ckpt
