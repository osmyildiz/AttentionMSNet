"""
Grad-CAM++ wrapper for AttentionMS-Net interpretability.

Generates brain-masked heatmaps showing which spatial regions
the model focuses on for classification decisions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAMPlusPlus:
    """Grad-CAM++ implementation for convolutional models.
    
    Extracts activation maps and gradients from a target layer,
    then computes weighted combination to produce class-discriminative
    localization heatmaps.
    
    Args:
        model: PyTorch model (must be in eval mode)
        target_layer: Layer to extract activations from
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM++ heatmap for an input image.
        
        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)
            target_class: Class index to explain (None = predicted class)
            
        Returns:
            heatmap: Normalized heatmap as numpy array (H, W), range [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        # Grad-CAM++ weights (second-order gradients)
        grads = self.gradients
        acts = self.activations
        
        # Alpha coefficients (Grad-CAM++ specific)
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = torch.sum(acts, dim=(2, 3), keepdim=True)
        
        alpha_numer = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3 + 1e-7
        alpha = alpha_numer / alpha_denom
        
        weights = torch.sum(alpha * F.relu(grads), dim=(2, 3), keepdim=True)
        
        # Weighted combination
        heatmap = torch.sum(weights * acts, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        
        # Resize to input dimensions
        heatmap = F.interpolate(
            heatmap, size=input_tensor.shape[2:],
            mode='bilinear', align_corners=False
        )
        
        # Normalize to [0, 1]
        heatmap = heatmap.squeeze().cpu().numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap


def apply_brain_mask(heatmap, image, erosion_kernel=5):
    """Apply Otsu thresholding + erosion to mask non-brain regions.
    
    Removes skull/background artifacts from Grad-CAM heatmaps
    by masking areas outside the brain parenchyma.
    
    Args:
        heatmap: Raw heatmap (H, W), range [0, 1]
        image: Original grayscale image as numpy array
        erosion_kernel: Size of erosion kernel (default 5)
        
    Returns:
        masked_heatmap: Brain-masked heatmap (H, W)
    """
    import cv2
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Otsu thresholding for brain mask
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Erode to remove skull boundary
    kernel = np.ones((erosion_kernel, erosion_kernel), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # Apply mask
    mask_float = mask.astype(np.float32) / 255.0
    masked = heatmap * mask_float
    
    return masked
