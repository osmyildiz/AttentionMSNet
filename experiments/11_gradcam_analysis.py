"""
Step 05: Grad-CAM++ Explainability
===================================
PURPOSE: Visualize WHERE the model looks when making decisions.
Uses best model from Step 03/04.

Outputs:
  - Per-class heatmap grids (3 classes × N samples)
  - Progression figure (Non-Dem → Moderate side by side)
  - Saved individual heatmaps for paper figures

No training - inference only. Runs fast (~10-15 min).
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.helpers import set_seed, load_config, get_device, ensure_dirs
from src.models.hybrid_model import HybridAttentionNet

print("=" * 70)
print("  Step 05: Grad-CAM++ Explainability")
print("=" * 70)

cfg = load_config("configs/config.yaml")
set_seed(cfg["project"]["seed"])
device = get_device()

NUM_CLASSES = cfg["data"]["num_classes"]
CLASS_NAMES = cfg["data"]["class_names"]
IMG_SIZE = cfg["data"]["image_size"]
BACKBONE = cfg["model"]["backbone"]
DROPOUT = cfg["model"]["dropout"]

ensure_dirs("outputs/gradcam", "outputs/figures")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ToTensorV2(),
])


# ══════════════════════════════════════════════════════════
# 1. Grad-CAM++ Implementation
# ══════════════════════════════════════════════════════════

class GradCAMPlusPlus:
    """
    Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks.
    Chattopadhay et al., WACV 2018.

    Works by computing weighted combination of positive partial derivatives
    of the last convolutional layer's activations.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM++ heatmap.

        Args:
            input_tensor: [1, C, H, W] normalized image
            target_class: class index (None = use predicted class)

        Returns:
            heatmap: [H, W] numpy array, values 0-1
            predicted_class: int
            confidence: float
        """
        self.model.eval()
        input_tensor = input_tensor.to(device).requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()
        confidence = probs[0, target_class].item()

        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Grad-CAM++ computation
        gradients = self.gradients[0]   # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Second and third derivatives approximation
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        # Alpha coefficients (Grad-CAM++ specific)
        denom = 2 * grad_2 + (activations * grad_3).sum(dim=(1, 2), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grad_2 / denom

        # Weights: positive gradients weighted by alpha
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))  # [C]

        # Weighted combination of activations
        heatmap = (weights.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=0)  # [H, W]
        heatmap = F.relu(heatmap)

        # Normalize to 0-1
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Upsample to input size
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(IMG_SIZE, IMG_SIZE),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        return heatmap, target_class, confidence


def get_target_layer(model, backbone_name):
    """Get the last convolutional layer for Grad-CAM."""
    if backbone_name == "efficientnet_b3":
        return model.features[-1]  # Last block of EfficientNet
    elif backbone_name == "resnet50":
        return model.layer4[-1]
    elif backbone_name == "densenet169":
        return model.features.denseblock4
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


def overlay_heatmap(image_np, heatmap, alpha=0.4, colormap=cm.jet):
    """Overlay heatmap on original image."""
    colored_heatmap = colormap(heatmap)[:, :, :3]  # [H, W, 3]
    overlay = (1 - alpha) * image_np + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def denormalize(tensor):
    """Convert normalized tensor back to displayable image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return img


# ══════════════════════════════════════════════════════════
# 2. Load Model
# ══════════════════════════════════════════════════════════

model_path = "outputs/models/arch_cnn_cbam_multiscale.pth"
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    print("Trying alternative paths...")
    for alt in ["outputs/models/C_CNN_CBAM_MultiScale.pth"]:
        if os.path.exists(alt):
            model_path = alt
            break
    else:
        print("ERROR: No trained hybrid model found. Run Step 03 first.")
        sys.exit(1)

print(f"Loading model: {model_path}")
model = HybridAttentionNet(
    backbone_name=BACKBONE,
    num_classes=NUM_CLASSES,
    pretrained=False,
    dropout=DROPOUT,
    use_attention=True,
    use_multiscale=True,
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

target_layer = get_target_layer(model, BACKBONE)
gradcam = GradCAMPlusPlus(model, target_layer)
print("Model loaded. Grad-CAM++ ready.")


# ══════════════════════════════════════════════════════════
# 3. Select Sample Images
# ══════════════════════════════════════════════════════════

SPLITS_DIR = Path(cfg["data"]["splits_dir"])
test_df = pd.read_csv(SPLITS_DIR / "test.csv")

SAMPLES_PER_CLASS = 5

selected = []
for label, cls in enumerate(CLASS_NAMES):
    class_imgs = test_df[test_df["class_name"] == cls]
    if len(class_imgs) >= SAMPLES_PER_CLASS:
        sampled = class_imgs.sample(n=SAMPLES_PER_CLASS, random_state=42)
    else:
        sampled = class_imgs  # Take all for minority class
    selected.append(sampled)

selected_df = pd.concat(selected, ignore_index=True)
print(f"\nSelected {len(selected_df)} images for Grad-CAM++ visualization:")
for cls in CLASS_NAMES:
    n = len(selected_df[selected_df["class_name"] == cls])
    print(f"  {cls:25s}  {n}")


# ══════════════════════════════════════════════════════════
# 4. Generate Heatmaps
# ══════════════════════════════════════════════════════════

print("\nGenerating heatmaps...")
results = []

for _, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Grad-CAM++"):
    img_path = row["path"]
    true_label = row["label"]
    class_name = row["class_name"]

    # Load and transform
    img = np.array(Image.open(img_path).convert("RGB"))
    transformed = val_transform(image=img)
    input_tensor = transformed["image"].unsqueeze(0)  # [1, C, H, W]

    # Original image for display
    orig_display = denormalize(transformed["image"])

    # Generate heatmap for TRUE class
    heatmap, pred_class, confidence = gradcam.generate(input_tensor, target_class=true_label)

    # Also generate for predicted class if different
    if pred_class != true_label:
        heatmap_pred, _, conf_pred = gradcam.generate(input_tensor, target_class=None)
    else:
        heatmap_pred = heatmap
        conf_pred = confidence

    overlay = overlay_heatmap(orig_display, heatmap)

    results.append({
        "path": img_path,
        "class_name": class_name,
        "true_label": true_label,
        "pred_label": pred_class,
        "confidence": confidence,
        "original": orig_display,
        "heatmap": heatmap,
        "overlay": overlay,
    })


# ══════════════════════════════════════════════════════════
# 5. Figure 1: Per-Class Heatmap Grid
# ══════════════════════════════════════════════════════════

print("\nGenerating figures...")

fig, axes = plt.subplots(NUM_CLASSES, SAMPLES_PER_CLASS * 2, figsize=(SAMPLES_PER_CLASS * 5, NUM_CLASSES * 3))

for row_idx, cls in enumerate(CLASS_NAMES):
    cls_results = [r for r in results if r["class_name"] == cls]
    for col_idx, res in enumerate(cls_results[:SAMPLES_PER_CLASS]):
        # Original
        ax_orig = axes[row_idx, col_idx * 2]
        ax_orig.imshow(res["original"])
        ax_orig.set_title(f"True: {cls.split()[0]}", fontsize=7)
        ax_orig.axis("off")

        # Overlay
        ax_over = axes[row_idx, col_idx * 2 + 1]
        ax_over.imshow(res["overlay"])
        correct = "✓" if res["pred_label"] == res["true_label"] else "✗"
        ax_over.set_title(f"Pred: {CLASS_NAMES[res['pred_label']].split()[0]} {correct}\n{res['confidence']:.2f}", fontsize=7)
        ax_over.axis("off")

    # Label row
    axes[row_idx, 0].set_ylabel(cls, fontsize=10, fontweight="bold", rotation=90, labelpad=15)

plt.suptitle("Grad-CAM++ Heatmaps by Class (Test Set)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/gradcam_per_class_grid.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: outputs/figures/gradcam_per_class_grid.png")


# ══════════════════════════════════════════════════════════
# 6. Figure 2: Disease Progression
# ══════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(NUM_CLASSES * 4, 7))

for col_idx, cls in enumerate(CLASS_NAMES):
    cls_results = [r for r in results if r["class_name"] == cls]
    if cls_results:
        best = cls_results[0]
        axes[0, col_idx].imshow(best["original"])
        axes[0, col_idx].set_title(cls, fontsize=10, fontweight="bold")
        axes[0, col_idx].axis("off")

        axes[1, col_idx].imshow(best["overlay"])
        axes[1, col_idx].set_title(f"Conf: {best['confidence']:.2f}", fontsize=9)
        axes[1, col_idx].axis("off")

axes[0, 0].set_ylabel("Original MRI", fontsize=11, fontweight="bold")
axes[1, 0].set_ylabel("Grad-CAM++", fontsize=11, fontweight="bold")

plt.suptitle("Attention Progression Across Dementia Stages", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/gradcam_progression.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: outputs/figures/gradcam_progression.png")


# ══════════════════════════════════════════════════════════
# 7. Save Individual Heatmaps
# ══════════════════════════════════════════════════════════

for i, res in enumerate(results):
    cls_short = res["class_name"].replace(" ", "_").lower()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(res["original"])
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    axes[1].imshow(res["heatmap"], cmap="jet")
    axes[1].set_title("Grad-CAM++ Heatmap")
    axes[1].axis("off")

    axes[2].imshow(res["overlay"])
    correct = "Correct" if res["pred_label"] == res["true_label"] else "Wrong"
    axes[2].set_title(f"Overlay ({correct}, conf={res['confidence']:.2f})")
    axes[2].axis("off")

    plt.suptitle(f"{res['class_name']} (Sample {i})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"outputs/gradcam/{cls_short}_sample_{i:02d}.png", dpi=200, bbox_inches="tight")
    plt.close()

print(f"  Saved {len(results)} individual heatmaps to outputs/gradcam/")


# ══════════════════════════════════════════════════════════
# 8. Summary Statistics
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Grad-CAM++ Summary")
print("=" * 60)

for cls in CLASS_NAMES:
    cls_results = [r for r in results if r["class_name"] == cls]
    correct = sum(1 for r in cls_results if r["pred_label"] == r["true_label"])
    avg_conf = np.mean([r["confidence"] for r in cls_results])
    print(f"  {cls:25s}  {correct}/{len(cls_results)} correct  avg_conf={avg_conf:.3f}")

print("\nOutputs:")
print("  outputs/figures/gradcam_per_class_grid.png    (paper Figure X)")
print("  outputs/figures/gradcam_progression.png       (paper Figure Y)")
print("  outputs/gradcam/*_sample_*.png                (individual)")
print("\nStep 05 DONE.")
