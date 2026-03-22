"""Backbone CNN models with feature extraction hooks."""
import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(name: str, pretrained: bool = True):
    """Get backbone model and its layer channel dimensions.
    
    Returns:
        model: backbone model
        layer_channels: dict mapping layer name -> output channels
        feature_dim: final feature dimension
    """
    if name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        # EfficientNet features are in model.features
        # Layer indices: [3]=48ch, [5]=96ch, [7]=232ch, [8]=384ch (before pool)
        layer_channels = {"layer2": 48, "layer3": 136, "layer4": 384}
        feature_dim = 1536
        return model, layer_channels, feature_dim

    elif name == "densenet169":
        weights = models.DenseNet169_Weights.DEFAULT if pretrained else None
        model = models.densenet169(weights=weights)
        layer_channels = {"layer2": 256, "layer3": 640, "layer4": 1664}
        feature_dim = 1664
        return model, layer_channels, feature_dim

    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        layer_channels = {"layer2": 512, "layer3": 1024, "layer4": 2048}
        feature_dim = 2048
        return model, layer_channels, feature_dim

    else:
        raise ValueError(f"Unknown backbone: {name}")
