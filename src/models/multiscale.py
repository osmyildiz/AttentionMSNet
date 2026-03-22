"""Multi-Scale Feature Fusion (FPN-style).
Extracts features from multiple backbone layers and fuses them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    """Fuse features from multiple backbone layers via lateral connections."""
    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: List of channel dims from backbone layers
                              e.g., [128, 256, 512] for layer2, layer3, layer4
            out_channels: Unified output channel dimension
        """
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.smooth = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feature_maps):
        """
        Args:
            feature_maps: List of feature maps from backbone layers
        Returns:
            Fused feature vector (B, out_channels * len(feature_maps))
        """
        # Lateral connections
        laterals = [l(f) for l, f in zip(self.laterals, feature_maps)]

        # Top-down pathway (add upsampled higher-level to lower-level)
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='nearest')
            laterals[i-1] = laterals[i-1] + up

        # Smooth and pool
        outputs = [self.pool(s(l)).flatten(1) for s, l in zip(self.smooth, laterals)]

        return torch.cat(outputs, dim=1)
