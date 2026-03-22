"""HybridAttentionNet: The full proposed architecture.
Backbone + CBAM Attention + Multi-Scale Fusion + Classification Head.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from .attention import CBAM
from .multiscale import MultiScaleFusion


class HybridAttentionNet(nn.Module):
    """
    Hybrid CNN with:
    1. Pretrained backbone (EfficientNet-B3 / DenseNet-169 / ResNet-50)
    2. CBAM attention after final feature layer
    3. Multi-scale feature fusion from intermediate layers
    4. Classification head with dropout
    """
    def __init__(self, backbone_name="efficientnet_b3", num_classes=4,
                 pretrained=True, dropout=0.3, use_attention=True,
                 use_multiscale=True, attention_reduction=16):
        super().__init__()
        self.use_attention = use_attention
        self.use_multiscale = use_multiscale
        self.backbone_name = backbone_name

        # --- Build backbone ---
        if backbone_name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b3(weights=weights)
            self.features = base.features
            self.pool = base.avgpool
            feature_dim = 1536
            ms_channels = [96, 232, 384]
            self.ms_indices = [4, 6, 7]  # Layer 4: 96ch, Layer 6: 232ch, Layer 7: 384ch
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.pool = nn.AdaptiveAvgPool2d(1)
            feature_dim = 2048
            ms_channels = [512, 1024, 2048]
        elif backbone_name == "densenet169":
            weights = models.DenseNet169_Weights.DEFAULT if pretrained else None
            base = models.densenet169(weights=weights)
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            feature_dim = 1664
            ms_channels = [256, 640, 1664]
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # --- Attention ---
        if use_attention:
            self.cbam = CBAM(feature_dim, reduction=attention_reduction)

        # --- Multi-Scale Fusion ---
        ms_out = 256
        if use_multiscale:
            self.multiscale = MultiScaleFusion(ms_channels, out_channels=ms_out)
            classifier_dim = feature_dim + ms_out * len(ms_channels)
        else:
            classifier_dim = feature_dim

        # --- Classifier Head ---
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def _extract_features_efficientnet(self, x):
        """Extract multi-scale features from EfficientNet."""
        intermediates = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if self.use_multiscale and i in self.ms_indices:
                intermediates.append(x)
        return x, intermediates

    def _extract_features_resnet(self, x):
        """Extract multi-scale features from ResNet."""
        x = self.layer0(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f4, [f2, f3, f4]

    def forward(self, x):
        # Extract features
        if self.backbone_name == "efficientnet_b3":
            feat, intermediates = self._extract_features_efficientnet(x)
        elif self.backbone_name == "resnet50":
            feat, intermediates = self._extract_features_resnet(x)
        else:
            # DenseNet - simplified
            feat = self.features(x)
            intermediates = [feat, feat, feat]  # Placeholder

        # Apply attention
        if self.use_attention:
            feat = self.cbam(feat)

        # Global average pooling
        pooled = self.pool(feat).flatten(1)

        # Multi-scale fusion
        if self.use_multiscale:
            ms_feat = self.multiscale(intermediates)
            combined = torch.cat([pooled, ms_feat], dim=1)
        else:
            combined = pooled

        # Classify
        return self.classifier(combined)

    def extract_features(self, x):
        """Extract deep features (before classifier) for ensemble."""
        if self.backbone_name == "efficientnet_b3":
            feat, intermediates = self._extract_features_efficientnet(x)
        elif self.backbone_name == "resnet50":
            feat, intermediates = self._extract_features_resnet(x)
        else:
            feat = self.features(x)
            intermediates = [feat, feat, feat]

        if self.use_attention:
            feat = self.cbam(feat)

        pooled = self.pool(feat).flatten(1)

        if self.use_multiscale:
            ms_feat = self.multiscale(intermediates)
            return torch.cat([pooled, ms_feat], dim=1)
        return pooled
