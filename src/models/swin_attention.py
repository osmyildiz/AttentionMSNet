"""
SwinAttentionNet: Multi-Scale Attention-Enhanced Swin Transformer
=================================================================
Matches abstract: "attention-enhanced multi-scale deep learning framework"

Architecture:
  1. Swin-Tiny backbone (pretrained ImageNet)
  2. Multi-scale feature extraction (stages 2, 3, 4)
  3. CBAM attention per scale (channel + spatial)
  4. Multi-scale feature aggregation (adaptive fusion)
  5. Classification head with dropout

Key difference from Kaggle baseline (Tarik):
  - Tarik: pooler_output (768-d flat) → ML classifiers
  - Ours:  multi-scale stage outputs → CBAM → fusion → end-to-end DL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel, AutoImageProcessor


# ═══════════════════════════════════════════════════════════════
# CBAM: Convolutional Block Attention Module
# ═══════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """Channel attention: WHAT to focus on."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """x: (B, N, C) — sequence of token features"""
        # Permute to (B, C, N) for pooling
        x_perm = x.permute(0, 2, 1)
        avg_out = self.fc(self.avg_pool(x_perm).squeeze(-1))
        max_out = self.fc(self.max_pool(x_perm).squeeze(-1))
        attn = self.sigmoid(avg_out + max_out).unsqueeze(1)  # (B, 1, C)
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial attention: WHERE to focus on."""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """x: (B, N, C) — sequence of token features"""
        # Along channel dim
        avg_out = x.mean(dim=-1, keepdim=True)   # (B, N, 1)
        max_out = x.max(dim=-1, keepdim=True)[0]  # (B, N, 1)
        cat = torch.cat([avg_out, max_out], dim=-1)  # (B, N, 2)
        cat = cat.permute(0, 2, 1)  # (B, 2, N)
        attn = self.sigmoid(self.conv(cat))  # (B, 1, N)
        attn = attn.permute(0, 2, 1)  # (B, N, 1)
        return x * attn


class CBAM(nn.Module):
    """CBAM: Channel-Spatial attention for sequence data."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ═══════════════════════════════════════════════════════════════
# Multi-Scale Feature Aggregation
# ═══════════════════════════════════════════════════════════════

class MultiScaleAggregator(nn.Module):
    """
    Adaptive multi-scale feature fusion.
    Projects each scale to common dim, then learns fusion weights.
    """
    def __init__(self, scale_dims, target_dim=256):
        super().__init__()
        self.n_scales = len(scale_dims)

        # Per-scale projection to common dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, target_dim),
                nn.LayerNorm(target_dim),
                nn.GELU(),
            )
            for d in scale_dims
        ])

        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(self.n_scales) / self.n_scales)

    def forward(self, scale_features):
        """
        Args:
            scale_features: list of (B, D_i) tensors (one per scale, already pooled)
        Returns:
            fused: (B, target_dim)
        """
        projected = [proj(feat) for proj, feat in zip(self.projections, scale_features)]
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * p for w, p in zip(weights, projected))
        return fused


# ═══════════════════════════════════════════════════════════════
# SwinAttentionNet — Full Model
# ═══════════════════════════════════════════════════════════════

class SwinAttentionNet(nn.Module):
    """
    Multi-scale attention-enhanced Swin Transformer for AD classification.

    Pipeline:
        Input (224×224×3)
         → Swin-Tiny stages 1-4
         → Extract stages 2, 3, 4 hidden states
         → CBAM attention per stage
         → Global average pooling per stage
         → Multi-scale adaptive fusion
         → Classification head
    """

    # Swin-Tiny stage output dimensions
    STAGE_DIMS = {
        1: 96,    # stage 1: (B, 56×56, 96)   — too low-level, skip
        2: 384,   # stage 2: (B, 28×28, 192)   — spatial details
        3: 768,   # stage 3: (B, 14×14, 384)   — mid-level patterns
        4: 768,   # stage 4: (B, 7×7, 768)     — high-level semantics
    }

    def __init__(
        self,
        num_classes=2,
        model_name="microsoft/swin-tiny-patch4-window7-224",
        freeze_stages=(0, 1),
        fusion_dim=256,
        dropout=0.4,
        cbam_reduction=16,
    ):
        super().__init__()
        self.num_classes = num_classes

        # 1. Swin backbone
        self.swin = SwinModel.from_pretrained(model_name)

        # Freeze early stages (embeddings + specified stages)
        self._freeze_stages(freeze_stages)

        # 2. CBAM attention for each extracted stage
        self.use_stages = [2, 3, 4]
        self.cbam_modules = nn.ModuleDict({
            f"stage_{s}": CBAM(self.STAGE_DIMS[s], reduction=cbam_reduction)
            for s in self.use_stages
        })

        # 3. Multi-scale aggregation
        scale_dims = [self.STAGE_DIMS[s] for s in self.use_stages]
        self.aggregator = MultiScaleAggregator(scale_dims, target_dim=fusion_dim)

        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def _freeze_stages(self, freeze_stages):
        """Freeze embeddings and specified Swin stages."""
        # Always freeze patch embeddings
        for param in self.swin.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified encoder layers
        # Swin-Tiny has 4 stages with [2, 2, 6, 2] layers
        stage_layer_counts = [2, 2, 6, 2]
        layer_idx = 0
        for stage_id, n_layers in enumerate(stage_layer_counts):
            if stage_id in freeze_stages:
                for i in range(n_layers):
                    for param in self.swin.encoder.layers[layer_idx + i].parameters():
                        param.requires_grad = False
            layer_idx += n_layers

    def extract_multiscale(self, pixel_values):
        """
        Extract multi-scale features from Swin stages.

        Returns dict: {stage_id: (B, N_tokens, C)}
        """
        outputs = self.swin(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states: tuple of (B, N_i, C_i) for each stage output
        # Index mapping: hidden_states[0] = embeddings output
        #                hidden_states[1] = after stage 1
        #                hidden_states[2] = after stage 2
        #                hidden_states[3] = after stage 3
        #                hidden_states[4] = after stage 4 (= last_hidden_state)
        hidden_states = outputs.hidden_states

        stage_features = {}
        for s in self.use_stages:
            stage_features[s] = hidden_states[s]  # (B, N_s, C_s)

        return stage_features

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (B, 3, 224, 224) preprocessed images
        Returns:
            logits: (B, num_classes)
            attention_info: dict with per-stage attention maps (for XAI)
        """
        # 1. Multi-scale extraction
        stage_features = self.extract_multiscale(pixel_values)

        # 2. CBAM attention per stage + global pooling
        pooled_features = []
        attention_info = {}

        for s in self.use_stages:
            feat = stage_features[s]           # (B, N, C)
            refined = self.cbam_modules[f"stage_{s}"](feat)  # (B, N, C)
            pooled = refined.mean(dim=1)       # (B, C) — global average pool
            pooled_features.append(pooled)

            # Store attention difference for XAI
            attention_info[f"stage_{s}"] = {
                "raw": feat.detach(),
                "refined": refined.detach(),
            }

        # 3. Multi-scale fusion
        fused = self.aggregator(pooled_features)  # (B, fusion_dim)

        # 4. Classification
        logits = self.classifier(fused)

        return logits, attention_info

    def get_fusion_weights(self):
        """Get learned scale importance (for paper Table)."""
        w = F.softmax(self.aggregator.fusion_weights, dim=0)
        return {f"stage_{s}": w[i].item() for i, s in enumerate(self.use_stages)}


# ═══════════════════════════════════════════════════════════════
# Focal Loss (from abstract: "focal loss with adaptive class weighting")
# ═══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples, focuses on hard ones.
    Particularly useful for Moderate Dementia (very few samples).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ═══════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SwinAttentionNet(num_classes=2).to(device)
    x = torch.randn(4, 3, 224, 224).to(device)

    logits, attn = model(x)
    print(f"Input:  {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Fusion weights: {model.get_fusion_weights()}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
