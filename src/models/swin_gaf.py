"""
SwinGAF v2: Swin-based Gated Attention Fusion
Uses entropy regularization to prevent gate collapse.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinGAF(nn.Module):
    def __init__(self, scale_dims=[384, 768, 768, 768], proj_dim=256,
                 num_classes=2, num_heads=4, num_csa_layers=2, dropout=0.3, **kwargs):
        super().__init__()
        self.num_scales = len(scale_dims)
        
        # Project each scale to common dim
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
            for d in scale_dims
        ])
        
        # Gate network - outputs logits, NOT softmax
        self.gate_net = nn.Sequential(
            nn.Linear(proj_dim * self.num_scales, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_scales),
        )
        
        # Cross-scale attention
        self.csa_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=proj_dim, nhead=num_heads, dim_feedforward=proj_dim*2,
                dropout=dropout, activation='gelu', batch_first=True
            ) for _ in range(num_csa_layers)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(proj_dim // 2, num_classes),
        )
        
        self._gate_logits = None  # store for regularization

    def forward(self, features_list, return_attention=False):
        # Project
        projected = [proj(feat) for proj, feat in zip(self.projectors, features_list)]
        scale_feat = torch.stack(projected, dim=1)  # [B, S, D]
        
        # Compute gates
        B, S, D = scale_feat.shape
        gate_input = scale_feat.reshape(B, S * D)
        gate_logits = self.gate_net(gate_input)  # [B, S]
        self._gate_logits = gate_logits
        gate_weights = F.softmax(gate_logits, dim=1)  # [B, S]
        
        # Apply gates
        gated = scale_feat * gate_weights.unsqueeze(-1)
        
        # Cross-scale attention
        x = gated
        for layer in self.csa_layers:
            x = layer(x)
        
        # Weighted pooling
        pooled = (x * gate_weights.unsqueeze(-1)).sum(dim=1)
        
        logits = self.classifier(pooled)
        
        if return_attention:
            info = {
                "gates": gate_weights.detach(),
                "scale_contributions": gate_weights.mean(dim=0).detach(),
            }
            return logits, info
        return logits
    
    def gate_entropy_loss(self):
        """Encourage non-collapsed gates. Add to training loss."""
        if self._gate_logits is None:
            return 0.0
        probs = F.softmax(self._gate_logits, dim=1)  # [B, S]
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        max_entropy = torch.log(torch.tensor(float(self.num_scales)))
        return -entropy / max_entropy  # negative = maximize entropy
    
    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
