"""Loss functions: Focal Loss for class imbalance."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing."""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # One-hot with smoothing
        targets_oh = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_oh = targets_oh * (1 - self.smoothing) + self.smoothing / n_classes
        
        loss = -(targets_oh * log_probs).sum(dim=-1)
        return loss.mean()
