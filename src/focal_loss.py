# src/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probas = torch.sigmoid(logits)
        targets = targets.float()
        p_t = probas * targets + (1 - probas) * (1 - targets)
        focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * (1 - p_t) ** self.gamma
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none') * focal_weight

        return loss.mean() if self.reduction == 'mean' else loss.sum()
