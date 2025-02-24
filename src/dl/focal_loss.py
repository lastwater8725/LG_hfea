import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """ Focal Loss for binary classification """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """ Compute Focal Loss """
        probas = torch.sigmoid(logits)  # Sigmoid 적용
        targets = targets.float()
        p_t = probas * targets + (1 - probas) * (1 - targets)
        focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * (1 - p_t) ** self.gamma
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none') * focal_weight
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
