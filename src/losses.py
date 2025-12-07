import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        union = torch.sum(probs + targets_one_hot, dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for class imbalance"""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class ComboLoss(nn.Module):
    """Combined CE + Focal + Dice loss"""
    def __init__(self, gamma=2.0, ce_weight=0.3, focal_weight=0.5, dice_weight=0.2):
        super().__init__()
        self.focal = FocalLoss(gamma)
        self.ce = nn.CrossEntropyLoss()
        self.dice = SoftDiceLoss()

        self.w_focal = focal_weight
        self.w_ce = ce_weight
        self.w_dice = dice_weight

    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets)
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)

        return (
            self.w_focal * loss_focal +
            self.w_ce * loss_ce +
            self.w_dice * loss_dice
        )
