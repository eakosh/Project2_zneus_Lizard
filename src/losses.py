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
    """Focal loss for class imbalance, optionally alpha-weighted per class"""
    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma

        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.clone().float())

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.class_weights is None:
            return focal.mean()

        alpha = self.class_weights[targets]
        return (alpha * focal).sum() / alpha.sum().clamp_min(1e-8)


class ComboLoss(nn.Module):
    """Combined CE + Focal + Dice loss"""
    def __init__(self, gamma=2.0, ce_weight=0.3, focal_weight=0.5, dice_weight=0.2,
                 class_weights=None):
        super().__init__()
        self.focal = FocalLoss(gamma, class_weights=class_weights)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
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
