import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transforms():
    """Augmentations for training (geometric + color + noise)"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=20,
            border_mode=cv2.BORDER_CONSTANT,
            value=0, mask_value=0, p=0.5
        ),

        A.OneOf([
            A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=1.0),
            A.HueSaturationValue(10, 15, 10, p=1.0)
        ], p=0.4),

        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
            A.GaussianBlur(3, p=1.0),
            A.MedianBlur(3, p=1.0),
        ], p=0.3),

        A.OneOf([
            A.ElasticTransform(alpha=20, sigma=20, alpha_affine=10,
                               border_mode=cv2.BORDER_CONSTANT,
                               value=0, mask_value=0, p=1.0),
            A.GridDistortion(distort_limit=0.05,
                             border_mode=cv2.BORDER_CONSTANT,
                             value=0, mask_value=0, p=1.0),
        ], p=0.2),

        A.OneOf([
            A.RGBShift(10, 10, 10, p=1.0),
            A.RandomBrightnessContrast(0.15, 0.15, p=1.0),
        ], p=0.3),

        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Only normalization for validation"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
