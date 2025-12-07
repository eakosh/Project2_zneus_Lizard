import os
import cv2
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from PIL import Image
import albumentations as A
import pandas as pd
from stain_norm import StainNormalizerMacenkoSafe


PATCH_SIZE = 224
STRIDE = 224
AUGS_PER_PATCH = 2
OUTPUT_DIR = "./patches"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG1 = "/kaggle/input/lizard-dataset/lizard_images1/Lizard_Images1"
IMG2 = "/kaggle/input/lizard-dataset/lizard_images2/Lizard_Images2"
LABELS = "/kaggle/input/lizard-dataset/lizard_labels/Lizard_Labels/Labels"
INFO_CSV = "/kaggle/input/lizard-dataset/lizard_labels/Lizard_Labels/info.csv"

train_augs = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.10,
        rotate_limit=20,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5
    ),

    A.OneOf([
        A.ColorJitter(0.2, 0.2, 0.2, 0.05),
        A.HueSaturationValue(10, 15, 10),
    ], p=0.4),

    A.OneOf([
        A.GaussNoise(),
        A.GaussianBlur(blur_limit=(3, 3)),
        A.MedianBlur(blur_limit=3),
    ], p=0.3),

    A.OneOf([
        A.ElasticTransform(alpha=20, sigma=20, border_mode=cv2.BORDER_CONSTANT),
        A.GridDistortion(distort_limit=0.05, border_mode=cv2.BORDER_CONSTANT),
    ], p=0.2),

    A.OneOf([
        A.RGBShift(10, 10, 10),
        A.RandomBrightnessContrast(0.15, 0.15),
    ], p=0.3)
], additional_targets={'mask': 'mask'})


def load_image(fname):
    p1 = os.path.join(IMG1, fname + ".png")
    p2 = os.path.join(IMG2, fname + ".png")
    if os.path.exists(p1): 
        return np.array(Image.open(p1))
    if os.path.exists(p2): 
        return np.array(Image.open(p2))
    raise FileNotFoundError


def load_semantic_mask(fname):
    mat = sio.loadmat(os.path.join(LABELS, fname + ".mat"))
    inst_map = mat["inst_map"]  # instance ids
    nuclei_id = np.squeeze(mat["id"])     # shape (N,)
    classes = np.squeeze(mat["class"])    # shape (N,)

    sem_mask = np.zeros_like(inst_map, dtype=np.uint8)

    for nucleus_id, cls in zip(nuclei_id, classes):
        sem_mask[inst_map == nucleus_id] = cls

    return sem_mask


def save_patch(img_patch, mask_patch, split, base_name):
    folder_img = os.path.join(OUTPUT_DIR, split, "img")
    folder_msk = os.path.join(OUTPUT_DIR, split, "mask")
    os.makedirs(folder_img, exist_ok=True)
    os.makedirs(folder_msk, exist_ok=True)

    Image.fromarray(img_patch).save(os.path.join(folder_img, base_name + ".png"))
    Image.fromarray(mask_patch).save(os.path.join(folder_msk, base_name + ".png"))


stain_norm = StainNormalizerMacenkoSafe("./stain_reference.png")

df = pd.read_csv(INFO_CSV)

for idx, row in tqdm(df.iterrows(), total=len(df)):
    fname = row["Filename"]
    split_id = row["Split"]

    split = "train" if split_id == 1 else "val" if split_id == 2 else "test"

    try:
        img = load_image(fname)
        img = stain_norm(img) 
        mask = load_semantic_mask(fname)
    except:
        continue

    H, W = mask.shape

    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):

            img_patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            msk_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            base = f"{fname}_{y}_{x}"

            save_patch(img_patch, msk_patch, split, base)

            if split == "train":
                for k in range(AUGS_PER_PATCH):
                    augmented = train_augs(image=img_patch, mask=msk_patch)
                    img_aug = augmented["image"]
                    msk_aug = augmented["mask"]

                    save_patch(img_aug, msk_aug, split, base + f"_aug{k}")
