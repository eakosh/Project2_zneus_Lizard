import os
from typing import Optional, Dict, List, Literal
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
import torch
from torch.utils.data import Dataset
import staintools


def load_image(images_path_1: str, images_path_2: str, filename: str) -> np.ndarray:
    name = f"{filename}.png"

    p1 = os.path.join(images_path_1, name)
    if os.path.exists(p1):
        return np.array(Image.open(p1).convert("RGB"))

    p2 = os.path.join(images_path_2, name)
    if os.path.exists(p2):
        return np.array(Image.open(p2).convert("RGB"))

    raise FileNotFoundError(name)


def load_label(labels_path: str, filename: str) -> Dict[str, np.ndarray]:
    mat_path = os.path.join(labels_path, f"{filename}.mat")
    label = sio.loadmat(mat_path)

    return {
        "inst_map": label["inst_map"].astype(np.int32),
        "id": label["id"].flatten().astype(np.int32),
        "class": label["class"].flatten().astype(np.int64),
    }


def make_type_map(inst_map: np.ndarray,
                  ids: np.ndarray,
                  classes: np.ndarray) -> np.ndarray:
    type_map = np.zeros_like(inst_map, dtype=np.int64)
    for nid, c in zip(ids, classes):
        type_map[inst_map == nid] = c
    return type_map


def is_blank(img: np.ndarray, threshold: float = 0.9) -> bool:
    gray = np.mean(img, axis=2)
    blank_ratio = np.mean(gray > 225)
    return blank_ratio > threshold


class StainNormalizer:
    def __init__(self, ref_img: np.ndarray):
        self.normalizer = staintools.StainNormalizer(method='macenko')
        target = staintools.LuminosityStandardizer.standardize(ref_img)
        self.normalizer.fit(target)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = staintools.LuminosityStandardizer.standardize(img)
        return self.normalizer.transform(img)


class LizardDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: int,
        patch_size: int = 256,
        stride: Optional[int] = None,
        stain_reference_path: str = "./data/stain_reference.png",
        transform=None,
        blank_threshold: float = 0.90
    ):
        super().__init__()

        self.data_root = data_root
        self.path_images1 = os.path.join(data_root, "lizard_images1", "Lizard_Images1")
        self.path_images2 = os.path.join(data_root, "lizard_images2", "Lizard_Images2")
        self.path_labels = os.path.join(data_root, "lizard_labels", "Lizard_Labels", "Labels")
        self.path_info_csv = os.path.join(data_root, "lizard_labels", "Lizard_Labels", "info.csv")

        info = pd.read_csv(self.path_info_csv)
        self.info = info[info["Split"] == split].reset_index(drop=True)

        self.filenames = self.info["Filename"].tolist()

        self.patch_size = patch_size
        self.stride = stride if stride else patch_size
        self.transform = transform
        self.blank_threshold = blank_threshold

        ref_img = np.array(Image.open(stain_reference_path).convert("RGB"))
        self.stain_norm = StainNormalizer(ref_img)

        self.patch_index = []
        self.build_index()

    def build_index(self):
        ps, st = self.patch_size, self.stride

        for i, fname in enumerate(self.filenames):
            label = load_label(self.path_labels, fname)
            H, W = label["inst_map"].shape

            ys = list(range(0, max(H - ps + 1, 1), st))
            xs = list(range(0, max(W - ps + 1, 1), st))

            if ys[-1] != H - ps:
                ys.append(H - ps)
            if xs[-1] != W - ps:
                xs.append(W - ps)

            for y in ys:
                for x in xs:
                    self.patch_index.append((i, fname, y, x))

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        img_idx, fname, y0, x0 = self.patch_index[idx]

        img = load_image(self.path_images1, self.path_images2, fname)
        img = self.stain_norm(img)

        patch = img[y0:y0+self.patch_size, x0:x0+self.patch_size]

        if is_blank(patch, self.blank_threshold):
            return self.__getitem__((idx + 1) % len(self.patch_index))

        lbl = load_label(self.path_labels, fname)
        inst = lbl["inst_map"]
        ids = lbl["id"]
        cls = lbl["class"]

        type_map_full = make_type_map(inst, ids, cls)
        type_patch = type_map_full[y0:y0+self.patch_size, x0:x0+self.patch_size]

        if self.transform:
            out = self.transform(image=patch, mask=type_patch)
            image = out["image"]
            type_map = out["mask"]
        else:
            image = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1)
            type_map = torch.from_numpy(type_patch)

        return {
            "image": image,
            "mask": type_map.long(),
            "filename": fname,
        }
