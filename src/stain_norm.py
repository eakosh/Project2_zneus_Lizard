import torch
import numpy as np
from torchstain import MacenkoNormalizer

class StainNormalizerMacenkoSafe:
    def __init__(self, reference_path):
        ref = np.array(Image.open(reference_path).convert("RGB"))
        ref_t = torch.from_numpy(ref).float().permute(2, 0, 1) / 255.0

        self.normalizer = MacenkoNormalizer()

        try:
            self.normalizer.fit(ref_t)
            self.ready = True
        except Exception:
            print("WARNING: Failed to fit Macenko normalizer. Using fallback.")
            self.ready = False

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not self.ready:
            return img

        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        norm, _, _ = self.normalizer.normalize(img_t)
        norm = (norm * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        return norm

