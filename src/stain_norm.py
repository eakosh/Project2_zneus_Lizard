import numpy as np
import staintools


class StainNormalizerMacenkoSafe:
    def __init__(self, reference_path):
        ref = staintools.read_image(reference_path)
        ref = staintools.LuminosityStandardizer.standardize(ref)

        self.normalizer = staintools.StainNormalizer(method='macenko')

        try:
            self.normalizer.fit(ref)
            self.ready = True
        except Exception:
            print("WARNING: Stain normalizer failed to fit. Using fallback (no normalization).")
            self.ready = False

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = staintools.LuminosityStandardizer.standardize(img)

        if not self.ready:
            return img
        
        
        return self.normalizer.transform(img)
        
