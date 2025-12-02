import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torchvision.utils import make_grid
import torch
import wandb


COLORS = {
    0: (0, 0, 0),          # background
    1: (255, 0, 0),        # neutrophil
    2: (0, 255, 0),        # epithelial
    3: (0, 0, 255),        # lymphocyte
    4: (255, 255, 0),      # plasma
    5: (255, 0, 255),      # eosinophil
    6: (0, 255, 255),      # connective tissue
}


def colorize_mask(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, rgb in COLORS.items():
        color[mask == cls] = rgb

    return color


class SegmentationVisualizer(pl.Callback):
    def __init__(self, num_samples=3, every_n_epochs=10):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):

        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        if not hasattr(trainer.logger, "experiment"):
            return
        if not isinstance(trainer.logger.experiment, wandb.wandb_sdk.wandb_run.Run):
            return

        val_loader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader))

        images, masks = batch
        images = images.to(pl_module.device)
        masks = masks.to(pl_module.device)

        logits = pl_module(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        cls_ious = pl_module.compute_per_class_iou(logits, masks)
        for cls, iou in cls_ious.items():
            trainer.logger.experiment.log({f"val/iou_class_{cls}": iou})

        img_logs = []

        for i in range(min(self.num_samples, len(images))):
            img = (images[i].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
            true_mask = masks[i].cpu().numpy()
            pred_mask = preds[i]

            true_color = colorize_mask(true_mask)
            pred_color = colorize_mask(pred_mask)

            combined = np.concatenate([img, true_color, pred_color], axis=1)
            img_logs.append(wandb.Image(combined, caption=f"epoch {epoch}, sample {i}"))

        trainer.logger.experiment.log({"val_examples": img_logs})
