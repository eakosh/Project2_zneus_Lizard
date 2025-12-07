# Project1_zneus_Lizard
The second project in a pair from the ZNEUS course at FIIT.

U-Net nuclei segmentation on the Lizard dataset (PNG patches 256×256) with Lightning.

## What’s where
- `src/` — training code
  - `train.py` — launches training
  - `config.py` — data paths (`DATA_ROOT`), hyperparameters, class weights, W&B flags
  - `model.py` — U-Net architecture
  - `patch_datamodule.py` / `patch_dataset.py` — Lightning data module and dataset for pre-cut patches
  - `transforms.py` — augmentations and preprocessing
  - `generate_patches.py` — optional script to create patches from raw slides/masks
  - `visualize.py` — validation visualization callback
- `eda/eda.ipynb` — exploratory notebook

## Quick start
1) Install deps: `pip install -r req.txt`
2) Set `DATA_ROOT` in `src/config.py` to your patches root (expects `train/`, `val/`, `test/` with `img/` and `mask/`)
3) Train: `python src/train.py`
   - Resume: `python src/train.py --resume_from_checkpoint checkpoints/last.ckpt`

## Outputs to submit
- Best checkpoints: `checkpoints/`
- Training logs/metrics: `logs/` (TensorBoard) and linked W&B run if enabled
- Visual samples: saved by `visualize.py` into `logs/` during validation

### TO-DO:
- [X] Data analysis
- [X] Data preprocessing and normalization
- [X] Data split
- [X] Augmentations
- [X] Configuration
- [X] Experiment tracking
- [X] Experiments - meaningful based on the results of previous experiments
- [X] Results and evaluation metrics
- [X] Clear code
- [X] Markdown documentation and comments