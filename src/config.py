import glob as _glob
import os as _os
import torch

# Data
_found = _glob.glob('/kaggle/input/**/patches/train/img', recursive=True)
DATA_ROOT = _os.path.dirname(_os.path.dirname(_found[0])) if _found else './patches'

CLASS_NAMES = {
    0: 'Background',
    1: 'Neutrophil',
    2: 'Epithelial',
    3: 'Lymphocyte',
    4: 'Plasma',
    5: 'Eosinophil',
    6: 'Connective Tissue'
}

CLASS_WEIGHTS = torch.tensor([
    1.00,   # 0 Background
    7.00,   # 1 Neutrophil
    1.43,   # 2 Epithelial
    2.16,   # 3 Lymphocyte
    4.16,   # 4 Plasma
    7.00,   # 5 Eosinophil
    2.11,   # 6 Connective tissue
], dtype=torch.float32)

RARE_CLASSES = {1, 5}       # Neutrophil, Eosinophil
OVERSAMPLE_FACTOR = 4 

# Model architecture
NUM_CLASSES = 7
IN_CHANNELS = 3

# Patch extraction 
PATCH_SIZE = 256
STRIDE = 256           

# Training parameters
BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 8

# Validation
VAL_BATCH_SIZE = 16

# Checkpointing
CHECKPOINT_DIR = './checkpoints'
LOG_DIR = './logs'
EXPERIMENT_NAME = "unet_attention_f_eval"

# Weights & Biases
USE_WANDB = True
WANDB_PROJECT = 'zneus2'
WANDB_ENTITY = 'eakosh-' 
WANDB_LOG_MODEL = True 
WANDB_WATCH_MODEL = True  

# Visualization
VISUALIZE_NUM_SAMPLES = 3
VISUALIZE_EVERY_N_EPOCHS = 10
VAL_IMG_DIR = DATA_ROOT + '/val/img'
VAL_MASK_DIR = DATA_ROOT + '/val/mask'