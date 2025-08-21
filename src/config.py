"""
src/config.py
Central config for reproducibility and hyperparameters
"""

# Global seed for reproducibility
SEED = 42

# Data
MAX_TEXT_SAMPLES = 5000  # number of text samples used after sampling
IMG_SIZE = 224  # target size for image transforms
BATCH_SIZE = 8  # default batch size for loaders
