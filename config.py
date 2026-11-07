"""
Configuration and hyperparameters for AI-Based Eye Disease Classification.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
SAMPLE_IMAGES_DIR = DATASET_DIR / "sample_images"

# File Paths
MODEL_PATH = MODELS_DIR / "eye_disease_model_fixed.keras"
RETINA_NPZ_PATH = DATASET_DIR / "retinamnist.npz"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PATH = OUTPUTS_DIR / "training_history.png"
ARCHITECTURE_PATH = OUTPUTS_DIR / "architecture.png"
SAMPLE_PREDICTIONS_PATH = OUTPUTS_DIR / "sample_predictions.png"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"

# Image and Model Parameters
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)

# Classes
CLASS_NAMES = ["Normal", "Diabetic Retinopathy"]
NUM_CLASSES = 2  # Binary classification

# Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
PATIENCE = 4

# Ensure writable Matplotlib cache directory (important for sandboxed/headless environments)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

