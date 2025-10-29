"""
Configuration constants for SkyGuard Tactical
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
PRETRAINED_DIR = DATA_DIR / "pretrained"

# Audio parameters
SAMPLE_RATE = 16000  # Hz
CHUNK_DURATION = 3.0  # seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# Model parameters
NUM_CLASSES = 11  # 10 drone models + 1 non-drone
EMBEDDING_DIM = 1024  # YAMNet embedding size
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

# Harmonic filter parameters
HARMONIC_FREQ_MIN = 500  # Hz
HARMONIC_FREQ_MAX = 5000  # Hz
HARMONIC_THRESHOLD = 0.3  # Minimum peak prominence

# OOD detection parameters
OOD_THRESHOLD = 3.0  # Mahalanobis distance threshold (tune on validation)

# Temporal smoothing parameters
WINDOW_SIZE = 5  # Number of predictions to average
HYSTERESIS_THRESHOLD = 0.7  # Confidence threshold for state change

# Server parameters
HOST = "0.0.0.0"
PORT = 8000
WEBSOCKET_HEARTBEAT = 1.0  # seconds

# Inference parameters
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for positive detection
MAX_LATENCY_MS = 200  # Target maximum latency

# Class names (to be populated from dataset)
CLASS_NAMES = [
    "Non-Drone",
    "Drone_Model_1",
    "Drone_Model_2",
    "Drone_Model_3",
    "Drone_Model_4",
    "Drone_Model_5",
    "Drone_Model_6",
    "Drone_Model_7",
    "Drone_Model_8",
    "Drone_Model_9",
    "Drone_Model_10",
]
