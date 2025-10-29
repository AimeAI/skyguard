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
WINDOW_SIZE = 7  # Number of predictions to average (increased for stability)
HYSTERESIS_THRESHOLD = 0.7  # Confidence threshold for state change

# Server parameters
HOST = "0.0.0.0"
PORT = 8000
WEBSOCKET_HEARTBEAT = 1.0  # seconds

# Inference parameters
CONFIDENCE_THRESHOLD = 0.15  # Minimum confidence for positive detection (adjusted for demo)
MAX_LATENCY_MS = 200  # Target maximum latency

# Class names - Real Class 1 UAS (NATO/CAF definition)
# Based on commercial quadcopters with documented acoustic signatures
CLASS_NAMES = [
    "Non-Drone",
    "DJI Mavic 3",        # High-pitched 3kHz harmonic
    "DJI Air 3",          # Mid-frequency whine, modulated RPM
    "DJI Mini 4 Pro",     # Light 2.5kHz prop whine
    "DJI Phantom 4 Pro",  # Classic buzz, 1.8-2.2kHz
    "Autel EVO II Pro",   # Similar to Phantom, deeper tone
    "Skydio 2+",          # Variable RPM, AI navigation pauses
    "DJI Air 2S",         # Mid-frequency modulated
    "DJI Mini 3 Pro",     # Rapid harmonics, micro quad
    "Parrot Anafi",       # Quieter, higher pitch 2.8-3.2kHz
    "DJI Inspire 2",      # Professional, deep rumble 1.2-1.6kHz
]

# Drone specifications database - tactical intel
DRONE_SPECS = {
    "Non-Drone": {
        "weight_kg": 0,
        "weight_class": "N/A",
        "max_range_km": 0,
        "max_flight_time_min": 0,
        "max_speed_kph": 0,
        "threat_level": "None",
        "description": "Background noise (birds, vehicles, wind)"
    },
    "DJI Mavic 3": {
        "weight_kg": 0.895,
        "weight_class": "< 1kg (Micro UAS)",
        "max_range_km": 30,
        "max_flight_time_min": 46,
        "max_speed_kph": 75,
        "threat_level": "Low",
        "camera": "Hasselblad 20MP + 12MP Telephoto (28x hybrid zoom)",
        "acoustic_signature": "High-pitched 3kHz harmonic tone",
        "description": "Consumer/prosumer foldable quadcopter, extended flight time"
    },
    "DJI Air 3": {
        "weight_kg": 0.720,
        "weight_class": "< 1kg (Micro UAS)",
        "max_range_km": 32,
        "max_flight_time_min": 46,
        "max_speed_kph": 70,
        "threat_level": "Low",
        "camera": "Dual 48MP wide + medium telephoto",
        "acoustic_signature": "Mid-frequency whine, modulated RPM",
        "description": "Compact dual-camera drone, obstacle sensing"
    },
    "DJI Mini 4 Pro": {
        "weight_kg": 0.249,
        "weight_class": "< 0.25kg (Sub-250g)",
        "max_range_km": 25,
        "max_flight_time_min": 34,
        "max_speed_kph": 58,
        "threat_level": "Very Low",
        "camera": "48MP, 4K/60fps HDR video",
        "acoustic_signature": "Light 2.5kHz prop whine, rapid harmonics",
        "description": "Sub-250g micro drone, omnidirectional obstacle sensing"
    },
    "DJI Phantom 4 Pro": {
        "weight_kg": 1.388,
        "weight_class": "1-4kg (Small UAS)",
        "max_range_km": 7,
        "max_flight_time_min": 30,
        "max_speed_kph": 72,
        "threat_level": "Low-Medium",
        "camera": "20MP 1-inch CMOS sensor",
        "acoustic_signature": "Classic drone buzz, 1.8-2.2kHz band",
        "description": "Professional photography/surveying platform"
    },
    "Autel EVO II Pro": {
        "weight_kg": 1.127,
        "weight_class": "1-4kg (Small UAS)",
        "max_range_km": 9,
        "max_flight_time_min": 40,
        "max_speed_kph": 72,
        "threat_level": "Low-Medium",
        "camera": "6K video, 20MP stills",
        "acoustic_signature": "Similar to Phantom, deeper tone (1.5-1.9kHz)",
        "description": "DJI competitor, extended range, modular payloads"
    },
    "Skydio 2+": {
        "weight_kg": 0.775,
        "weight_class": "< 1kg (Micro UAS)",
        "max_range_km": 6,
        "max_flight_time_min": 27,
        "max_speed_kph": 58,
        "threat_level": "Low",
        "camera": "12MP, 4K/60fps HDR",
        "acoustic_signature": "Variable RPM modulation, AI navigation pauses",
        "description": "AI-powered autonomous tracking, obstacle avoidance (6x 4K nav cameras)",
        "special_note": "Advanced autonomy - can operate without GPS"
    },
    "DJI Air 2S": {
        "weight_kg": 0.595,
        "weight_class": "< 1kg (Micro UAS)",
        "max_range_km": 18.5,
        "max_flight_time_min": 31,
        "max_speed_kph": 68,
        "threat_level": "Low",
        "camera": "20MP 1-inch CMOS, 5.4K video",
        "acoustic_signature": "Mid-frequency modulated whine",
        "description": "Compact high-end camera drone"
    },
    "DJI Mini 3 Pro": {
        "weight_kg": 0.249,
        "weight_class": "< 0.25kg (Sub-250g)",
        "max_range_km": 25,
        "max_flight_time_min": 34,
        "max_speed_kph": 58,
        "threat_level": "Very Low",
        "camera": "48MP, 4K HDR video, true vertical shooting",
        "acoustic_signature": "Rapid harmonics, micro quadcopter whine",
        "description": "Sub-250g, tri-directional obstacle sensing"
    },
    "Parrot Anafi": {
        "weight_kg": 0.320,
        "weight_class": "< 0.5kg (Micro UAS)",
        "max_range_km": 4,
        "max_flight_time_min": 25,
        "max_speed_kph": 55,
        "threat_level": "Very Low",
        "camera": "21MP, 4K HDR video, 180° tilt gimbal",
        "acoustic_signature": "Quieter, higher pitch 2.8-3.2kHz",
        "description": "Ultra-portable foldable, quietest in class, unique 180° gimbal"
    },
    "DJI Inspire 2": {
        "weight_kg": 3.290,
        "weight_class": "1-4kg (Small UAS)",
        "max_range_km": 7,
        "max_flight_time_min": 27,
        "max_speed_kph": 94,
        "threat_level": "Medium",
        "camera": "Zenmuse X5S/X7 (professional cinema cameras)",
        "acoustic_signature": "Deep rumble 1.2-1.6kHz, mechanical vibration at 80Hz",
        "description": "Professional cinematography platform, dual operator support, obstacle avoidance",
        "special_note": "Heavier professional drone - louder, faster, cinema-grade payload"
    }
}
