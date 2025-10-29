# PROJECT: SkyGuard Tactical - Drone Audio Detection System
# HACKATHON: "Shazam for Drones" - November 15, 2025
# GOAL: Win 1st place ($10,000 CAD)

## MISSION BRIEF
Build a real-time acoustic drone detection system that:
- Detects and classifies 10 drone models + non-drone sounds
- Works in noisy environments (wind, traffic, birds)
- Provides tactical interface for military operators
- Demonstrates multi-sensor localization capability
- Achieves >95% accuracy with <200ms latency

## TECHNICAL ARCHITECTURE

### System Pipeline (4 Stages):

1. **Stage 1: Harmonic Pre-Filter**
   - Input: Raw audio (16kHz mono)
   - Process: FFT analysis for 500-5000Hz harmonic peaks
   - Output: Pass/reject decision
   - Purpose: Fast rejection of obvious non-drones
   - Target: <50ms processing time

2. **Stage 2: Transfer Learning Classifier**
   - Input: 3-second audio chunks
   - Process: Mel-spectrogram → Pretrained YAMNet/PANN → Fine-tuned classifier
   - Output: 11-class probabilities (10 drones + non-drone) + embeddings
   - Purpose: Accurate classification
   - Target: >95% accuracy, ~150ms inference

3. **Stage 3: OOD Detection**
   - Input: Embedding vectors from Stage 2
   - Process: Mahalanobis distance from class centroids
   - Output: In-distribution / Out-of-distribution flag
   - Purpose: Detect unknown drone models
   - Target: Flag unknowns with >80% recall

4. **Stage 4: Temporal Smoothing**
   - Input: Stream of classifications
   - Process: 3-second sliding window with majority voting + hysteresis
   - Output: Stable, non-flickering classification
   - Purpose: Production-grade reliability
   - Target: No flicker on continuous audio

---

## PROJECT STRUCTURE
```
skyguard-tactical/
├── backend/
│   ├── main.py                 # FastAPI server with WebSocket
│   ├── models/
│   │   ├── __init__.py
│   │   ├── harmonic_filter.py  # Stage 1: FFT-based pre-filter
│   │   ├── classifier.py       # Stage 2: Transfer learning model
│   │   ├── ood_detector.py     # Stage 3: Out-of-distribution detection
│   │   └── temporal_smoother.py # Stage 4: Sliding window smoothing
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── preprocessor.py     # Audio loading, resampling, normalization
│   │   └── feature_extractor.py # Mel-spectrogram generation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py            # Model training script
│   │   ├── dataset.py          # PyTorch dataset loader
│   │   └── augmentation.py     # Audio augmentation (noise, time-shift)
│   ├── inference/
│   │   ├── __init__.py
│   │   └── pipeline.py         # Full inference pipeline (Stages 1-4)
│   ├── config.py               # Configuration constants
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/
│   │   ├── page.tsx            # Main dashboard page
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── ThreatStatus.tsx    # Red/Yellow/Green threat indicator
│   │   ├── Spectrogram.tsx     # Real-time spectrogram visualization
│   │   ├── ClassificationDisplay.tsx # Drone model + confidence
│   │   ├── EventLog.tsx        # Detection history with timestamps
│   │   ├── MetricsDashboard.tsx # Accuracy, latency, confusion matrix
│   │   └── TacticalMap.tsx     # Geospatial view with bearing lines
│   ├── lib/
│   │   └── websocket.ts        # WebSocket client for real-time updates
│   ├── public/
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   └── next.config.js
├── data/
│   ├── raw/                    # Original hackathon dataset (user provides)
│   ├── processed/              # Preprocessed spectrograms
│   ├── pretrained/             # Downloaded YAMNet/PANN weights
│   └── models/                 # Trained model checkpoints
├── demo/
│   ├── audio_clips/            # 7 demo clips for presentation
│   ├── demo_script.md          # Step-by-step demo narration
│   └── presentation.pdf        # 5-slide deck
├── tests/
│   ├── test_harmonic_filter.py
│   ├── test_classifier.py
│   ├── test_ood_detector.py
│   └── test_pipeline.py
├── docs/
│   ├── ARCHITECTURE.md         # System design documentation
│   ├── TRAINING.md             # Model training guide
│   └── DEPLOYMENT.md           # Production deployment guide
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## DETAILED IMPLEMENTATION SPECS

### BACKEND: backend/config.py
```python
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
```

### BACKEND: backend/audio/preprocessor.py
```python
"""
Audio preprocessing utilities
"""
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
from backend.config import SAMPLE_RATE, CHUNK_SAMPLES

class AudioPreprocessor:
    """Handles audio loading, resampling, and normalization"""
    
    def __init__(self, target_sr: int = SAMPLE_RATE):
        self.target_sr = target_sr
    
    def load_audio(self, path: Path, duration: Optional[float] = None) -> np.ndarray:
        """
        Load audio file and resample to target sample rate
        
        Args:
            path: Path to audio file
            duration: Optional duration to load (seconds)
        
        Returns:
            Audio signal as numpy array
        """
        audio, sr = librosa.load(path, sr=self.target_sr, duration=duration, mono=True)
        return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio: Input audio signal
        
        Returns:
            Normalized audio
        """
        if audio.max() == 0:
            return audio
        return audio / np.abs(audio).max()
    
    def chunk_audio(self, audio: np.ndarray, chunk_size: int = CHUNK_SAMPLES) -> list[np.ndarray]:
        """
        Split audio into fixed-size chunks with 50% overlap
        
        Args:
            audio: Input audio signal
            chunk_size: Size of each chunk in samples
        
        Returns:
            List of audio chunks
        """
        hop_size = chunk_size // 2
        chunks = []
        
        for start in range(0, len(audio) - chunk_size + 1, hop_size):
            chunk = audio[start:start + chunk_size]
            if len(chunk) == chunk_size:
                chunks.append(chunk)
        
        return chunks
    
    def pad_audio(self, audio: np.ndarray, target_length: int = CHUNK_SAMPLES) -> np.ndarray:
        """
        Pad or truncate audio to target length
        
        Args:
            audio: Input audio signal
            target_length: Desired length in samples
        
        Returns:
            Padded/truncated audio
        """
        if len(audio) > target_length:
            return audio[:target_length]
        elif len(audio) < target_length:
            pad_length = target_length - len(audio)
            return np.pad(audio, (0, pad_length), mode='constant')
        return audio
    
    def add_noise(self, audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """
        Add Gaussian noise for data augmentation
        
        Args:
            audio: Input audio signal
            noise_factor: Standard deviation of noise
        
        Returns:
            Noisy audio
        """
        noise = np.random.normal(0, noise_factor, audio.shape)
        return audio + noise
    
    def time_shift(self, audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
        """
        Randomly shift audio in time for data augmentation
        
        Args:
            audio: Input audio signal
            shift_max: Maximum shift as fraction of audio length
        
        Returns:
            Time-shifted audio
        """
        shift_amount = int(np.random.uniform(-shift_max, shift_max) * len(audio))
        return np.roll(audio, shift_amount)
```

### BACKEND: backend/audio/feature_extractor.py
```python
"""
Feature extraction for audio classification
"""
import librosa
import numpy as np
from backend.config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS

class FeatureExtractor:
    """Extracts mel-spectrograms and other features from audio"""
    
    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log mel-spectrogram
        
        Args:
            audio: Input audio signal
        
        Returns:
            Log mel-spectrogram (n_mels, time_steps)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=8000
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
        """
        Extract MFCC features
        
        Args:
            audio: Input audio signal
            n_mfcc: Number of MFCCs to extract
        
        Returns:
            MFCC features (n_mfcc, time_steps)
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return mfccs
    
    def extract_fft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract FFT for harmonic analysis
        
        Args:
            audio: Input audio signal
        
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        frequency = np.fft.rfftfreq(len(audio), 1.0 / self.sr)
        
        return frequency, magnitude
```

### BACKEND: backend/models/harmonic_filter.py
```python
"""
Stage 1: Harmonic Pre-Filter
Fast rejection of non-drone sounds based on harmonic analysis
"""
import numpy as np
from scipy import signal
from backend.config import HARMONIC_FREQ_MIN, HARMONIC_FREQ_MAX, HARMONIC_THRESHOLD, SAMPLE_RATE

class HarmonicFilter:
    """
    Detects presence of harmonic peaks characteristic of drone propellers
    """
    
    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        freq_min: int = HARMONIC_FREQ_MIN,
        freq_max: int = HARMONIC_FREQ_MAX,
        threshold: float = HARMONIC_THRESHOLD
    ):
        self.sr = sr
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.threshold = threshold
    
    def has_drone_harmonics(self, audio: np.ndarray) -> bool:
        """
        Check if audio contains drone-like harmonic patterns
        
        Args:
            audio: Input audio signal
        
        Returns:
            True if harmonics detected, False otherwise
        """
        # Compute FFT
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        frequency = np.fft.rfftfreq(len(audio), 1.0 / self.sr)
        
        # Focus on drone frequency range
        freq_mask = (frequency >= self.freq_min) & (frequency <= self.freq_max)
        freq_range = frequency[freq_mask]
        mag_range = magnitude[freq_mask]
        
        if len(mag_range) == 0:
            return False
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            mag_range,
            prominence=self.threshold * mag_range.max()
        )
        
        # Drone propellers create multiple harmonics
        # Look for at least 3 significant peaks
        return len(peaks) >= 3
    
    def get_dominant_frequency(self, audio: np.ndarray) -> float:
        """
        Get dominant frequency in drone range
        
        Args:
            audio: Input audio signal
        
        Returns:
            Dominant frequency in Hz
        """
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        frequency = np.fft.rfftfreq(len(audio), 1.0 / self.sr)
        
        freq_mask = (frequency >= self.freq_min) & (frequency <= self.freq_max)
        freq_range = frequency[freq_mask]
        mag_range = magnitude[freq_mask]
        
        if len(mag_range) == 0:
            return 0.0
        
        dominant_idx = np.argmax(mag_range)
        return float(freq_range[dominant_idx])
```

### BACKEND: backend/models/classifier.py
```python
"""
Stage 2: Transfer Learning Classifier
Fine-tuned model for drone classification
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from pathlib import Path
from typing import Tuple
from backend.config import NUM_CLASSES, EMBEDDING_DIM

class DroneClassifier(nn.Module):
    """
    Transfer learning classifier using pretrained audio model
    """
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Use pretrained YAMNet or similar
        # For now, placeholder for feature extractor
        self.feature_extractor = None  # Will be loaded from pretrained
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, channels, height, width)
        
        Returns:
            Tuple of (logits, embeddings)
        """
        # Extract embeddings from pretrained model
        if self.feature_extractor is not None:
            embeddings = self.feature_extractor(x)
        else:
            # Placeholder: assume input is already embeddings
            embeddings = x.view(x.size(0), -1)
        
        # Classify
        logits = self.classifier(embeddings)
        
        return logits, embeddings
    
    def load_pretrained(self, model_path: Path):
        """Load pretrained weights"""
        # Implementation will depend on chosen pretrained model
        pass
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        torch.save(self.state_dict(), path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        self.load_state_dict(torch.load(path))
```

### BACKEND: backend/models/ood_detector.py
```python
"""
Stage 3: Out-of-Distribution Detection
Detects unknown drone models using Mahalanobis distance
"""
import numpy as np
import torch
from typing import List, Tuple
from backend.config import OOD_THRESHOLD, NUM_CLASSES

class OODDetector:
    """
    Detects out-of-distribution samples using Mahalanobis distance
    """
    
    def __init__(self, threshold: float = OOD_THRESHOLD):
        self.threshold = threshold
        self.class_means: List[np.ndarray] = []
        self.class_covs: List[np.ndarray] = []
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Compute class statistics from training data
        
        Args:
            embeddings: Feature embeddings (n_samples, embedding_dim)
            labels: Class labels (n_samples,)
        """
        num_classes = len(np.unique(labels))
        self.class_means = []
        self.class_covs = []
        
        for class_idx in range(num_classes):
            class_embeddings = embeddings[labels == class_idx]
            
            # Compute mean and covariance
            mean = np.mean(class_embeddings, axis=0)
            cov = np.cov(class_embeddings.T)
            
            # Add regularization to avoid singular matrix
            cov += np.eye(cov.shape[0]) * 1e-6
            
            self.class_means.append(mean)
            self.class_covs.append(cov)
        
        self.is_fitted = True
    
    def mahalanobis_distance(
        self,
        embedding: np.ndarray,
        class_idx: int
    ) -> float:
        """
        Compute Mahalanobis distance to a class
        
        Args:
            embedding: Feature embedding
            class_idx: Index of class
        
        Returns:
            Mahalanobis distance
        """
        if not self.is_fitted:
            raise RuntimeError("OODDetector must be fitted before use")
        
        mean = self.class_means[class_idx]
        cov = self.class_covs[class_idx]
        
        diff = embedding - mean
        inv_cov = np.linalg.inv(cov)
        distance = np.sqrt(diff @ inv_cov @ diff.T)
        
        return float(distance)
    
    def is_ood(self, embedding: np.ndarray, predicted_class: int) -> bool:
        """
        Determine if sample is out-of-distribution
        
        Args:
            embedding: Feature embedding
            predicted_class: Predicted class from classifier
        
        Returns:
            True if OOD, False if in-distribution
        """
        distance = self.mahalanobis_distance(embedding, predicted_class)
        return distance > self.threshold
    
    def save(self, path: Path):
        """Save OOD detector statistics"""
        np.savez(
            path,
            class_means=self.class_means,
            class_covs=self.class_covs,
            threshold=self.threshold
        )
    
    def load(self, path: Path):
        """Load OOD detector statistics"""
        data = np.load(path, allow_pickle=True)
        self.class_means = list(data['class_means'])
        self.class_covs = list(data['class_covs'])
        self.threshold = float(data['threshold'])
        self.is_fitted = True
```

### BACKEND: backend/models/temporal_smoother.py
```python
"""
Stage 4: Temporal Smoothing
Stabilizes classifications over time
"""
from collections import deque
import numpy as np
from backend.config import WINDOW_SIZE, HYSTERESIS_THRESHOLD

class TemporalSmoother:
    """
    Smooths classification results using sliding window and hysteresis
    """
    
    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        hysteresis_threshold: float = HYSTERESIS_THRESHOLD
    ):
        self.window_size = window_size
        self.hysteresis_threshold = hysteresis_threshold
        self.prediction_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.current_state = None
    
    def update(self, prediction: int, confidence: float) -> Tuple[int, float]:
        """
        Update with new prediction and return smoothed result
        
        Args:
            prediction: Class prediction
            confidence: Prediction confidence
        
        Returns:
            Tuple of (smoothed_prediction, smoothed_confidence)
        """
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
        if len(self.prediction_history) < self.window_size:
            # Not enough history yet, return current
            self.current_state = prediction
            return prediction, confidence
        
        # Majority voting
        predictions_array = np.array(self.prediction_history)
        unique, counts = np.unique(predictions_array, return_counts=True)
        majority_prediction = unique[np.argmax(counts)]
        
        # Average confidence for majority class
        majority_confidences = [
            conf for pred, conf in zip(self.prediction_history, self.confidence_history)
            if pred == majority_prediction
        ]
        avg_confidence = np.mean(majority_confidences)
        
        # Hysteresis: only change state if confidence is high enough
        if self.current_state != majority_prediction:
            if avg_confidence >= self.hysteresis_threshold:
                self.current_state = majority_prediction
        
        return self.current_state, avg_confidence
    
    def reset(self):
        """Reset smoother state"""
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.current_state = None
```

---

## PART 1 COMPLETE

This is the foundation. Next I'll provide:
- Training pipeline
- Inference pipeline
- FastAPI server
- Frontend components