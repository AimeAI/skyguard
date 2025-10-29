"""
Complete inference pipeline (Stages 1-4)
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import time
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.harmonic_filter import HarmonicFilter
from models.classifier import DroneClassifier
from models.ood_detector import OODDetector
from models.temporal_smoother import TemporalSmoother
from audio.preprocessor import AudioPreprocessor
from audio.feature_extractor import FeatureExtractor
from config import (
    MODEL_DIR, NUM_CLASSES, CLASS_NAMES,
    CONFIDENCE_THRESHOLD, CHUNK_SAMPLES
)


class InferencePipeline:
    """
    Complete inference pipeline for real-time drone detection
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        ood_path: Optional[Path] = None,
        device: str = None
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device

        print("Loading inference pipeline...")

        # Stage 1: Harmonic filter
        self.harmonic_filter = HarmonicFilter()

        # Stage 2: Classifier
        self.model = DroneClassifier(num_classes=NUM_CLASSES, use_simple_feature_extractor=True)

        # Try to load trained weights if available
        if model_path and model_path.exists():
            try:
                self.model.load_checkpoint(model_path)
                print(f"✓ Loaded model from {model_path}")
            except Exception as e:
                print(f"⚠ Could not load model weights: {e}")
                print("  Using randomly initialized model (for testing only)")
        else:
            print("⚠ No trained model found - using random initialization")
            print("  For production, train a model first!")

        self.model.to(device)
        self.model.eval()

        # Stage 3: OOD detector
        self.ood_detector = OODDetector()
        if ood_path and ood_path.exists():
            try:
                self.ood_detector.load(ood_path)
                print(f"✓ Loaded OOD detector from {ood_path}")
            except Exception as e:
                print(f"⚠ Could not load OOD detector: {e}")
        else:
            print("⚠ No OOD detector found - OOD detection disabled")

        # Stage 4: Temporal smoother
        self.temporal_smoother = TemporalSmoother()

        # Audio processing
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = FeatureExtractor()

        print(f"✓ Pipeline loaded on {device}")

    def process_audio(self, audio: np.ndarray) -> Dict:
        """
        Process audio through full pipeline

        Args:
            audio: Raw audio signal (must be CHUNK_SAMPLES long)

        Returns:
            Dictionary with detection results
        """
        start_time = time.time()

        # Ensure correct length
        audio = self.preprocessor.pad_audio(audio, CHUNK_SAMPLES)
        audio = self.preprocessor.normalize_audio(audio)

        # Stage 1: Harmonic pre-filter
        has_harmonics = self.harmonic_filter.has_drone_harmonics(audio)

        if not has_harmonics:
            # Fast rejection
            return {
                'detected': False,
                'class_name': 'Non-Drone',
                'class_id': 0,
                'confidence': 1.0,
                'is_ood': False,
                'latency_ms': (time.time() - start_time) * 1000,
                'stage': 'harmonic_filter'
            }

        # Stage 2: Classification
        mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
        mel_spec_tensor = mel_spec_tensor.to(self.device)

        with torch.no_grad():
            logits, embeddings = self.model(mel_spec_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, 1)

            confidence = confidence.item()
            predicted = predicted.item()
            embeddings = embeddings.cpu().numpy()[0]

        # Stage 3: OOD detection
        is_ood = False
        if self.ood_detector.is_fitted:
            is_ood = self.ood_detector.is_ood(embeddings, predicted)

        # Stage 4: Temporal smoothing
        smoothed_pred, smoothed_conf = self.temporal_smoother.update(
            predicted, confidence
        )

        # Determine detection
        detected = smoothed_conf >= CONFIDENCE_THRESHOLD and smoothed_pred > 0

        result = {
            'detected': detected,
            'class_name': CLASS_NAMES[smoothed_pred] if not is_ood else 'Unknown Drone',
            'class_id': int(smoothed_pred),
            'confidence': float(smoothed_conf),
            'is_ood': is_ood,
            'latency_ms': (time.time() - start_time) * 1000,
            'stage': 'full_pipeline',
            'raw_prediction': int(predicted),
            'raw_confidence': float(confidence),
            'dominant_frequency': self.harmonic_filter.get_dominant_frequency(audio)
        }

        return result

    def reset(self):
        """Reset temporal smoother state"""
        self.temporal_smoother.reset()
