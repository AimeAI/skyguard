"""
Complete Inference Pipeline
Integrates all 4 stages: Harmonic Filter → Classifier → OOD → Temporal Smoother
"""
import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.harmonic_filter import HarmonicFilter
from models.classifier import DroneClassifier
from models.ood_detector import OODDetector
from models.temporal_smoother import TemporalSmoother
from audio.feature_extractor import FeatureExtractor
from audio.preprocessor import AudioPreprocessor
from config import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    SAMPLE_RATE,
    CHUNK_SAMPLES
)


class InferencePipeline:
    """
    Complete 4-stage inference pipeline for drone detection

    Stage 1: Harmonic Filter (fast rejection)
    Stage 2: CNN Classifier (drone identification)
    Stage 3: OOD Detector (unknown drone detection)
    Stage 4: Temporal Smoother (stable predictions)
    """

    def __init__(
        self,
        model_path: Path = None,
        ood_path: Path = None,
        device: str = None
    ):
        """
        Initialize inference pipeline

        Args:
            model_path: Path to trained model weights (.pth)
            ood_path: Path to OOD detector statistics (.npz)
            device: 'cpu', 'cuda', or None (auto-detect)
        """
        # Device setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Initializing SkyGuard Pipeline on {self.device}")

        # Stage 1: Harmonic Filter
        self.harmonic_filter = HarmonicFilter()
        print("✓ Stage 1: Harmonic Filter loaded")

        # Stage 2: Classifier
        self.classifier = DroneClassifier(use_simple_feature_extractor=True)

        if model_path and model_path.exists():
            try:
                self.classifier.load_checkpoint(model_path)
                print(f"✓ Stage 2: Classifier loaded from {model_path}")
            except Exception as e:
                print(f"⚠ Warning: Could not load model weights: {e}")
                print("  Using random initialization (for demo/testing)")
        else:
            print("⚠ Warning: No model weights provided")
            print("  Using random initialization (for demo/testing)")

        self.classifier.to(self.device)
        self.classifier.eval()

        # Stage 3: OOD Detector
        self.ood_detector = OODDetector()

        if ood_path and ood_path.exists():
            try:
                self.ood_detector.load(ood_path)
                print(f"✓ Stage 3: OOD Detector loaded from {ood_path}")
            except Exception as e:
                print(f"⚠ Warning: Could not load OOD detector: {e}")
        else:
            print("⚠ Warning: OOD Detector not fitted (will skip OOD checks)")

        # Stage 4: Temporal Smoother
        self.temporal_smoother = TemporalSmoother()
        print("✓ Stage 4: Temporal Smoother initialized")

        # Audio processing utilities
        self.feature_extractor = FeatureExtractor()
        self.preprocessor = AudioPreprocessor()
        print("✓ Audio processors ready")

        print("=" * 60)
        print("Pipeline initialization complete!")
        print("=" * 60)

    def process_audio(self, audio: np.ndarray) -> Dict:
        """
        Process audio through complete pipeline

        Args:
            audio: Raw audio signal (mono, 16kHz)

        Returns:
            Dictionary with detection results:
            {
                'detected': bool,          # Final detection decision
                'class_id': int,          # Predicted class (0 = Non-Drone)
                'class_name': str,        # Human-readable class name
                'confidence': float,      # Prediction confidence [0-1]
                'is_ood': bool,          # Out-of-distribution flag
                'stage': str,            # Which stage made final decision
                'latency_ms': float,     # Processing time in milliseconds
                'dominant_frequency': float,  # Peak frequency (Hz)
                'error': str (optional)  # Error message if processing failed
            }
        """
        start_time = time.time()

        try:
            # ========================================
            # VALIDATION
            # ========================================
            if len(audio) == 0:
                raise ValueError("Empty audio array")

            if not np.isfinite(audio).all():
                raise ValueError("Audio contains NaN or Inf values")

            if audio.dtype not in [np.float32, np.float64]:
                audio = audio.astype(np.float32)

            # Pad or truncate to expected length
            audio = self.preprocessor.pad_audio(audio, CHUNK_SAMPLES)

            # Normalize
            audio = self.preprocessor.normalize_audio(audio)

            # ========================================
            # STAGE 1: HARMONIC FILTER (Fast Rejection)
            # ========================================
            has_harmonics = self.harmonic_filter.has_drone_harmonics(audio)
            dominant_freq = self.harmonic_filter.get_dominant_frequency(audio)

            if not has_harmonics:
                # Fast rejection - not drone-like
                latency = (time.time() - start_time) * 1000
                return {
                    'detected': False,
                    'class_id': 0,
                    'class_name': 'Non-Drone',
                    'confidence': 1.0,
                    'is_ood': False,
                    'stage': 'harmonic_filter',
                    'latency_ms': latency,
                    'dominant_frequency': dominant_freq
                }

            # ========================================
            # STAGE 2: FEATURE EXTRACTION + CLASSIFICATION
            # ========================================
            # Extract mel-spectrogram
            mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)

            # Convert to tensor: (batch=1, channels=1, n_mels, time_steps)
            mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0).float()
            mel_tensor = mel_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                logits, embeddings = self.classifier(mel_tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)

                class_id = int(predicted.item())
                conf_value = float(confidence.item())
                embedding_np = embeddings.cpu().numpy()[0]

            # ========================================
            # STAGE 3: OUT-OF-DISTRIBUTION DETECTION
            # ========================================
            is_ood = False
            if self.ood_detector.is_fitted:
                try:
                    is_ood = self.ood_detector.is_ood(embedding_np, class_id)
                except Exception as e:
                    print(f"OOD detection failed: {e}")
                    is_ood = False

            # ========================================
            # STAGE 4: TEMPORAL SMOOTHING
            # ========================================
            smoothed_class, smoothed_conf = self.temporal_smoother.update(
                class_id, conf_value
            )

            # ========================================
            # FINAL DECISION
            # ========================================
            # Detected if:
            # 1. Confidence above threshold
            # 2. Not classified as "Non-Drone" (class 0)
            # 3. Not flagged as out-of-distribution
            detected = (
                smoothed_conf >= CONFIDENCE_THRESHOLD and
                smoothed_class != 0 and
                not is_ood
            )

            latency = (time.time() - start_time) * 1000

            return {
                'detected': detected,
                'class_id': smoothed_class,
                'class_name': CLASS_NAMES[smoothed_class] if smoothed_class < len(CLASS_NAMES) else f"Unknown_{smoothed_class}",
                'confidence': smoothed_conf,
                'is_ood': is_ood,
                'stage': 'full_pipeline',
                'latency_ms': latency,
                'dominant_frequency': dominant_freq
            }

        except Exception as e:
            # Error handling - return safe default
            latency = (time.time() - start_time) * 1000
            print(f"Error in pipeline: {e}")

            return {
                'detected': False,
                'class_id': 0,
                'class_name': 'Error',
                'confidence': 0.0,
                'is_ood': False,
                'stage': 'error',
                'latency_ms': latency,
                'dominant_frequency': 0.0,
                'error': str(e)
            }

    def reset(self):
        """
        Reset temporal smoother state

        Call this when starting a new audio stream or
        after a long pause in detection
        """
        self.temporal_smoother.reset()
        print("Pipeline state reset")

    def get_statistics(self) -> Dict:
        """
        Get pipeline statistics and configuration

        Returns:
            Dictionary with pipeline stats
        """
        return {
            'device': self.device,
            'sample_rate': SAMPLE_RATE,
            'chunk_samples': CHUNK_SAMPLES,
            'num_classes': len(CLASS_NAMES),
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'ood_fitted': self.ood_detector.is_fitted,
            'smoother_window_size': self.temporal_smoother.window_size,
            'smoother_threshold': self.temporal_smoother.hysteresis_threshold
        }


# Simple test
if __name__ == "__main__":
    print("Testing SkyGuard Pipeline...")
    print("=" * 60)

    # Initialize pipeline (without model weights for testing)
    pipeline = InferencePipeline()

    print("\nTest 1: Silent audio")
    print("-" * 40)
    silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    result1 = pipeline.process_audio(silence)
    print(f"Detected: {result1['detected']}")
    print(f"Stage: {result1['stage']}")
    print(f"Latency: {result1['latency_ms']:.2f}ms")

    print("\nTest 2: Synthetic drone signal")
    print("-" * 40)
    t = np.linspace(0, 3, CHUNK_SAMPLES)
    drone_signal = (
        np.sin(2 * np.pi * 500 * t) +
        0.8 * np.sin(2 * np.pi * 1000 * t) +
        0.6 * np.sin(2 * np.pi * 1500 * t) +
        0.4 * np.sin(2 * np.pi * 2000 * t)
    ).astype(np.float32)

    result2 = pipeline.process_audio(drone_signal)
    print(f"Detected: {result2['detected']}")
    print(f"Class: {result2['class_name']}")
    print(f"Confidence: {result2['confidence']:.2f}")
    print(f"Latency: {result2['latency_ms']:.2f}ms")
    print(f"Dominant Freq: {result2['dominant_frequency']:.1f} Hz")

    print("\nTest 3: Noise")
    print("-" * 40)
    noise = np.random.randn(CHUNK_SAMPLES).astype(np.float32) * 0.1
    result3 = pipeline.process_audio(noise)
    print(f"Detected: {result3['detected']}")
    print(f"Stage: {result3['stage']}")
    print(f"Latency: {result3['latency_ms']:.2f}ms")

    print("\n" + "=" * 60)
    print("Pipeline test complete!")
    print("=" * 60)

    # Print statistics
    print("\nPipeline Configuration:")
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
