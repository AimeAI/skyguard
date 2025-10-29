"""
Feature extraction for audio classification
"""
import librosa
import numpy as np
from typing import Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS


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
