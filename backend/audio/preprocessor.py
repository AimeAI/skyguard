"""
Audio preprocessing utilities
"""
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SAMPLE_RATE, CHUNK_SAMPLES


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

    def chunk_audio(self, audio: np.ndarray, chunk_size: int = CHUNK_SAMPLES) -> list:
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
