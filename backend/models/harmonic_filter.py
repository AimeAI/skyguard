"""
Stage 1: Harmonic Pre-Filter
Fast rejection of non-drone sounds based on harmonic analysis
"""
import numpy as np
from scipy import signal
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import HARMONIC_FREQ_MIN, HARMONIC_FREQ_MAX, HARMONIC_THRESHOLD, SAMPLE_RATE


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

        # Check minimum energy (reject very quiet signals)
        if mag_range.max() < 100:  # Minimum magnitude threshold
            return False

        # Find peaks with stricter requirements
        peaks, properties = signal.find_peaks(
            mag_range,
            prominence=self.threshold * mag_range.max(),
            height=mag_range.max() * 0.1  # Peaks must be at least 10% of max
        )

        # Drone propellers create multiple harmonics
        # Look for at least 3 significant peaks
        if len(peaks) < 3:
            return False

        # Additional check: peaks should have reasonable spacing
        # (harmonics are typically evenly spaced)
        if len(peaks) >= 2:
            peak_freqs = freq_range[peaks]
            freq_diffs = np.diff(peak_freqs)
            # Check if frequency differences are somewhat consistent
            # (coefficient of variation < 0.5 suggests harmonic structure)
            if len(freq_diffs) > 0:
                cv = np.std(freq_diffs) / (np.mean(freq_diffs) + 1e-6)
                if cv > 0.8:  # Too irregular for drone harmonics
                    return False

        return True

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
