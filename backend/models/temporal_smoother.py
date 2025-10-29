"""
Stage 4: Temporal Smoothing
Stabilizes classifications over time
"""
from collections import deque
import numpy as np
from typing import Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import WINDOW_SIZE, HYSTERESIS_THRESHOLD


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
        majority_prediction = int(unique[np.argmax(counts)])

        # Average confidence for majority class
        majority_confidences = [
            conf for pred, conf in zip(self.prediction_history, self.confidence_history)
            if pred == majority_prediction
        ]
        avg_confidence = float(np.mean(majority_confidences))

        # Hysteresis: only change state if confidence is high enough
        if self.current_state is None:
            self.current_state = majority_prediction
        elif self.current_state != majority_prediction:
            if avg_confidence >= self.hysteresis_threshold:
                self.current_state = majority_prediction

        return self.current_state, avg_confidence

    def reset(self):
        """Reset smoother state"""
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.current_state = None
