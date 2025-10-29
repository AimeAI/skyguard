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

        # Improved Hysteresis Logic:
        # - Easy to enter "drone detected" state (lower threshold)
        # - Hard to leave "drone detected" state (higher threshold)
        # - This prevents flickering when drone is continuously present

        if self.current_state is None:
            # First prediction
            self.current_state = majority_prediction
        elif self.current_state == 0:
            # Currently "Non-Drone" - switch to drone if detected with low threshold
            if majority_prediction != 0 and avg_confidence >= 0.3:  # Easy to detect
                self.current_state = majority_prediction
        else:
            # Currently showing a drone detection
            if majority_prediction == 0:
                # Want to switch back to "Non-Drone" - require high confidence
                if avg_confidence >= 0.85:  # Hard to clear
                    self.current_state = 0
            elif majority_prediction != self.current_state:
                # Switching between different drone types - moderate threshold
                if avg_confidence >= self.hysteresis_threshold:
                    self.current_state = majority_prediction

        # Return confidence for current_state, not majority_prediction
        current_state_confidences = [
            conf for pred, conf in zip(self.prediction_history, self.confidence_history)
            if pred == self.current_state
        ]
        final_confidence = float(np.mean(current_state_confidences)) if current_state_confidences else avg_confidence

        return self.current_state, final_confidence

    def reset(self):
        """Reset smoother state"""
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.current_state = None
