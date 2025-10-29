"""
Stage 3: Out-of-Distribution Detection
Detects unknown drone models using Mahalanobis distance
"""
import numpy as np
from typing import List
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import OOD_THRESHOLD


class OODDetector:
    """
    Detects out-of-distribution samples using Mahalanobis distance
    """

    def __init__(self, threshold: float = OOD_THRESHOLD):
        self.threshold = threshold
        self.class_means = []
        self.class_covs = []
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

            if len(class_embeddings) == 0:
                # Handle empty class
                self.class_means.append(np.zeros(embeddings.shape[1]))
                self.class_covs.append(np.eye(embeddings.shape[1]))
                continue

            # Compute mean and covariance
            mean = np.mean(class_embeddings, axis=0)

            # For covariance, handle case where we have few samples
            if len(class_embeddings) > 1:
                cov = np.cov(class_embeddings.T)
            else:
                cov = np.eye(embeddings.shape[1])

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

        try:
            inv_cov = np.linalg.inv(cov)
            distance = np.sqrt(diff @ inv_cov @ diff.T)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            inv_cov = np.linalg.pinv(cov)
            distance = np.sqrt(np.abs(diff @ inv_cov @ diff.T))

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
        if not self.is_fitted:
            return False  # Can't determine without fitting

        distance = self.mahalanobis_distance(embedding, predicted_class)
        return distance > self.threshold

    def save(self, path: Path):
        """Save OOD detector statistics"""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            class_means=np.array(self.class_means, dtype=object),
            class_covs=np.array(self.class_covs, dtype=object),
            threshold=self.threshold,
            is_fitted=self.is_fitted
        )

    def load(self, path: Path):
        """Load OOD detector statistics"""
        data = np.load(path, allow_pickle=True)
        self.class_means = list(data['class_means'])
        self.class_covs = list(data['class_covs'])
        self.threshold = float(data['threshold'])
        self.is_fitted = bool(data['is_fitted'])
