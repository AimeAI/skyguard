"""
Stage 2: Transfer Learning Classifier
Fine-tuned model for drone classification
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import NUM_CLASSES, EMBEDDING_DIM


class DroneClassifier(nn.Module):
    """
    Transfer learning classifier using pretrained audio model
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = 0.3,
        use_simple_feature_extractor: bool = True
    ):
        super().__init__()

        self.use_simple_feature_extractor = use_simple_feature_extractor
        self.embedding_dim = embedding_dim

        if use_simple_feature_extractor:
            # Simple CNN feature extractor for testing
            # Input: (batch, 1, 128, time_steps)
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),  # Fixed size output
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, embedding_dim),
                nn.ReLU()
            )
        else:
            # Placeholder for YAMNet or other pretrained model
            # Will be loaded from pretrained weights
            self.feature_extractor = None

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
            x: Input tensor (batch, channels, height, width) or embeddings

        Returns:
            Tuple of (logits, embeddings)
        """
        # Extract embeddings
        if self.use_simple_feature_extractor:
            embeddings = self.feature_extractor(x)
        else:
            if self.feature_extractor is not None:
                embeddings = self.feature_extractor(x)
            else:
                # Assume input is already embeddings
                embeddings = x.view(x.size(0), -1)

        # Classify
        logits = self.classifier(embeddings)

        return logits, embeddings

    def load_pretrained(self, model_path: Path):
        """Load pretrained weights"""
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)

    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict()
        }, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
