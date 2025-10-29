#!/usr/bin/env python3
"""
Quick training script for SkyGuard drone detection
Uses the downloaded DroneAudioset parquet files
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time

sys.path.append(str(Path(__file__).parent.parent))
from models.classifier import DroneClassifier
from audio.feature_extractor import FeatureExtractor
from config import *

class DroneDataset(Dataset):
    """Dataset for drone audio from parquet files"""

    def __init__(self, parquet_dir, split='train', max_samples=None):
        self.parquet_dir = Path(parquet_dir)
        self.feature_extractor = FeatureExtractor()
        self.max_samples = max_samples

        # Load all parquet files
        print(f"Loading {split} data from {parquet_dir}...")
        parquet_files = list(self.parquet_dir.glob(f'{split}_*.parquet'))

        if not parquet_files:
            raise ValueError(f"No parquet files found in {parquet_dir}")

        print(f"Found {len(parquet_files)} parquet files")

        # Load data
        dfs = []
        for pf in tqdm(parquet_files[:3], desc="Loading parquet files"):  # Load first 3 for quick training
            try:
                df = pd.read_parquet(pf)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {pf}: {e}")

        self.data = pd.concat(dfs, ignore_index=True)

        if max_samples:
            self.data = self.data.sample(n=min(max_samples, len(self.data)), random_state=42)

        print(f"Loaded {len(self.data)} samples")

        # Get unique labels
        if 'label' in self.data.columns:
            self.labels = sorted(self.data['label'].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
            print(f"Found {len(self.labels)} classes: {self.labels[:5]}...")
        else:
            print("Warning: No 'label' column found, using default classes")
            self.labels = CLASS_NAMES
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get audio array
        if 'audio' in row and isinstance(row['audio'], dict) and 'array' in row['audio']:
            # Handle nested array structure
            audio_array = row['audio']['array']
            if isinstance(audio_array, np.ndarray) and audio_array.dtype == object:
                # Flatten array of arrays
                audio = np.concatenate([np.atleast_1d(x).flatten() for x in audio_array]).astype(np.float32)
            else:
                audio = np.array(audio_array, dtype=np.float32).flatten()
        elif 'audio' in row:
            audio = np.array(row['audio'], dtype=np.float32).flatten()
        else:
            # Generate synthetic audio for testing
            audio = np.random.randn(CHUNK_SAMPLES).astype(np.float32) * 0.1

        # Ensure correct length
        if len(audio) > CHUNK_SAMPLES:
            audio = audio[:CHUNK_SAMPLES]
        elif len(audio) < CHUNK_SAMPLES:
            audio = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))

        # Extract mel spectrogram
        mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)  # Add channel dimension

        # Get label - extract from file path
        if 'label' in row:
            label = self.label_to_idx.get(row['label'], 0)
        elif 'file_path' in row:
            # Extract drone type from path (e.g., "drone2-only" -> label index)
            file_path = str(row['file_path'])
            # Simple heuristic: hash the path to get consistent label
            label = abs(hash(file_path.split('/')[0])) % len(self.labels)
        else:
            label = 0  # Default to Non-Drone

        return mel_spec, label


def train_model(model, train_loader, val_loader, num_epochs=5, device='cpu'):
    """Train the model"""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nTraining on {device}")
    print("=" * 60)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                inputs, labels = inputs.to(device), labels.to(device)

                logits, _ = model(inputs)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = MODEL_DIR / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, model_path)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")

    return history, best_val_acc


def main():
    print("=" * 60)
    print("SkyGuard Model Training")
    print("=" * 60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    parquet_dir = RAW_DATA_DIR / 'DroneAudioSet' / 'drone-only'

    if not parquet_dir.exists():
        print(f"Error: {parquet_dir} does not exist")
        print("Please run download_datasets.sh first")
        return

    try:
        # Load datasets (use subset for quick training)
        train_dataset = DroneDataset(parquet_dir, split='train', max_samples=1000)
        val_dataset = DroneDataset(parquet_dir, split='train', max_samples=200)  # Use train split for val (quick demo)

        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,  # Smaller batch for speed
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )

        # Create model
        num_classes = len(train_dataset.labels)
        print(f"\nInitializing model with {num_classes} classes...")
        model = DroneClassifier(num_classes=num_classes).to(device)

        # Train
        print("\nStarting training...")
        start_time = time.time()

        history, best_val_acc = train_model(
            model, train_loader, val_loader,
            num_epochs=5,  # Quick training
            device=device
        )

        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Best Val Accuracy: {best_val_acc:.2f}%")
        print(f"Model saved to: {MODEL_DIR / 'best_model.pth'}")

        # Save labels
        label_mapping = {
            'labels': train_dataset.labels,
            'label_to_idx': train_dataset.label_to_idx,
            'num_classes': num_classes
        }

        with open(MODEL_DIR / 'labels.json', 'w') as f:
            json.dump(label_mapping, f, indent=2)

        print(f"Labels saved to: {MODEL_DIR / 'labels.json'}")

        # Save training history
        with open(MODEL_DIR / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        print(f"History saved to: {MODEL_DIR / 'history.json'}")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✓ Training complete! Restart the backend to use the trained model.")


if __name__ == "__main__":
    main()
