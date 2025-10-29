#!/usr/bin/env python3
"""
Train SkyGuard model on synthetic Class 1 UAS audio data
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time

sys.path.append(str(Path(__file__).parent.parent))
from models.classifier import DroneClassifier
from audio.feature_extractor import FeatureExtractor
from config import *

class SyntheticDroneDataset(Dataset):
    """Dataset for synthetic drone audio"""

    def __init__(self, data_dir, train=True, train_split=0.8):
        self.data_dir = Path(data_dir)
        self.feature_extractor = FeatureExtractor()

        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.classes = metadata['classes']
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}

        # Load all file paths
        all_samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name.replace(" ", "_")
            files = list(class_dir.glob("*.npy"))
            for f in files:
                all_samples.append({'path': f, 'label': class_name})

        # Split train/val
        np.random.seed(42)
        np.random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_split)

        if train:
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]

        print(f"{'Train' if train else 'Val'} dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        audio = np.load(sample['path'])

        # Extract mel spectrogram
        mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)  # Add channel dimension

        # Get label
        label = self.label_to_idx[sample['label']]

        return mel_spec, label


def train_model(model, train_loader, val_loader, num_epochs=20, device='cpu'):
    """Train the model"""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * 2)  # Higher LR for synthetic data

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
    print("SkyGuard Model Training - Class 1 UAS Acoustic Signatures")
    print("=" * 60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic_drone_audio'

    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        print("Please run generate_synthetic_drone_audio.py first")
        return

    try:
        # Load datasets
        train_dataset = SyntheticDroneDataset(data_dir, train=True)
        val_dataset = SyntheticDroneDataset(data_dir, train=False)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )

        # Create model
        num_classes = len(train_dataset.classes)
        print(f"\nInitializing model with {num_classes} classes...")
        print(f"Classes: {train_dataset.classes}")
        model = DroneClassifier(num_classes=num_classes).to(device)

        # Train
        print("\nStarting training...")
        start_time = time.time()

        history, best_val_acc = train_model(
            model, train_loader, val_loader,
            num_epochs=20,
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
            'labels': train_dataset.classes,
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
