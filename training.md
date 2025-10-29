📋 ALL MARKDOWN FILES FOR CLAUDE CODE

PART 2: Training, Inference, and Backend Server
markdown### BACKEND: backend/training/dataset.py
"""
PyTorch dataset for drone audio classification
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from backend.audio.preprocessor import AudioPreprocessor
from backend.audio.feature_extractor import FeatureExtractor
from backend.config import CHUNK_SAMPLES

class DroneAudioDataset(Dataset):
    """
    Dataset for loading and preprocessing drone audio samples
    """
    
    def __init__(
        self,
        audio_paths: List[Path],
        labels: List[int],
        augment: bool = False,
        preprocessor: Optional[AudioPreprocessor] = None,
        feature_extractor: Optional[FeatureExtractor] = None
    ):
        """
        Args:
            audio_paths: List of paths to audio files
            labels: List of integer labels
            augment: Whether to apply data augmentation
            preprocessor: AudioPreprocessor instance
            feature_extractor: FeatureExtractor instance
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.augment = augment
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.feature_extractor = feature_extractor or FeatureExtractor()
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess a single sample
        
        Returns:
            Tuple of (mel_spectrogram, label)
        """
        # Load audio
        audio = self.preprocessor.load_audio(self.audio_paths[idx])
        
        # Pad/truncate to fixed length
        audio = self.preprocessor.pad_audio(audio, CHUNK_SAMPLES)
        
        # Normalize
        audio = self.preprocessor.normalize_audio(audio)
        
        # Augmentation (training only)
        if self.augment:
            if np.random.random() > 0.5:
                audio = self.preprocessor.add_noise(audio)
            if np.random.random() > 0.5:
                audio = self.preprocessor.time_shift(audio)
        
        # Extract mel-spectrogram
        mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
        
        # Convert to tensor (add channel dimension)
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
        
        return mel_spec_tensor, self.labels[idx]

def load_dataset(
    data_dir: Path,
    split: str = 'train'
) -> Tuple[List[Path], List[int]]:
    """
    Load dataset file paths and labels
    
    Args:
        data_dir: Root directory containing audio files
        split: 'train', 'val', or 'test'
    
    Returns:
        Tuple of (audio_paths, labels)
    """
    # Assume directory structure: data_dir/split/class_name/*.wav
    audio_paths = []
    labels = []
    
    split_dir = data_dir / split
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    
    for class_idx, class_dir in enumerate(class_dirs):
        audio_files = list(class_dir.glob('*.wav')) + list(class_dir.glob('*.mp3'))
        audio_paths.extend(audio_files)
        labels.extend([class_idx] * len(audio_files))
    
    return audio_paths, labels

def create_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Root directory containing audio files
        batch_size: Batch size for training
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load datasets
    train_paths, train_labels = load_dataset(data_dir, 'train')
    val_paths, val_labels = load_dataset(data_dir, 'val')
    test_paths, test_labels = load_dataset(data_dir, 'test')
    
    # Create datasets
    train_dataset = DroneAudioDataset(train_paths, train_labels, augment=True)
    val_dataset = DroneAudioDataset(val_paths, val_labels, augment=False)
    test_dataset = DroneAudioDataset(test_paths, test_labels, augment=False)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
```

### BACKEND: backend/training/train.py
"""
Model training script
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Tuple
import json

from backend.config import (
    NUM_CLASSES, LEARNING_RATE, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, MODEL_DIR, PROCESSED_DATA_DIR
)
from backend.models.classifier import DroneClassifier
from backend.models.ood_detector import OODDetector
from backend.training.dataset import create_dataloaders

class Trainer:
    """
    Model trainer with early stopping and checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(inputs)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model
        
        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, _ = self.model(inputs)
                loss = self.criterion(logits, labels)
                
                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int = NUM_EPOCHS, patience: int = EARLY_STOPPING_PATIENCE):
        """
        Full training loop with early stopping
        
        Args:
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.save_checkpoint(MODEL_DIR / 'best_model.pth')
                print("✓ Saved best model")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
        
        # Save final model and history
        self.save_checkpoint(MODEL_DIR / 'final_model.pth')
        self.save_history(MODEL_DIR / 'training_history.json')
        
        print("\n✓ Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, path)
    
    def save_history(self, path: Path):
        """Save training history"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)

def extract_embeddings(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings for OOD detector training
    
    Args:
        model: Trained classifier
        dataloader: Data loader
        device: Device to use
    
    Returns:
        Tuple of (embeddings, labels)
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Extracting embeddings'):
            inputs = inputs.to(device)
            _, embeddings = model(inputs)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    return embeddings, labels

def main():
    """Main training function"""
    print("=== SkyGuard Tactical - Model Training ===\n")
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        PROCESSED_DATA_DIR,
        batch_size=32,
        num_workers=4
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = DroneClassifier(num_classes=NUM_CLASSES)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train()
    
    # Extract embeddings for OOD detector
    print("\nExtracting embeddings for OOD detector...")
    model.load_state_dict(
        torch.load(MODEL_DIR / 'best_model.pth')['model_state_dict']
    )
    
    train_embeddings, train_labels = extract_embeddings(model, train_loader)
    
    # Train OOD detector
    print("Training OOD detector...")
    ood_detector = OODDetector()
    ood_detector.fit(train_embeddings, train_labels)
    ood_detector.save(MODEL_DIR / 'ood_detector.npz')
    
    print("\n✓ All training complete!")

if __name__ == '__main__':
    main()
```

### BACKEND: backend/inference/pipeline.py
"""
Complete inference pipeline (Stages 1-4)
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import time

from backend.models.harmonic_filter import HarmonicFilter
from backend.models.classifier import DroneClassifier
from backend.models.ood_detector import OODDetector
from backend.models.temporal_smoother import TemporalSmoother
from backend.audio.preprocessor import AudioPreprocessor
from backend.audio.feature_extractor import FeatureExtractor
from backend.config import (
    MODEL_DIR, NUM_CLASSES, CLASS_NAMES,
    CONFIDENCE_THRESHOLD, CHUNK_SAMPLES
)

class InferencePipeline:
    """
    Complete inference pipeline for real-time drone detection
    """
    
    def __init__(
        self,
        model_path: Path = MODEL_DIR / 'best_model.pth',
        ood_path: Path = MODEL_DIR / 'ood_detector.npz',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        
        # Initialize components
        print("Loading inference pipeline...")
        
        # Stage 1: Harmonic filter
        self.harmonic_filter = HarmonicFilter()
        
        # Stage 2: Classifier
        self.model = DroneClassifier(num_classes=NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Stage 3: OOD detector
        self.ood_detector = OODDetector()
        self.ood_detector.load(ood_path)
        
        # Stage 4: Temporal smoother
        self.temporal_smoother = TemporalSmoother()
        
        # Audio processing
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = FeatureExtractor()
        
        print(f"✓ Pipeline loaded on {device}")
    
    def process_audio(self, audio: np.ndarray) -> Dict:
        """
        Process audio through full pipeline
        
        Args:
            audio: Raw audio signal (must be CHUNK_SAMPLES long)
        
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        # Ensure correct length
        audio = self.preprocessor.pad_audio(audio, CHUNK_SAMPLES)
        audio = self.preprocessor.normalize_audio(audio)
        
        # Stage 1: Harmonic pre-filter
        has_harmonics = self.harmonic_filter.has_drone_harmonics(audio)
        
        if not has_harmonics:
            # Fast rejection
            return {
                'detected': False,
                'class_name': 'Non-Drone',
                'class_id': 0,
                'confidence': 1.0,
                'is_ood': False,
                'latency_ms': (time.time() - start_time) * 1000,
                'stage': 'harmonic_filter'
            }
        
        # Stage 2: Classification
        mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
        mel_spec_tensor = mel_spec_tensor.to(self.device)
        
        with torch.no_grad():
            logits, embeddings = self.model(mel_spec_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            confidence = confidence.item()
            predicted = predicted.item()
            embeddings = embeddings.cpu().numpy()[0]
        
        # Stage 3: OOD detection
        is_ood = self.ood_detector.is_ood(embeddings, predicted)
        
        # Stage 4: Temporal smoothing
        smoothed_pred, smoothed_conf = self.temporal_smoother.update(
            predicted, confidence
        )
        
        # Determine detection
        detected = smoothed_conf >= CONFIDENCE_THRESHOLD and smoothed_pred > 0
        
        result = {
            'detected': detected,
            'class_name': CLASS_NAMES[smoothed_pred] if not is_ood else 'Unknown Drone',
            'class_id': int(smoothed_pred),
            'confidence': float(smoothed_conf),
            'is_ood': is_ood,
            'latency_ms': (time.time() - start_time) * 1000,
            'stage': 'full_pipeline',
            'raw_prediction': int(predicted),
            'raw_confidence': float(confidence),
            'dominant_frequency': self.harmonic_filter.get_dominant_frequency(audio)
        }
        
        return result
    
    def reset(self):
        """Reset temporal smoother state"""
        self.temporal_smoother.reset()
```

### BACKEND: backend/main.py
"""
FastAPI server with WebSocket support for real-time audio streaming
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import numpy as np
import json
from typing import List
import asyncio
import base64

from backend.inference.pipeline import InferencePipeline
from backend.config import HOST, PORT, SAMPLE_RATE, CHUNK_SAMPLES

# Initialize FastAPI app
app = FastAPI(title="SkyGuard Tactical API")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference pipeline
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize inference pipeline on startup"""
    global pipeline
    print("Initializing inference pipeline...")
    pipeline = InferencePipeline()
    print("✓ Server ready")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "SkyGuard Tactical",
        "version": "1.0.0"
    }

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "pipeline_loaded": pipeline is not None,
        "sample_rate": SAMPLE_RATE,
        "chunk_size": CHUNK_SAMPLES,
    }

@app.post("/api/reset")
async def reset_pipeline():
    """Reset temporal smoother state"""
    if pipeline:
        pipeline.reset()
        return {"status": "reset"}
    raise HTTPException(status_code=500, detail="Pipeline not initialized")

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming
    
    Expected message format:
    {
        "type": "audio",
        "data": "base64_encoded_audio",
        "sample_rate": 16000
    }
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'audio':
                # Decode base64 audio
                audio_bytes = base64.b64decode(message['data'])
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Process through pipeline
                result = pipeline.process_audio(audio)
                
                # Send result back
                await websocket.send_json({
                    "type": "detection",
                    "data": result
                })
                
                # Broadcast to all clients
                await manager.broadcast({
                    "type": "detection",
                    "data": result
                })
            
            elif message['type'] == 'ping':
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        manager.disconnect(websocket)

@app.post("/api/process")
async def process_audio_file(audio_base64: str):
    """
    Process audio file (for testing)
    
    Args:
        audio_base64: Base64 encoded audio data
    
    Returns:
        Detection results
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # Process
        result = pipeline.process_audio(audio)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
```

### BACKEND: backend/requirements.txt
```
# Core dependencies
numpy>=1.24.0
scipy>=1.10.0
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0
pyaudio>=0.2.13

# Server
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
websockets>=11.0.0
python-multipart>=0.0.6

# Utilities
tqdm>=4.65.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

### BACKEND: backend/Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## **PART 3: Frontend Components**

### FRONTEND: frontend/package.json
```json
{
  "name": "skyguard-tactical-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.0.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^5.2.0",
    "@types/node": "^20.8.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "tailwindcss": "^3.3.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "chart.js": "^4.4.0",
    "react-chartjs-2": "^5.2.0",
    "leaflet": "^1.9.4",
    "react-leaflet": "^4.2.1",
    "@types/leaflet": "^1.9.8",
    "lucide-react": "^0.263.1",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.0.0"
  },
  "devDependencies": {
    "eslint": "^8.50.0",
    "eslint-config-next": "14.0.0"
  }
}
```

### FRONTEND: frontend/lib/websocket.ts
```typescript
/**
 * WebSocket client for real-time audio streaming
 */

export interface DetectionResult {
  detected: boolean;
  class_name: string;
  class_id: number;
  confidence: number;
  is_ood: boolean;
  latency_ms: number;
  stage: string;
  raw_prediction?: number;
  raw_confidence?: number;
  dominant_frequency?: number;
}

export class AudioWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  
  constructor(url: string = 'ws://localhost:8000/ws/audio') {
    this.url = url;
  }
  
  connect(
    onMessage: (result: DetectionResult) => void,
    onError?: (error: Event) => void,
    onClose?: (event: CloseEvent) => void
  ): void {
    try {
      this.ws = new WebSocket(this.url);
      
      this.ws.onopen = () => {
        console.log('✓ WebSocket connected');
        this.reconnectAttempts = 0;
      };
      
      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'detection') {
          onMessage(message.data);
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (onError) onError(error);
      };
      
      this.ws.onclose = (event) => {
        console.log('WebSocket closed');
        if (onClose) onClose(event);
        
        // Auto-reconnect
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
          setTimeout(() => this.connect(onMessage, onError, onClose), this.reconnectDelay);
          this.reconnectDelay *= 2; // Exponential backoff
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
    }
  }
  
  sendAudio(audioData: Float32Array, sampleRate: number = 16000): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      // Convert to base64
      const bytes = new Uint8Array(audioData.buffer);
      const base64 = btoa(String.fromCharCode(...bytes));
      
      const message = {
        type: 'audio',
        data: base64,
        sample_rate: sampleRate
      };
      
      this.ws.send(JSON.stringify(message));
    }
  }
  
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
  
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}
```

### FRONTEND: frontend/components/ThreatStatus.tsx
```typescript
/**
 * Threat status indicator (Red/Yellow/Green)
 */
import React from 'react';

interface ThreatStatusProps {
  detected: boolean;
  confidence: number;
  className?: string;
}

export const ThreatStatus: React.FC = ({
  detected,
  confidence,
  className = ''
}) => {
  const getThreatLevel = () => {
    if (!detected) return 'clear';
    if (confidence >= 0.9) return 'high';
    if (confidence >= 0.7) return 'medium';
    return 'low';
  };
  
  const level = getThreatLevel();
  
  const statusConfig = {
    clear: {
      bg: 'bg-green-500',
      text: 'ALL CLEAR',
      textColor: 'text-green-900',
      ring: 'ring-green-600'
    },
    low: {
      bg: 'bg-yellow-500',
      text: 'POSSIBLE THREAT',
      textColor: 'text-yellow-900',
      ring: 'ring-yellow-600'
    },
    medium: {
      bg: 'bg-orange-500',
      text: 'THREAT DETECTED',
      textColor: 'text-orange-900',
      ring: 'ring-orange-600'
    },
    high: {
      bg: 'bg-red-600',
      text: 'HIGH THREAT',
      textColor: 'text-red-100',
      ring: 'ring-red-700'
    }
  };
  
  const config = statusConfig[level];
  
  return (
    
      
        
          {config.text}
        
        {detected && (
          
            Confidence: {(confidence * 100).toFixed(1)}%
          
        )}
      
    
  );
};
```

### FRONTEND: frontend/components/ClassificationDisplay.tsx
```typescript
/**
 * Displays classification results with confidence bars
 */
import React from 'react';
import { AlertCircle } from 'lucide-react';

interface ClassificationDisplayProps {
  className: string;
  confidence: number;
  isOOD: boolean;
  latencyMs: number;
}

export const ClassificationDisplay: React.FC = ({
  className,
  confidence,
  isOOD,
  latencyMs
}) => {
  return (
    
      Detection Results
      
      
        {/* Class Name */}
        
          
            Classification
          
          
            {className}
            {isOOD && (
              
                
                Unknown
              
            )}
          
        
        
        {/* Confidence Bar */}
        
          
            Confidence
            {(confidence * 100).toFixed(1)}%
          
          
            = 0.9 ? 'bg-green-500' :
                confidence >= 0.7 ? 'bg-yellow-500' :
                'bg-orange-500'
              }`}
              style={{ width: `${confidence * 100}%` }}
            />
          
        
        
        {/* Latency */}
        
          Processing Time
          
            {latencyMs.toFixed(0)}ms
          
        
      
    
  );
};
```

---