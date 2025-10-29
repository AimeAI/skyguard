"""
Organize downloaded datasets into train/val/test splits
"""
import os
import shutil
from pathlib import Path
import random
import json
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def get_audio_files(directory):
    """Get all audio files from directory"""
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(Path(root) / file)

    return audio_files

def organize_droneaudioset():
    """Organize DroneAudioset into train/val/test"""
    print("\n" + "="*60)
    print("Organizing DroneAudioset")
    print("="*60)

    drone_dir = RAW_DIR / "DroneAudioSet"

    if not drone_dir.exists():
        print("⚠ DroneAudioset not found")
        return

    # Get all audio files
    audio_files = get_audio_files(drone_dir)
    print(f"Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        print("⚠ No audio files found - dataset may still be downloading")
        return

    # Group by drone type (infer from filename or path)
    drone_types = defaultdict(list)

    for audio_file in audio_files:
        # Try to infer drone type from path or filename
        # This will depend on the actual dataset structure
        parent_dir = audio_file.parent.name
        drone_types[parent_dir].append(audio_file)

    print(f"Found {len(drone_types)} drone types")

    # Map to our 10 classes
    drone_classes = list(drone_types.keys())[:10]  # Take first 10

    for i, drone_class in enumerate(drone_classes, 1):
        files = drone_types[drone_class]
        random.shuffle(files)

        # Split files
        n_train = int(len(files) * TRAIN_RATIO)
        n_val = int(len(files) * VAL_RATIO)

        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]

        # Copy to train/val/test
        class_name = f"Drone_Model_{i}"

        for split, split_files in [('train', train_files),
                                    ('val', val_files),
                                    ('test', test_files)]:
            split_dir = RAW_DIR / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for src_file in split_files:
                dst_file = split_dir / src_file.name
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)

        print(f"✓ {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

def organize_esc50():
    """Organize ESC-50 environmental sounds for Non-Drone class"""
    print("\n" + "="*60)
    print("Organizing ESC-50 (Non-Drone class)")
    print("="*60)

    esc_dir = RAW_DIR / "ESC-50"

    if not esc_dir.exists():
        print("⚠ ESC-50 not found")
        return

    # Get relevant environmental sounds
    audio_files = get_audio_files(esc_dir)

    if len(audio_files) == 0:
        print("⚠ No audio files found in ESC-50")
        return

    print(f"Found {len(audio_files)} environmental sounds")

    # Shuffle and split
    random.shuffle(audio_files)
    n_train = int(len(audio_files) * TRAIN_RATIO)
    n_val = int(len(audio_files) * VAL_RATIO)

    train_files = audio_files[:n_train]
    val_files = audio_files[n_train:n_train+n_val]
    test_files = audio_files[n_train+n_val:]

    # Copy to Non-Drone class
    for split, split_files in [('train', train_files),
                                ('val', val_files),
                                ('test', test_files)]:
        split_dir = RAW_DIR / split / "Non-Drone"
        split_dir.mkdir(parents=True, exist_ok=True)

        for src_file in split_files:
            dst_file = split_dir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)

    print(f"✓ Non-Drone: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

def create_dataset_stats():
    """Create statistics about the organized dataset"""
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)

    stats = {}

    for split in ['train', 'val', 'test']:
        split_dir = RAW_DIR / split
        if not split_dir.exists():
            continue

        stats[split] = {}

        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                audio_files = get_audio_files(class_dir)
                stats[split][class_dir.name] = len(audio_files)

        total = sum(stats[split].values())
        print(f"\n{split.upper()}: {total} files")
        for class_name, count in sorted(stats[split].items()):
            print(f"  {class_name}: {count}")

    # Save stats to JSON
    stats_file = BASE_DIR / "data" / "dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Stats saved to {stats_file}")

    return stats

def main():
    print("="*60)
    print("SkyGuard Dataset Organization")
    print("="*60)

    # Set random seed for reproducibility
    random.seed(42)

    # Create directory structure
    for split in ['train', 'val', 'test']:
        for i in range(11):  # 11 classes
            class_name = f"Drone_Model_{i}" if i > 0 else "Non-Drone"
            class_dir = RAW_DIR / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

    # Organize datasets
    organize_droneaudioset()
    organize_esc50()

    # Create statistics
    stats = create_dataset_stats()

    print("\n" + "="*60)
    print("✓ Dataset organization complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review dataset statistics above")
    print("2. Run training: python3 backend/training/train.py")

if __name__ == "__main__":
    main()
