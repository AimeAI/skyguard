#!/usr/bin/env python3
"""
Generate synthetic drone audio based on documented acoustic signatures
for Class 1 UAS models
"""
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import SAMPLE_RATE, CHUNK_DURATION

def generate_drone_audio(drone_type, duration=3.0, sample_rate=16000):
    """
    Generate synthetic audio matching documented acoustic signatures
    for specific Class 1 UAS models
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.zeros_like(t)

    if drone_type == "DJI Mavic 3":
        # High-pitched 3 kHz harmonic tone
        fundamentals = [3000, 6000, 9000]  # 3kHz fundamental + harmonics
        for i, freq in enumerate(fundamentals):
            amplitude = 0.3 / (i + 1)
            signal += amplitude * np.sin(2 * np.pi * freq * t)

    elif drone_type == "DJI Air 3" or drone_type == "DJI Air 2S":
        # Mid-frequency whine, modulated RPM
        base_freq = 2200
        modulation = 200 * np.sin(2 * np.pi * 2 * t)  # 2Hz RPM modulation
        for harmonic in range(1, 5):
            freq = base_freq * harmonic + modulation
            amplitude = 0.25 / harmonic
            signal += amplitude * np.sin(2 * np.pi * freq * t)

    elif drone_type == "DJI Mini 4 Pro" or drone_type == "DJI Mini 3 Pro":
        # Light 2.5 kHz prop whine, rapid harmonics
        fundamentals = [2500, 5000, 7500, 10000]
        for i, freq in enumerate(fundamentals):
            amplitude = 0.2 / (i + 1)
            signal += amplitude * np.sin(2 * np.pi * freq * t)

    elif drone_type == "DJI Phantom 4 Pro":
        # Classic "drone buzz," 1.8–2.2 kHz band
        freq_range = np.random.uniform(1800, 2200)
        for harmonic in range(1, 6):
            freq = freq_range * harmonic
            amplitude = 0.3 / harmonic
            signal += amplitude * np.sin(2 * np.pi * freq * t)

    elif drone_type == "Autel EVO II Pro":
        # Similar to Phantom but deeper tone (1.5-1.9 kHz)
        freq_range = np.random.uniform(1500, 1900)
        for harmonic in range(1, 6):
            freq = freq_range * harmonic
            amplitude = 0.35 / harmonic
            signal += amplitude * np.sin(2 * np.pi * freq * t)

    elif drone_type == "Skydio 2+":
        # Variable RPM modulation; distinct AI self-navigation pauses
        base_freq = 2000
        # Add pauses (AI navigation moments)
        pause_mask = np.ones_like(t)
        pause_intervals = np.random.choice([True, False], size=len(t), p=[0.15, 0.85])
        pause_mask[pause_intervals] = 0.1

        # Variable RPM
        rpm_variation = 300 * np.sin(2 * np.pi * 1.5 * t)
        for harmonic in range(1, 5):
            freq = base_freq * harmonic + rpm_variation
            amplitude = 0.25 / harmonic
            signal += amplitude * np.sin(2 * np.pi * freq * t) * pause_mask

    elif drone_type == "Parrot Anafi":
        # Quieter, higher pitch (2.8-3.2 kHz), foldable lightweight
        base_freq = np.random.uniform(2800, 3200)
        for harmonic in range(1, 4):
            freq = base_freq * harmonic
            amplitude = 0.18 / harmonic  # Quieter
            signal += amplitude * np.sin(2 * np.pi * freq * t)

    elif drone_type == "DJI Inspire 2":
        # Professional/cinematic, heavier (3.3kg), deeper rumble (1.2-1.6 kHz)
        base_freq = np.random.uniform(1200, 1600)
        # Add motor rumble
        for harmonic in range(1, 7):
            freq = base_freq * harmonic
            amplitude = 0.4 / harmonic  # Louder, professional grade
            signal += amplitude * np.sin(2 * np.pi * freq * t)
        # Add mechanical vibration
        vibration = 0.1 * np.sin(2 * np.pi * 80 * t)  # 80 Hz mechanical
        signal += vibration

    else:
        # Non-Drone (background noise)
        signal = np.random.randn(len(t)) * 0.05

    # Add slight background noise
    signal += np.random.randn(len(t)) * 0.02

    # Normalize
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.7

    return signal.astype(np.float32)


def generate_training_dataset(output_dir, samples_per_class=100):
    """Generate synthetic training dataset"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    drone_models = [
        "Non-Drone",
        "DJI Mavic 3",
        "DJI Air 3",
        "DJI Mini 4 Pro",
        "DJI Phantom 4 Pro",
        "Autel EVO II Pro",
        "Skydio 2+",
        "DJI Air 2S",
        "DJI Mini 3 Pro",
        "Parrot Anafi",
        "DJI Inspire 2",
    ]

    print(f"Generating synthetic training data...")
    print(f"Output: {output_dir}")
    print(f"Samples per class: {samples_per_class}")
    print("=" * 60)

    all_data = []

    for drone_type in drone_models:
        print(f"\nGenerating {drone_type}...")
        class_dir = output_dir / drone_type.replace(" ", "_")
        class_dir.mkdir(exist_ok=True)

        for i in range(samples_per_class):
            audio = generate_drone_audio(drone_type, duration=CHUNK_DURATION)

            # Save as numpy array
            filename = class_dir / f"{drone_type.replace(' ', '_')}_{i:04d}.npy"
            np.save(filename, audio)

            all_data.append({
                'file_path': str(filename),
                'label': drone_type,
                'duration': CHUNK_DURATION,
                'sample_rate': SAMPLE_RATE
            })

        print(f"  ✓ Generated {samples_per_class} samples")

    # Save metadata
    import json
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump({
            'num_classes': len(drone_models),
            'classes': drone_models,
            'samples_per_class': samples_per_class,
            'total_samples': len(all_data),
            'sample_rate': SAMPLE_RATE,
            'duration': CHUNK_DURATION,
            'samples': all_data
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✓ Generated {len(all_data)} total samples")
    print(f"✓ Metadata saved to: {metadata_file}")
    print("=" * 60)

    return all_data


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent.parent / "data" / "synthetic_drone_audio"
    generate_training_dataset(output_dir, samples_per_class=150)
