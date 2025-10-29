#!/usr/bin/env python3
"""Generate test audio for different drone types"""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'backend'))
from training.generate_synthetic_drone_audio import generate_drone_audio

# Generate 5 seconds of audio for each drone type
drones = [
    "DJI Mavic 3",
    "DJI Phantom 4 Pro",
    "DJI Mini 4 Pro",
    "Skydio 2+",
    "Non-Drone"
]

print("Generating test audio files...")
for drone in drones:
    audio = generate_drone_audio(drone, duration=5.0)
    filename = f"test_{drone.replace(' ', '_')}.npy"
    np.save(filename, audio)
    print(f"✓ Generated {filename}")

print("\nTo test:")
print("1. Open frontend/index.html in your browser")
print("2. Click 'Connect to Backend'")
print("3. Click 'Start Detection'")
print("4. Play one of these test files through your speakers:")
for drone in drones:
    print(f"   - test_{drone.replace(' ', '_')}.npy")
