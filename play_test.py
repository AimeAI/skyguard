#!/usr/bin/env python3
"""Play test audio files to test the detection system"""
import numpy as np
import subprocess
import sys
from pathlib import Path

def play_audio(audio_array, sample_rate=16000):
    """Play audio using macOS 'afplay' command"""
    # Save as temporary wav file
    import wave
    import struct

    temp_file = "temp_test.wav"

    # Normalize to 16-bit PCM
    audio_int16 = np.int16(audio_array * 32767)

    with wave.open(temp_file, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes (16-bit)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    print(f"Playing audio... (press Ctrl+C to stop)")
    subprocess.run(['afplay', temp_file])

    # Cleanup
    Path(temp_file).unlink()

if __name__ == "__main__":
    test_files = [
        "test_DJI_Mavic_3.npy",
        "test_DJI_Phantom_4_Pro.npy",
        "test_DJI_Mini_4_Pro.npy",
        "test_Skydio_2+.npy",
        "test_Non-Drone.npy"
    ]

    print("Available test files:")
    for i, f in enumerate(test_files, 1):
        print(f"{i}. {f}")

    print("\nSelect a file to play (1-5), or 'a' to play all:")
    choice = input("> ").strip()

    if choice == 'a':
        for test_file in test_files:
            if Path(test_file).exists():
                print(f"\n{'='*60}")
                print(f"Playing: {test_file}")
                print('='*60)
                audio = np.load(test_file)
                play_audio(audio)
            else:
                print(f"File not found: {test_file}")
    elif choice.isdigit() and 1 <= int(choice) <= 5:
        test_file = test_files[int(choice) - 1]
        if Path(test_file).exists():
            print(f"\nPlaying: {test_file}")
            audio = np.load(test_file)
            play_audio(audio)
        else:
            print(f"File not found: {test_file}")
    else:
        print("Invalid choice")
