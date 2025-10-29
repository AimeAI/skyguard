"""
Test Harmonic Filter
"""
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.harmonic_filter import HarmonicFilter


def test_harmonic_filter():
    """Test harmonic filter with synthetic signals"""
    filter = HarmonicFilter()

    # Test 1: Pure sine wave (should NOT pass - only 1 peak)
    print("Test 1: Pure sine wave (1000 Hz)")
    t = np.linspace(0, 3, 48000)
    pure_sine = np.sin(2 * np.pi * 1000 * t)
    result1 = filter.has_drone_harmonics(pure_sine)
    print(f"  Has drone harmonics: {result1}")
    print(f"  Expected: False (only 1 peak)\n")

    # Test 2: Multiple harmonics (simulated drone)
    print("Test 2: Multiple harmonics (500, 1000, 1500, 2000 Hz)")
    drone_signal = (
        np.sin(2 * np.pi * 500 * t) +
        0.8 * np.sin(2 * np.pi * 1000 * t) +
        0.6 * np.sin(2 * np.pi * 1500 * t) +
        0.4 * np.sin(2 * np.pi * 2000 * t)
    )
    result2 = filter.has_drone_harmonics(drone_signal)
    dominant_freq = filter.get_dominant_frequency(drone_signal)
    print(f"  Has drone harmonics: {result2}")
    print(f"  Dominant frequency: {dominant_freq:.1f} Hz")
    print(f"  Expected: True (4 peaks)\n")

    # Test 3: White noise (should NOT pass)
    print("Test 3: White noise")
    noise = np.random.randn(48000) * 0.1
    result3 = filter.has_drone_harmonics(noise)
    print(f"  Has drone harmonics: {result3}")
    print(f"  Expected: False (no clear peaks)\n")

    # Test 4: Low frequency sound (out of range)
    print("Test 4: Low frequency (100 Hz, below 500 Hz threshold)")
    low_freq = np.sin(2 * np.pi * 100 * t)
    result4 = filter.has_drone_harmonics(low_freq)
    print(f"  Has drone harmonics: {result4}")
    print(f"  Expected: False (out of drone freq range)\n")

    print("=" * 50)
    print("SUMMARY:")
    print(f"  Test 1 (Pure sine): {'PASS' if not result1 else 'FAIL'}")
    print(f"  Test 2 (Multi-harmonic): {'PASS' if result2 else 'FAIL'}")
    print(f"  Test 3 (Noise): {'PASS' if not result3 else 'FAIL'}")
    print(f"  Test 4 (Low freq): {'PASS' if not result4 else 'FAIL'}")

    all_pass = (not result1) and result2 and (not result3) and (not result4)
    print(f"\nAll tests: {'✓ PASSED' if all_pass else '✗ FAILED'}")


if __name__ == "__main__":
    test_harmonic_filter()
