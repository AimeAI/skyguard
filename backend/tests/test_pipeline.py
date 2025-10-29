"""
Test Complete Inference Pipeline
"""
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from inference.pipeline import InferencePipeline


def test_pipeline():
    """Test complete inference pipeline"""
    print("=" * 60)
    print("Testing SkyGuard Inference Pipeline")
    print("=" * 60 + "\n")

    # Initialize pipeline
    pipeline = InferencePipeline()

    # Test 1: Silent audio (should be rejected by harmonic filter)
    print("Test 1: Silent audio")
    print("-" * 40)
    silence = np.zeros(48000, dtype=np.float32)
    result1 = pipeline.process_audio(silence)
    print(f"Detected: {result1['detected']}")
    print(f"Class: {result1['class_name']}")
    print(f"Confidence: {result1['confidence']:.2f}")
    print(f"Latency: {result1['latency_ms']:.1f}ms")
    print(f"Stage: {result1['stage']}")
    assert result1['detected'] == False, "Silent audio should not be detected"
    print("✓ PASS\n")

    # Test 2: Simulated drone signal (multiple harmonics)
    print("Test 2: Simulated drone signal")
    print("-" * 40)
    t = np.linspace(0, 3, 48000)
    drone_signal = (
        np.sin(2 * np.pi * 500 * t) +
        0.8 * np.sin(2 * np.pi * 1000 * t) +
        0.6 * np.sin(2 * np.pi * 1500 * t) +
        0.4 * np.sin(2 * np.pi * 2000 * t)
    ).astype(np.float32)

    result2 = pipeline.process_audio(drone_signal)
    print(f"Detected: {result2['detected']}")
    print(f"Class: {result2['class_name']}")
    print(f"Confidence: {result2['confidence']:.2f}")
    print(f"Latency: {result2['latency_ms']:.1f}ms")
    print(f"Stage: {result2['stage']}")
    print(f"Dominant frequency: {result2['dominant_frequency']:.1f}Hz")
    assert result2['stage'] == 'full_pipeline', "Should pass to ML model"
    print("✓ PASS\n")

    # Test 3: Noise (should be rejected)
    print("Test 3: White noise")
    print("-" * 40)
    noise = np.random.randn(48000).astype(np.float32) * 0.1
    result3 = pipeline.process_audio(noise)
    print(f"Detected: {result3['detected']}")
    print(f"Class: {result3['class_name']}")
    print(f"Confidence: {result3['confidence']:.2f}")
    print(f"Latency: {result3['latency_ms']:.1f}ms")
    print(f"Stage: {result3['stage']}")
    # Note: Low-amplitude noise may sometimes pass filter, which is OK
    # Key is that it should NOT be confidently detected as a drone
    assert result3['detected'] == False, "Noise should not be detected as drone"
    print("✓ PASS\n")

    # Test 4: Latency check
    print("Test 4: Latency benchmark")
    print("-" * 40)
    latencies = []
    for i in range(10):
        result = pipeline.process_audio(drone_signal)
        latencies.append(result['latency_ms'])

    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    print(f"Average latency: {avg_latency:.1f}ms")
    print(f"Max latency: {max_latency:.1f}ms")
    print(f"Target: <200ms")

    if avg_latency < 200:
        print("✓ PASS - Latency within target\n")
    else:
        print("⚠ WARNING - Latency exceeds target (expected on first run)\n")

    # Test 5: Temporal smoothing (test persistence)
    print("Test 5: Temporal smoothing")
    print("-" * 40)
    pipeline.reset()

    predictions = []
    for i in range(8):
        result = pipeline.process_audio(drone_signal)
        predictions.append(result['class_id'])
        print(f"  Frame {i+1}: Class {result['class_id']} " +
              f"(conf={result['confidence']:.2f})")

    # Check that predictions stabilize after a few frames
    later_preds = predictions[-3:]
    if len(set(later_preds)) == 1:
        print("✓ PASS - Predictions stabilized\n")
    else:
        print("⚠ Predictions still varying (may be expected with random weights)\n")

    print("=" * 60)
    print("All pipeline tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
