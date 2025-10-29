"""
Test Temporal Smoother
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.temporal_smoother import TemporalSmoother


def test_temporal_smoother():
    """Test temporal smoother functionality"""
    smoother = TemporalSmoother(window_size=5, hysteresis_threshold=0.7)

    print("Test: Temporal Smoothing with Flickering Predictions\n")
    print("Simulating: Drone A → Brief false detection → Back to Drone A")
    print("=" * 60)

    # Sequence: mostly class 1, with a brief spike of class 2
    predictions = [1, 1, 1, 2, 1, 1, 1, 1]
    confidences = [0.9, 0.85, 0.88, 0.6, 0.87, 0.89, 0.91, 0.92]

    results = []
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        smoothed_pred, smoothed_conf = smoother.update(pred, conf)
        results.append((smoothed_pred, smoothed_conf))
        print(f"Step {i+1}: Raw={pred} (conf={conf:.2f}) → " +
              f"Smoothed={smoothed_pred} (conf={smoothed_conf:.2f})")

    # Check: brief spike to class 2 should be filtered out
    # After step 5+, should stabilize back to class 1
    final_pred = results[-1][0]
    print(f"\n✓ Final prediction: {final_pred}")
    print(f"  Expected: 1 (should filter out brief spike to class 2)")

    if final_pred == 1:
        print("\n✓ PASS: Temporal smoother correctly filtered flickering")
    else:
        print("\n✗ FAIL: Temporal smoother did not smooth properly")

    # Test 2: Strong sustained change
    print("\n" + "=" * 60)
    print("Test 2: Sustained Prediction Change (High Confidence)")
    print("=" * 60)

    smoother.reset()
    predictions2 = [1, 1, 1, 2, 2, 2, 2, 2]
    confidences2 = [0.8, 0.8, 0.8, 0.95, 0.93, 0.92, 0.91, 0.90]

    for i, (pred, conf) in enumerate(zip(predictions2, confidences2)):
        smoothed_pred, smoothed_conf = smoother.update(pred, conf)
        print(f"Step {i+1}: Raw={pred} (conf={conf:.2f}) → " +
              f"Smoothed={smoothed_pred} (conf={smoothed_conf:.2f})")

    # Should eventually change to class 2
    print(f"\n✓ Final prediction: {smoothed_pred}")
    print(f"  Expected: 2 (should accept sustained high-confidence change)")

    if smoothed_pred == 2:
        print("\n✓ PASS: Temporal smoother accepted sustained change")
    else:
        print("\n✗ FAIL: Temporal smoother did not accept sustained change")


if __name__ == "__main__":
    test_temporal_smoother()
