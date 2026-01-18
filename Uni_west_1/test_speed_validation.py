"""
Quick test script for speed validation improvements.
Tests speed calculation on a small sample of frames.
"""

import sys
# Insert Uni_west_1 first to get the correct version
sys.path.insert(0, '/Users/navaneethmalingan/3D/Uni_west_1')
sys.path.insert(1, '/Users/navaneethmalingan/3D')

import cv2 as cv
import numpy as np
from speed_estimator import SpeedEstimator, VehicleTracker

DATASET_DIR = "/Users/navaneethmalingan/3D/Uni_west_1"
LOOKUP_TABLE = f"{DATASET_DIR}/calibration-lookup-table.npy"

def main():
    import sys
    print("=" * 60, flush=True)
    print("SPEED VALIDATION TEST", flush=True)
    print("=" * 60, flush=True)
    print(f"Speed thresholds:", flush=True)
    print(f"  MAX_SPEED_KMH = {VehicleTracker.MAX_SPEED_KMH} km/h", flush=True)
    print(f"  MIN_SPEED_KMH = {VehicleTracker.MIN_SPEED_KMH} km/h", flush=True)
    print(f"  MAX_POSITION_JUMP = {VehicleTracker.MAX_POSITION_JUMP} m", flush=True)
    sys.stdout.flush()

    estimator = SpeedEstimator(LOOKUP_TABLE)

    # Process 300 frames from first video
    video_path = f"{DATASET_DIR}/GOPR0574.MP4"
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    estimator.fps = fps

    print(f"\nProcessing 300 frames from {video_path}...")
    print(f"Video FPS: {fps}")

    all_speeds = []
    all_positions = []

    for frame_idx in range(300):
        ret, frame = cap.read()
        if not ret:
            break

        output, tracks = estimator.process_frame(frame)

        for track_id, track in tracks.items():
            if track.speeds and track.frames_since_seen == 0:
                speed = list(track.speeds)[-1]
                pos = track.positions[-1] if track.positions else (0, 0)
                all_speeds.append(speed)
                all_positions.append(pos)

        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/300")

    cap.release()

    print(f"\nResults:")
    print(f"  Total speed samples: {len(all_speeds)}")

    if all_speeds:
        valid_speeds = [s for s in all_speeds if s > 0]
        print(f"  Non-zero speeds: {len(valid_speeds)}")

        if valid_speeds:
            print(f"  Speed range: {min(valid_speeds):.1f} - {max(valid_speeds):.1f} km/h")
            print(f"  Average speed: {np.mean(valid_speeds):.1f} km/h")
            print(f"  Std dev: {np.std(valid_speeds):.1f}")

            # Check for any speeds over threshold
            over_threshold = [s for s in all_speeds if s > VehicleTracker.MAX_SPEED_KMH]
            print(f"  Over {VehicleTracker.MAX_SPEED_KMH} km/h: {len(over_threshold)}")

    if all_positions:
        x_coords = [p[0] for p in all_positions]
        y_coords = [p[1] for p in all_positions]
        print(f"\n  Position X range: {min(x_coords):.1f} to {max(x_coords):.1f} m")
        print(f"  Position Y range: {min(y_coords):.1f} to {max(y_coords):.1f} m")

if __name__ == "__main__":
    main()
