"""
Vehicle Speed Estimator for Uni_west_1 Dataset

Processes videos and outputs annotated videos with:
- 3D bounding boxes
- Vehicle tracking
- Speed estimation in km/h

Usage:
    python speed_estimator.py              # Process all videos
    python speed_estimator.py 0            # Process first video only
    python speed_estimator.py 1 2          # Process videos at index 1 and 2
"""

import sys
sys.path.insert(0, '/Users/navaneethmalingan/3D')

import cv2 as cv
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque, defaultdict
import os

# Use calibrated generator for Uni_west_1 dataset
from bbox_3d_calibrated import BBox3DGenerator, BBox3D, BBox2D

# Dataset paths
DATASET_DIR = "/Users/navaneethmalingan/3D/Uni_west_1"
LOOKUP_TABLE = f"{DATASET_DIR}/calibration-lookup-table.npy"
OUTPUT_DIR = f"{DATASET_DIR}/output"

# Videos
VIDEOS = [
    "GOPR0574.MP4",
    "GOPR0575.MP4",
    "GOPR0581.MP4"
]


@dataclass
class TrackedVehicle:
    """Represents a tracked vehicle across frames."""
    track_id: int
    positions: deque  # World coordinates (x, y) over time
    timestamps: deque  # Frame timestamps
    speeds: deque     # Calculated speeds (km/h)
    bbox_3d: Optional[BBox3D] = None
    color: Tuple[int, int, int] = (0, 255, 0)
    frames_since_seen: int = 0

    def __post_init__(self):
        if not isinstance(self.positions, deque):
            self.positions = deque(maxlen=30)
        if not isinstance(self.timestamps, deque):
            self.timestamps = deque(maxlen=30)
        if not isinstance(self.speeds, deque):
            self.speeds = deque(maxlen=30)


class VehicleTracker:
    """Tracks vehicles across frames and calculates speeds."""

    # Speed validation thresholds
    MAX_SPEED_KMH = 150.0       # Maximum realistic speed (km/h)
    MIN_SPEED_KMH = 0.5         # Minimum speed to consider moving
    MAX_POSITION_JUMP = 5.0     # Maximum position jump between frames (meters)

    def __init__(self, iou_threshold: float = 0.3, max_frames_lost: int = 15):
        self.tracks: Dict[int, TrackedVehicle] = {}
        self.next_track_id = 0
        self.iou_threshold = iou_threshold
        self.max_frames_lost = max_frames_lost

        # Generate random colors for tracks
        np.random.seed(42)
        self.colors = [(int(c[0]), int(c[1]), int(c[2]))
                       for c in np.random.randint(50, 255, (100, 3))]

    def _compute_iou(self, box1: BBox2D, box2: BBox2D) -> float:
        """Compute IoU between two 2D bounding boxes."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, detections: List[BBox3D], frame_time: float) -> Dict[int, TrackedVehicle]:
        """Update tracks with new detections."""
        # Mark all tracks as not seen this frame
        for track in self.tracks.values():
            track.frames_since_seen += 1

        # Match detections to existing tracks
        unmatched_detections = list(range(len(detections)))

        if self.tracks and detections:
            # Compute IoU matrix
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(detections)))

            for i, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                if track.bbox_3d is not None:
                    for j, det in enumerate(detections):
                        iou_matrix[i, j] = self._compute_iou(track.bbox_3d.bbox_2d, det.bbox_2d)

            # Greedy matching
            matched_dets = set()

            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break

                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                track_id = track_ids[i]

                # Update track
                track = self.tracks[track_id]
                det = detections[j]

                track.bbox_3d = det
                new_pos = (det.center[0], det.center[1])

                # Validate position - check for unrealistic jumps
                if track.positions:
                    last_pos = track.positions[-1]
                    jump_dist = np.sqrt((new_pos[0] - last_pos[0])**2 + (new_pos[1] - last_pos[1])**2)

                    # Skip this position if jump is too large (likely calibration error)
                    if jump_dist > self.MAX_POSITION_JUMP:
                        # Don't update position, keep the old one
                        track.frames_since_seen = 0
                        continue

                track.positions.append(new_pos)
                track.timestamps.append(frame_time)
                track.frames_since_seen = 0

                # Calculate speed if we have enough history
                if len(track.positions) >= 2:
                    speed = self._calculate_speed(track)
                    # Validate speed before adding
                    if speed <= self.MAX_SPEED_KMH:
                        track.speeds.append(speed)
                    elif track.speeds:
                        # Use last valid speed instead
                        track.speeds.append(track.speeds[-1])

                matched_dets.add(j)

                # Remove matched row and column
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

            unmatched_detections = [j for j in range(len(detections)) if j not in matched_dets]

        # Create new tracks for unmatched detections
        for j in unmatched_detections:
            det = detections[j]
            track = TrackedVehicle(
                track_id=self.next_track_id,
                positions=deque(maxlen=30),
                timestamps=deque(maxlen=30),
                speeds=deque(maxlen=30),
                bbox_3d=det,
                color=self.colors[self.next_track_id % len(self.colors)]
            )
            track.positions.append((det.center[0], det.center[1]))
            track.timestamps.append(frame_time)

            self.tracks[self.next_track_id] = track
            self.next_track_id += 1

        # Remove lost tracks
        lost_tracks = [tid for tid, t in self.tracks.items()
                       if t.frames_since_seen > self.max_frames_lost]
        for tid in lost_tracks:
            del self.tracks[tid]

        return self.tracks

    def _calculate_speed(self, track: TrackedVehicle) -> float:
        """Calculate speed from recent positions (in km/h) with smoothing."""
        if len(track.positions) < 2:
            return 0.0

        # Use positions from last few frames for smoothing
        n_frames = min(15, len(track.positions))

        positions = list(track.positions)[-n_frames:]
        timestamps = list(track.timestamps)[-n_frames:]

        # Validate positions are within reasonable world coordinate range
        for pos in positions:
            if abs(pos[0]) > 50 or abs(pos[1]) > 50:
                return 0.0  # Invalid world coordinates

        # Calculate displacement in world coordinates (meters)
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        distance = np.sqrt(dx**2 + dy**2)

        # Time elapsed
        dt = timestamps[-1] - timestamps[0]

        if dt <= 0:
            return 0.0

        # Speed in m/s, convert to km/h
        speed_ms = distance / dt
        speed_kmh = speed_ms * 3.6

        # Apply minimum speed threshold (stationary vehicles)
        if speed_kmh < self.MIN_SPEED_KMH:
            return 0.0

        return speed_kmh


class SpeedEstimator:
    """Main class for vehicle speed estimation."""

    def __init__(self, lookup_table_path: str):
        self.generator = BBox3DGenerator(lookup_table_path)
        self.tracker = VehicleTracker(iou_threshold=0.3, max_frames_lost=15)
        self.fps = 30.0
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[int, TrackedVehicle]]:
        """Process a single frame and return annotated image with tracks."""
        # Detect vehicles
        bboxes_3d = self.generator.process_image(frame, conf_threshold=0.3)

        # Update tracker
        frame_time = self.frame_count / self.fps
        tracks = self.tracker.update(bboxes_3d, frame_time)

        # Draw results
        output = self._draw_results(frame, tracks)

        self.frame_count += 1
        return output, tracks

    def _draw_results(self, frame: np.ndarray, tracks: Dict[int, TrackedVehicle]) -> np.ndarray:
        """Draw 3D boxes and speed info on frame."""
        output = frame.copy()

        for track_id, track in tracks.items():
            if track.bbox_3d is None or track.frames_since_seen > 0:
                continue

            bbox = track.bbox_3d
            color = track.color

            # Draw 3D box
            self._draw_3d_box(output, bbox, color)

            # Calculate average speed
            if track.speeds:
                avg_speed = np.mean(list(track.speeds)[-10:])
            else:
                avg_speed = 0.0

            # Draw speed label
            corners = bbox.corners_image
            top_y = int(corners[4:8, 1].min()) - 10
            left_x = int(corners[:, 0].min())

            # Speed text
            speed_text = f"ID:{track_id} {avg_speed:.1f} km/h"

            # Position text
            pos_text = f"({bbox.center[0]:.1f}, {bbox.center[1]:.1f})m"

            # Draw background
            (tw1, th1), _ = cv.getTextSize(speed_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            (tw2, th2), _ = cv.getTextSize(pos_text, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            cv.rectangle(output, (left_x, top_y - th1 - th2 - 10),
                        (left_x + max(tw1, tw2) + 8, top_y + 4), (0, 0, 0), -1)

            cv.putText(output, speed_text, (left_x + 4, top_y - th2 - 4),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv.putText(output, pos_text, (left_x + 4, top_y),
                      cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return output

    def _draw_3d_box(self, image: np.ndarray, bbox: BBox3D, color: Tuple[int, int, int]):
        """Draw 3D wireframe bounding box with corner numbers."""
        corners = bbox.corners_image.astype(int)
        h, w = image.shape[:2]
        corners = np.clip(corners, [0, 0], [w-1, h-1])

        # Back edges (darker)
        back_color = tuple(max(0, c // 2) for c in color)
        cv.line(image, tuple(corners[2]), tuple(corners[3]), back_color, 2)
        cv.line(image, tuple(corners[6]), tuple(corners[7]), back_color, 2)
        cv.line(image, tuple(corners[2]), tuple(corners[6]), back_color, 2)
        cv.line(image, tuple(corners[3]), tuple(corners[7]), back_color, 2)

        # Side edges
        cv.line(image, tuple(corners[0]), tuple(corners[3]), color, 2)
        cv.line(image, tuple(corners[1]), tuple(corners[2]), color, 2)
        cv.line(image, tuple(corners[4]), tuple(corners[7]), color, 2)
        cv.line(image, tuple(corners[5]), tuple(corners[6]), color, 2)

        # Bottom face
        cv.line(image, tuple(corners[0]), tuple(corners[1]), color, 2)
        cv.line(image, tuple(corners[1]), tuple(corners[2]), color, 2)
        cv.line(image, tuple(corners[2]), tuple(corners[3]), color, 2)
        cv.line(image, tuple(corners[3]), tuple(corners[0]), color, 2)

        # Top face
        cv.line(image, tuple(corners[4]), tuple(corners[5]), color, 2)
        cv.line(image, tuple(corners[5]), tuple(corners[6]), color, 2)
        cv.line(image, tuple(corners[6]), tuple(corners[7]), color, 2)
        cv.line(image, tuple(corners[7]), tuple(corners[4]), color, 2)

        # Vertical edges
        cv.line(image, tuple(corners[0]), tuple(corners[4]), color, 2)
        cv.line(image, tuple(corners[1]), tuple(corners[5]), color, 2)

        # Front face highlight (green)
        front_color = (0, 255, 0)
        cv.line(image, tuple(corners[0]), tuple(corners[1]), front_color, 3)
        cv.line(image, tuple(corners[4]), tuple(corners[5]), front_color, 3)
        cv.line(image, tuple(corners[0]), tuple(corners[4]), front_color, 3)
        cv.line(image, tuple(corners[1]), tuple(corners[5]), front_color, 3)

        # Draw corner numbers
        # Bottom corners: 0=front-left, 1=front-right, 2=rear-right, 3=rear-left
        # Top corners: 4=front-left, 5=front-right, 6=rear-right, 7=rear-left
        for i, corner in enumerate(corners):
            # Yellow circle with black outline
            cv.circle(image, tuple(corner), 8, (0, 255, 255), -1)
            cv.circle(image, tuple(corner), 8, (0, 0, 0), 1)
            # Corner number
            cv.putText(image, str(i), (corner[0] - 4, corner[1] + 4),
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    def process_video(self, video_path: str, output_path: str, max_frames: int = None):
        """Process entire video and save result."""
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        self.fps = cap.get(cv.CAP_PROP_FPS)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(f"Video: {width}x{height} @ {self.fps:.1f}fps", flush=True)
        print(f"Processing {total_frames} frames", flush=True)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, self.fps, (width, height))

        self.frame_count = 0
        self.tracker = VehicleTracker()  # Reset tracker

        # Collect speed data for analysis
        all_speeds = defaultdict(list)

        while self.frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            output, tracks = self.process_frame(frame)
            out.write(output)

            # Collect speeds
            for track_id, track in tracks.items():
                if track.speeds and track.frames_since_seen == 0:
                    avg_speed = np.mean(list(track.speeds)[-5:])
                    all_speeds[track_id].append({
                        'frame': self.frame_count,
                        'speed': avg_speed,
                        'pos_x': track.bbox_3d.center[0] if track.bbox_3d else 0,
                        'pos_y': track.bbox_3d.center[1] if track.bbox_3d else 0
                    })

            if self.frame_count % 30 == 0:
                print(f"  Frame {self.frame_count}/{total_frames} ({100*self.frame_count/total_frames:.1f}%)", flush=True)

        cap.release()
        out.release()

        print(f"\nDone! Output saved to: {output_path}", flush=True)

        # Print speed analysis
        self._print_speed_analysis(all_speeds)

        return all_speeds

    def _print_speed_analysis(self, all_speeds: Dict):
        """Print speed analysis summary with validation stats."""
        if not all_speeds:
            print("No speed data collected")
            return

        print("\n" + "=" * 60)
        print("SPEED ANALYSIS")
        print("=" * 60)
        print(f"Validation thresholds: MAX_SPEED={VehicleTracker.MAX_SPEED_KMH} km/h, "
              f"MAX_POSITION_JUMP={VehicleTracker.MAX_POSITION_JUMP}m")

        valid_tracks = 0
        total_samples = 0

        for track_id, data in all_speeds.items():
            if len(data) < 10:
                continue

            speeds = [d['speed'] for d in data]
            # Filter out zero speeds for analysis
            moving_speeds = [s for s in speeds if s > VehicleTracker.MIN_SPEED_KMH]

            if not moving_speeds:
                continue

            valid_tracks += 1
            total_samples += len(moving_speeds)

            avg_speed = np.mean(moving_speeds)
            max_speed = np.max(moving_speeds)
            min_speed = np.min(moving_speeds)
            std_speed = np.std(moving_speeds)

            # Get position range
            x_positions = [d['pos_x'] for d in data]
            y_positions = [d['pos_y'] for d in data]

            print(f"\nTrack {track_id}:")
            print(f"  Avg Speed: {avg_speed:.1f} km/h")
            print(f"  Min/Max: {min_speed:.1f} - {max_speed:.1f} km/h")
            print(f"  Std Dev: {std_speed:.1f}")
            print(f"  Moving samples: {len(moving_speeds)}/{len(speeds)}")
            print(f"  X range: {min(x_positions):.1f} to {max(x_positions):.1f}m")
            print(f"  Y range: {min(y_positions):.1f} to {max(y_positions):.1f}m")

        print(f"\n{'='*60}")
        print(f"Summary: {valid_tracks} valid tracks, {total_samples} total speed samples")


def main():
    print("=" * 60, flush=True)
    print("SPEED ESTIMATOR - Uni_west_1 Dataset", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parse command line arguments
    if len(sys.argv) > 1:
        video_indices = [int(arg) for arg in sys.argv[1:] if arg.isdigit()]
    else:
        video_indices = list(range(len(VIDEOS)))

    print(f"Will process videos: {[VIDEOS[i] for i in video_indices if i < len(VIDEOS)]}", flush=True)

    for idx in video_indices:
        if idx >= len(VIDEOS):
            print(f"Skipping invalid index: {idx}", flush=True)
            continue

        video_name = VIDEOS[idx]
        video_path = f"{DATASET_DIR}/{video_name}"
        output_path = f"{OUTPUT_DIR}/{video_name.replace('.MP4', '_speed.mp4')}"

        print(f"\n{'='*60}", flush=True)
        print(f"Processing: {video_name}", flush=True)
        print(f"{'='*60}", flush=True)

        estimator = SpeedEstimator(LOOKUP_TABLE)

        # Process full video
        estimator.process_video(video_path, output_path, max_frames=None)

    print("\n" + "=" * 60, flush=True)
    print("ALL DONE!", flush=True)
    print(f"Output videos saved to: {OUTPUT_DIR}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
