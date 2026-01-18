"""
Vehicle Speed Estimator
Uses 3D bounding box detection and calibration to estimate vehicle speeds.

Approach:
1. Detect vehicles using YOLO + 3D bbox projection
2. Track vehicles across frames using IoU matching
3. Calculate speed from world coordinate displacement over time
"""

import cv2 as cv
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
from bbox_3d_generator import BBox3DGenerator, BBox3D, BBox2D


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
            self.positions = deque(maxlen=30)  # Keep last 30 positions (1 second at 30fps)
        if not isinstance(self.timestamps, deque):
            self.timestamps = deque(maxlen=30)
        if not isinstance(self.speeds, deque):
            self.speeds = deque(maxlen=30)


class VehicleTracker:
    """Tracks vehicles across frames and calculates speeds."""

    def __init__(self, iou_threshold: float = 0.3, max_frames_lost: int = 10):
        self.tracks: Dict[int, TrackedVehicle] = {}
        self.next_track_id = 0
        self.iou_threshold = iou_threshold
        self.max_frames_lost = max_frames_lost

        # Generate random colors for tracks
        np.random.seed(42)
        self.colors = [(int(c[0]), int(c[1]), int(c[2]))
                       for c in np.random.randint(0, 255, (100, 3))]

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
            matched_tracks = set()
            matched_dets = set()

            while True:
                # Find best match
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
                track.positions.append((det.center[0], det.center[1]))
                track.timestamps.append(frame_time)
                track.frames_since_seen = 0

                # Calculate speed if we have enough history
                if len(track.positions) >= 2:
                    speed = self._calculate_speed(track)
                    track.speeds.append(speed)

                matched_tracks.add(i)
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
        """Calculate speed from recent positions (in km/h)."""
        if len(track.positions) < 2:
            return 0.0

        # Use positions from last few frames for smoothing
        # More frames = smoother but slower response
        n_frames = min(15, len(track.positions))

        positions = list(track.positions)[-n_frames:]
        timestamps = list(track.timestamps)[-n_frames:]

        # Calculate displacement
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        distance = np.sqrt(dx**2 + dy**2)  # meters (uncorrected)

        # Calculate average distance from camera origin
        # Used for distance-based scale correction
        avg_x = np.mean([p[0] for p in positions])
        avg_y = np.mean([p[1] for p in positions])
        dist_from_origin = np.sqrt(avg_x**2 + avg_y**2)

        # Apply distance-based scale correction
        # The calibration is non-linear - position affects scale
        BASE_SCALE = 8.0

        # Distance correction based on position
        if dist_from_origin > 0.1:
            # Base distance factor
            normalized_dist = (0.8 - dist_from_origin) / 0.7
            normalized_dist = max(0, min(1, normalized_dist))
            distance_factor = 1.0 + 3.2 * (normalized_dist ** 1.5)

            # Additional correction for left side of frame (negative X)
            # Cars on left side are further from calibration center
            if avg_x < 0:
                left_factor = 1.0 + abs(avg_x) * 2.0
                distance_factor *= left_factor
        else:
            distance_factor = 4.2

        SCALE_CORRECTION = BASE_SCALE * distance_factor
        distance = distance * SCALE_CORRECTION

        # Calculate time
        dt = timestamps[-1] - timestamps[0]  # seconds

        if dt <= 0:
            return 0.0

        # Speed in m/s, convert to km/h
        speed_ms = distance / dt
        speed_kmh = speed_ms * 3.6

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
                avg_speed = np.mean(list(track.speeds)[-10:])  # Average of last 10 readings
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

            # Draw trajectory
            if len(track.positions) > 1:
                self._draw_trajectory(output, track, bbox)

        return output

    def _draw_3d_box(self, image: np.ndarray, bbox: BBox3D, color: Tuple[int, int, int]):
        """Draw 3D wireframe bounding box."""
        corners = bbox.corners_image.astype(int)
        h, w = image.shape[:2]
        corners = np.clip(corners, [0, 0], [w-1, h-1])

        # Back edges (darker)
        back_color = tuple(c // 2 for c in color)
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

        # Front face highlight
        front_color = (0, 255, 0)
        cv.line(image, tuple(corners[0]), tuple(corners[1]), front_color, 3)
        cv.line(image, tuple(corners[4]), tuple(corners[5]), front_color, 3)
        cv.line(image, tuple(corners[0]), tuple(corners[4]), front_color, 3)
        cv.line(image, tuple(corners[1]), tuple(corners[5]), front_color, 3)

    def _draw_trajectory(self, image: np.ndarray, track: TrackedVehicle, bbox: BBox3D):
        """Draw vehicle trajectory on image."""
        # We'll draw small dots at recent positions
        # This requires projecting world coords back to image, which is complex
        # For now, we'll skip trajectory visualization
        pass

    def process_video(self, video_path: str, output_path: str):
        """Process entire video and save result."""
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        self.fps = cap.get(cv.CAP_PROP_FPS)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {self.fps:.1f}fps, {total_frames} frames")

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, self.fps, (width, height))

        self.frame_count = 0

        # Track speeds for the test vehicle (blue car)
        test_vehicle_speeds = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output, tracks = self.process_frame(frame)
            out.write(output)

            # Log speeds
            for track_id, track in tracks.items():
                if track.speeds and track.frames_since_seen == 0:
                    avg_speed = np.mean(list(track.speeds)[-5:])
                    # Check if this might be the blue test car (based on position)
                    if track.bbox_3d and -2 < track.bbox_3d.center[0] < 5:
                        test_vehicle_speeds.append((self.frame_count, track_id, avg_speed))

            if self.frame_count % 30 == 0:
                print(f"Frame {self.frame_count}/{total_frames} ({100*self.frame_count/total_frames:.1f}%)")

            self.frame_count += 1

        cap.release()
        out.release()

        print(f"\nDone! Output saved to: {output_path}")

        # Analyze test vehicle speeds
        if test_vehicle_speeds:
            print("\n" + "="*60)
            print("SPEED ANALYSIS")
            print("="*60)

            # Group by track ID
            from collections import defaultdict
            track_speeds = defaultdict(list)
            for frame, tid, speed in test_vehicle_speeds:
                track_speeds[tid].append(speed)

            for tid, speeds in track_speeds.items():
                if len(speeds) > 10:  # Only report tracks with significant data
                    avg = np.mean(speeds)
                    std = np.std(speeds)
                    print(f"Track {tid}: avg={avg:.1f} km/h, std={std:.1f}, samples={len(speeds)}")


def main():
    import sys
    calibration_dir = "/Users/navaneethmalingan/3D/Calibration"

    # Process all speed videos
    speeds = [20, 30, 40, 50, 60]

    # If command line argument provided, process only that speed
    if len(sys.argv) > 1:
        speeds = [int(sys.argv[1])]

    for speed in speeds:
        video_path = f"/Users/navaneethmalingan/3D/{speed}kmph.mp4"
        output_path = f"/Users/navaneethmalingan/3D/{speed}kmph_speed.mp4"

        print(f"\n{'='*60}")
        print(f"Processing {speed} km/h video")
        print(f"{'='*60}")

        estimator = SpeedEstimator(f"{calibration_dir}/calibration-lookup-table.npy")
        estimator.process_video(video_path, output_path)


if __name__ == "__main__":
    main()
