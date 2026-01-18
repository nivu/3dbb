"""
Vehicle Distance & Overtaking Analyzer for Uni_west_1 Dataset

Calculates:
- Distance between vehicles (center-to-center)
- Overtaking events with minimum lateral distance
- Following distances (same lane)

Outputs:
- Annotated video with distance lines
- CSV file with all distance measurements
- Overtaking events log
"""

import cv2 as cv
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import csv
import os

from bbox_3d_calibrated import BBox3DGenerator

# Paths
DATASET_DIR = "/Users/navaneethmalingan/3D/Uni_west_1"
LOOKUP_TABLE = f"{DATASET_DIR}/calibration-lookup-table.npy"
OUTPUT_DIR = f"{DATASET_DIR}/output"

# Speed scale (from previous calibration)
SPEED_SCALE = 0.25


@dataclass
class OvertakingEvent:
    """Records an overtaking event between two vehicles."""
    frame_start: int
    frame_end: int
    vehicle_a_id: int
    vehicle_b_id: int
    min_lateral_distance: float  # meters
    min_total_distance: float    # meters
    overtaking_vehicle: int      # ID of vehicle that overtook


class StableTracker:
    """IoU-based vehicle tracker with position history."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0)
        ]

    def _iou(self, box1, box2):
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        return inter / (area1 + area2 - inter)

    def update(self, bboxes, frame_time):
        for tid in self.tracks:
            self.tracks[tid]['frames_lost'] += 1

        track_ids = list(self.tracks.keys())
        matched_dets = set()

        if track_ids and bboxes:
            iou_matrix = np.zeros((len(track_ids), len(bboxes)))
            for i, tid in enumerate(track_ids):
                for j, bbox in enumerate(bboxes):
                    if self.tracks[tid]['last_bbox'] is not None:
                        iou_matrix[i, j] = self._iou(
                            self.tracks[tid]['last_bbox'].bbox_2d, bbox.bbox_2d
                        )

            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = iou_matrix.max()
                if max_iou < 0.1:
                    break
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                tid = track_ids[i]
                bbox = bboxes[j]

                pos = (bbox.center[0], bbox.center[1])
                self.tracks[tid]['positions'].append(pos)
                self.tracks[tid]['timestamps'].append(frame_time)
                self.tracks[tid]['last_bbox'] = bbox
                self.tracks[tid]['frames_lost'] = 0
                matched_dets.add(j)
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        for j, bbox in enumerate(bboxes):
            if j not in matched_dets:
                pos = (bbox.center[0], bbox.center[1])
                self.tracks[self.next_id] = {
                    'positions': deque([pos], maxlen=60),
                    'timestamps': deque([frame_time], maxlen=60),
                    'last_bbox': bbox,
                    'frames_lost': 0,
                    'color': self.colors[self.next_id % len(self.colors)]
                }
                self.next_id += 1

        to_delete = [tid for tid, t in self.tracks.items() if t['frames_lost'] > 30]
        for tid in to_delete:
            del self.tracks[tid]

        return self.tracks

    def get_speed(self, track):
        if len(track['positions']) < 5:
            return 0.0
        positions = list(track['positions'])[-15:]
        timestamps = list(track['timestamps'])[-15:]
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        dist = np.sqrt(dx**2 + dy**2)
        dt = timestamps[-1] - timestamps[0]
        if dt <= 0:
            return 0.0
        speed = (dist / dt) * 3.6 * SPEED_SCALE
        return 0.0 if speed < 1.0 else min(speed, 120)


class DistanceAnalyzer:
    """Analyzes distances between vehicles and detects overtaking events."""

    # Thresholds
    OVERTAKING_LATERAL_THRESHOLD = 5.0   # meters - max lateral distance to consider overtaking
    OVERTAKING_LONG_THRESHOLD = 10.0     # meters - max longitudinal distance
    MIN_DISTANCE_DISPLAY = 15.0          # meters - only show distances below this

    def __init__(self):
        self.distance_log = []  # All distance measurements
        self.overtaking_events = []  # Detected overtaking events
        self.active_overtakes = {}  # Currently tracked potential overtakes

    def calculate_distances(self, tracks: Dict, frame_idx: int, frame_time: float) -> List[Dict]:
        """Calculate all pairwise distances between active vehicles."""
        distances = []

        # Get active tracks (visible this frame)
        active = {tid: t for tid, t in tracks.items()
                  if t['frames_lost'] == 0 and t['last_bbox'] is not None}

        track_ids = list(active.keys())

        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                tid_a = track_ids[i]
                tid_b = track_ids[j]

                pos_a = active[tid_a]['positions'][-1]
                pos_b = active[tid_b]['positions'][-1]

                # Calculate distances
                dx = pos_b[0] - pos_a[0]  # Longitudinal (along road)
                dy = pos_b[1] - pos_a[1]  # Lateral (across road)
                total_dist = np.sqrt(dx**2 + dy**2)

                dist_record = {
                    'frame': frame_idx,
                    'time': frame_time,
                    'vehicle_a': tid_a,
                    'vehicle_b': tid_b,
                    'pos_a_x': pos_a[0],
                    'pos_a_y': pos_a[1],
                    'pos_b_x': pos_b[0],
                    'pos_b_y': pos_b[1],
                    'distance_longitudinal': abs(dx),
                    'distance_lateral': abs(dy),
                    'distance_total': total_dist
                }

                distances.append(dist_record)
                self.distance_log.append(dist_record)

                # Check for overtaking
                self._check_overtaking(tid_a, tid_b, pos_a, pos_b, dx, dy,
                                       total_dist, frame_idx, frame_time)

        return distances

    def _check_overtaking(self, tid_a, tid_b, pos_a, pos_b, dx, dy,
                          total_dist, frame_idx, frame_time):
        """Detect and track overtaking events."""
        pair_key = tuple(sorted([tid_a, tid_b]))

        lateral_dist = abs(dy)
        long_dist = abs(dx)

        # Check if vehicles are close enough to be overtaking
        if (lateral_dist < self.OVERTAKING_LATERAL_THRESHOLD and
            long_dist < self.OVERTAKING_LONG_THRESHOLD):

            if pair_key not in self.active_overtakes:
                # Start tracking potential overtake
                self.active_overtakes[pair_key] = {
                    'frame_start': frame_idx,
                    'min_lateral': lateral_dist,
                    'min_total': total_dist,
                    'initial_dx': dx,  # Who was ahead initially
                    'tid_a': tid_a,
                    'tid_b': tid_b
                }
            else:
                # Update minimum distances
                ot = self.active_overtakes[pair_key]
                ot['min_lateral'] = min(ot['min_lateral'], lateral_dist)
                ot['min_total'] = min(ot['min_total'], total_dist)
                ot['last_dx'] = dx
                ot['frame_end'] = frame_idx
        else:
            # Vehicles separated - check if overtake completed
            if pair_key in self.active_overtakes:
                ot = self.active_overtakes[pair_key]

                # Check if positions swapped (overtake completed)
                if 'last_dx' in ot and np.sign(ot['initial_dx']) != np.sign(ot.get('last_dx', ot['initial_dx'])):
                    # Determine who overtook whom
                    if ot['initial_dx'] > 0:
                        overtaker = ot['tid_a'] if ot.get('last_dx', 0) < 0 else ot['tid_b']
                    else:
                        overtaker = ot['tid_b'] if ot.get('last_dx', 0) > 0 else ot['tid_a']

                    event = OvertakingEvent(
                        frame_start=ot['frame_start'],
                        frame_end=ot.get('frame_end', frame_idx),
                        vehicle_a_id=ot['tid_a'],
                        vehicle_b_id=ot['tid_b'],
                        min_lateral_distance=ot['min_lateral'],
                        min_total_distance=ot['min_total'],
                        overtaking_vehicle=overtaker
                    )
                    self.overtaking_events.append(event)
                    print(f"  OVERTAKING DETECTED: Vehicle {overtaker} overtook, "
                          f"min lateral distance: {ot['min_lateral']:.2f}m")

                del self.active_overtakes[pair_key]

    def save_results(self, output_dir: str, video_name: str):
        """Save distance log and overtaking events to CSV files."""
        # Save all distances
        dist_path = os.path.join(output_dir, f"{video_name}_distances.csv")
        with open(dist_path, 'w', newline='') as f:
            if self.distance_log:
                writer = csv.DictWriter(f, fieldnames=self.distance_log[0].keys())
                writer.writeheader()
                writer.writerows(self.distance_log)
        print(f"  Saved {len(self.distance_log)} distance records to {dist_path}")

        # Save overtaking events
        ot_path = os.path.join(output_dir, f"{video_name}_overtaking.csv")
        with open(ot_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_start', 'frame_end', 'vehicle_a', 'vehicle_b',
                            'min_lateral_dist_m', 'min_total_dist_m', 'overtaking_vehicle'])
            for e in self.overtaking_events:
                writer.writerow([e.frame_start, e.frame_end, e.vehicle_a_id,
                                e.vehicle_b_id, f"{e.min_lateral_distance:.2f}",
                                f"{e.min_total_distance:.2f}", e.overtaking_vehicle])
        print(f"  Saved {len(self.overtaking_events)} overtaking events to {ot_path}")


def draw_distances(frame, tracks, distances, analyzer):
    """Draw distance lines between nearby vehicles."""
    for dist in distances:
        if dist['distance_total'] > analyzer.MIN_DISTANCE_DISPLAY:
            continue

        tid_a = dist['vehicle_a']
        tid_b = dist['vehicle_b']

        if tid_a not in tracks or tid_b not in tracks:
            continue

        bbox_a = tracks[tid_a]['last_bbox']
        bbox_b = tracks[tid_b]['last_bbox']

        if bbox_a is None or bbox_b is None:
            continue

        # Get center points in image coordinates
        center_a = bbox_a.corners_image[:4].mean(axis=0).astype(int)
        center_b = bbox_b.corners_image[:4].mean(axis=0).astype(int)

        # Draw line
        color = (0, 165, 255)  # Orange
        cv.line(frame, tuple(center_a), tuple(center_b), color, 2)

        # Draw distance label at midpoint
        mid_x = (center_a[0] + center_b[0]) // 2
        mid_y = (center_a[1] + center_b[1]) // 2

        label = f"{dist['distance_total']:.1f}m"
        cv.rectangle(frame, (mid_x - 30, mid_y - 12), (mid_x + 30, mid_y + 5), (0, 0, 0), -1)
        cv.putText(frame, label, (mid_x - 25, mid_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    return frame


def draw_3d_box(frame, bbox, color, speed, tid):
    """Draw 3D bounding box with speed label."""
    corners = bbox.corners_image.astype(int)

    # Front face (green)
    cv.line(frame, tuple(corners[0]), tuple(corners[1]), (0, 255, 0), 2)
    cv.line(frame, tuple(corners[4]), tuple(corners[5]), (0, 255, 0), 2)
    cv.line(frame, tuple(corners[0]), tuple(corners[4]), (0, 255, 0), 2)
    cv.line(frame, tuple(corners[1]), tuple(corners[5]), (0, 255, 0), 2)

    # Sides
    cv.line(frame, tuple(corners[0]), tuple(corners[3]), color, 2)
    cv.line(frame, tuple(corners[1]), tuple(corners[2]), color, 2)
    cv.line(frame, tuple(corners[4]), tuple(corners[7]), color, 2)
    cv.line(frame, tuple(corners[5]), tuple(corners[6]), color, 2)

    # Back
    cv.line(frame, tuple(corners[2]), tuple(corners[3]), (100, 100, 100), 2)
    cv.line(frame, tuple(corners[6]), tuple(corners[7]), (100, 100, 100), 2)
    cv.line(frame, tuple(corners[2]), tuple(corners[6]), (100, 100, 100), 2)
    cv.line(frame, tuple(corners[3]), tuple(corners[7]), (100, 100, 100), 2)

    # Top
    cv.line(frame, tuple(corners[4]), tuple(corners[5]), color, 2)
    cv.line(frame, tuple(corners[5]), tuple(corners[6]), color, 2)
    cv.line(frame, tuple(corners[6]), tuple(corners[7]), color, 2)
    cv.line(frame, tuple(corners[7]), tuple(corners[4]), color, 2)

    # Label
    label = f"ID:{tid} {speed:.1f}km/h"
    top_y = int(corners[4:8, 1].min()) - 10
    left_x = int(corners[:, 0].min())
    cv.rectangle(frame, (left_x, top_y - 25), (left_x + 160, top_y + 5), (0, 0, 0), -1)
    cv.putText(frame, label, (left_x + 5, top_y),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def process_video(video_path: str, output_path: str, generator, max_frames: int = None):
    """Process video with distance analysis."""
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total = min(total, max_frames)

    video_name = os.path.basename(video_path).replace('.MP4', '').replace('.mp4', '')

    print(f"\nProcessing: {video_path}")
    print(f"  Resolution: {width}x{height} @ {fps}fps")
    print(f"  Frames: {total}")

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = StableTracker()
    analyzer = DistanceAnalyzer()

    frame_idx = 0

    while frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vehicles
        bboxes = generator.process_image(frame, conf_threshold=0.3)

        # Update tracker
        frame_time = frame_idx / fps
        tracks = tracker.update(bboxes, frame_time)

        # Calculate distances
        distances = analyzer.calculate_distances(tracks, frame_idx, frame_time)

        # Draw 3D boxes and speeds
        for tid, track in tracks.items():
            if track['frames_lost'] > 0 or track['last_bbox'] is None:
                continue
            speed = tracker.get_speed(track)
            draw_3d_box(frame, track['last_bbox'], track['color'], speed, tid)

        # Draw distance lines
        frame = draw_distances(frame, tracks, distances, analyzer)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"    {frame_idx}/{total} ({100*frame_idx/total:.0f}%)")

    cap.release()
    out.release()

    print(f"  Video saved to: {output_path}")

    # Save CSV results
    analyzer.save_results(OUTPUT_DIR, video_name)

    return analyzer


def main():
    import sys

    print("=" * 60)
    print("DISTANCE & OVERTAKING ANALYZER - Uni_west_1")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize generator
    generator = BBox3DGenerator(LOOKUP_TABLE)

    # Videos to process
    videos = [
        ("GOPR0574.MP4", None),      # Full video
        ("GOPR0575.MP4", None),      # Full video
        ("GOPR0581.MP4", 5000),      # First 5000 frames (large video)
    ]

    # Check command line args
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        if idx < len(videos):
            videos = [videos[idx]]

    for video_name, max_frames in videos:
        video_path = os.path.join(DATASET_DIR, video_name)
        output_path = os.path.join(OUTPUT_DIR, video_name.replace('.MP4', '_distance.mp4'))

        if os.path.exists(video_path):
            analyzer = process_video(video_path, output_path, generator, max_frames)

            # Print summary
            print(f"\n  Summary for {video_name}:")
            print(f"    Total distance measurements: {len(analyzer.distance_log)}")
            print(f"    Overtaking events detected: {len(analyzer.overtaking_events)}")

            if analyzer.overtaking_events:
                print(f"\n    Overtaking Events:")
                for e in analyzer.overtaking_events:
                    print(f"      - Frames {e.frame_start}-{e.frame_end}: "
                          f"Vehicle {e.overtaking_vehicle} overtook, "
                          f"min distance: {e.min_lateral_distance:.2f}m (lateral), "
                          f"{e.min_total_distance:.2f}m (total)")
        else:
            print(f"  WARNING: {video_path} not found")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
