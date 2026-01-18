"""
Interactive 3D Bounding Box Adjustment Tool for Uni_west_1 Dataset

Allows manual adjustment of 3D bounding box corners to fit vehicles accurately.
Extracts a frame from the video for adjustment.

Controls:
- Click near a corner to select it (highlighted in cyan)
- Drag to move the selected corner
- Press 1-8 to select corners directly
- Press 's' to SAVE current adjustment (only saved adjustments affect calibration)
- Press 'x' to SKIP/ignore this vehicle (won't be saved)
- Press 'l' to load previously saved corners
- Press 'r' to reset to original positions
- Press 'n' to go to next vehicle
- Press 'f' to go forward 30 frames
- Press 'b' to go back 30 frames
- Press 'g' to go to a specific frame number (type in terminal)
- Press 'v' to go to next video
- Press 'q' to quit
"""

import sys
sys.path.insert(0, '/Users/navaneethmalingan/3D')

import numpy as np
import cv2 as cv
import json
from pathlib import Path
from typing import List, Optional

from bbox_3d_generator import BBox3DGenerator, BBox2D, BBox3D

# Dataset paths
DATASET_DIR = "/Users/navaneethmalingan/3D/Uni_west_1"
LOOKUP_TABLE = f"{DATASET_DIR}/calibration-lookup-table.npy"

# Videos
VIDEOS = [
    "GOPR0574.MP4",
    "GOPR0575.MP4",
    "GOPR0581.MP4"
]


class Interactive3DBBoxTool:
    """Interactive tool for adjusting 3D bounding box corners"""

    CORNER_NAMES = [
        "0:front-left-bottom", "1:front-right-bottom",
        "2:rear-right-bottom", "3:rear-left-bottom",
        "4:front-left-top", "5:front-right-top",
        "6:rear-right-top", "7:rear-left-top"
    ]

    def __init__(self, lookup_table_path: str):
        self.lookup_table_path = lookup_table_path
        self.generator = BBox3DGenerator(lookup_table_path)

        # Video state
        self.current_video_idx = 0
        self.current_frame_num = 100
        self.frame_step = 30  # Smaller steps for better navigation

        # Image state
        self.original_image = None
        self.image = None
        self.img_h = None
        self.img_w = None

        # Detection state
        self.bboxes_3d: List[BBox3D] = []
        self.current_bbox_idx = 0
        self.corners: np.ndarray = None
        self.corners_original: np.ndarray = None
        self.selected_corner = None
        self.dragging = False

        # Save path
        self.save_path = Path(DATASET_DIR) / "corner_adjustments.json"

        # Window
        self.window_name = "3D BBox Tool - S=save, L=load, R=reset, N=next vehicle, F=next frame, Q=quit"

    def load_frame_from_video(self, video_idx: int, frame_num: int) -> bool:
        """Load a specific frame from a video."""
        if video_idx >= len(VIDEOS):
            return False

        video_path = f"{DATASET_DIR}/{VIDEOS[video_idx]}"
        cap = cv.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"ERROR: Could not open {video_path}")
            return False

        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if frame_num >= total_frames:
            frame_num = total_frames - 1

        cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"ERROR: Could not read frame {frame_num}")
            return False

        self.original_image = frame
        self.image = frame.copy()
        self.img_h, self.img_w = frame.shape[:2]
        self.current_video_idx = video_idx
        self.current_frame_num = frame_num

        print(f"\nLoaded: {VIDEOS[video_idx]} frame {frame_num}")
        return True

    def detect_vehicles(self, conf_threshold: float = 0.3):
        """Detect vehicles using YOLO"""
        if self.original_image is None:
            print("No image loaded")
            return

        self.bboxes_3d = self.generator.process_image(self.original_image, conf_threshold)
        print(f"Detected {len(self.bboxes_3d)} vehicles")

        if self.bboxes_3d:
            self.current_bbox_idx = 0
            self.load_current_bbox()

    def load_current_bbox(self):
        """Load corners from current bounding box"""
        if not self.bboxes_3d:
            return

        bbox = self.bboxes_3d[self.current_bbox_idx]
        self.corners = bbox.corners_image.copy()
        self.corners_original = bbox.corners_image.copy()

        print(f"\n{'='*50}")
        print(f"Vehicle {self.current_bbox_idx + 1}/{len(self.bboxes_3d)}")
        print(f"  Class: {bbox.bbox_2d.class_name}")
        print(f"  2D Box: ({bbox.bbox_2d.x1}, {bbox.bbox_2d.y1}) - ({bbox.bbox_2d.x2}, {bbox.bbox_2d.y2})")
        print(f"  World Pos: ({bbox.center[0]:.2f}, {bbox.center[1]:.2f}) m")
        print(f"{'='*50}")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv.EVENT_LBUTTONDOWN:
            if self.corners is not None:
                distances = np.linalg.norm(self.corners - np.array([x, y]), axis=1)
                nearest = np.argmin(distances)
                if distances[nearest] < 25:
                    self.selected_corner = nearest
                    self.dragging = True

        elif event == cv.EVENT_MOUSEMOVE:
            if self.dragging and self.selected_corner is not None:
                self.corners[self.selected_corner] = [x, y]
                self.redraw()

        elif event == cv.EVENT_LBUTTONUP:
            self.dragging = False

    def redraw(self):
        """Redraw the image with current corners"""
        self.image = self.original_image.copy()

        if self.corners is None:
            return

        corners = self.corners.astype(int)

        # Draw filled faces with transparency
        overlay = self.image.copy()

        # Front face (green)
        front_pts = np.array([corners[0], corners[1], corners[5], corners[4]])
        cv.fillPoly(overlay, [front_pts], (0, 200, 0))

        # Right side (blue)
        right_pts = np.array([corners[1], corners[2], corners[6], corners[5]])
        cv.fillPoly(overlay, [right_pts], (200, 100, 100))

        # Left side (red)
        left_pts = np.array([corners[0], corners[3], corners[7], corners[4]])
        cv.fillPoly(overlay, [left_pts], (100, 100, 200))

        cv.addWeighted(overlay, 0.25, self.image, 0.75, 0, self.image)

        # Draw all edges in red
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
        ]
        for i, j in edges:
            cv.line(self.image, tuple(corners[i]), tuple(corners[j]), (0, 0, 255), 2)

        # Front face in green (thicker)
        for i, j in [(0, 1), (4, 5), (0, 4), (1, 5)]:
            cv.line(self.image, tuple(corners[i]), tuple(corners[j]), (0, 255, 0), 3)

        # Draw corner points with numbers
        for i, corner in enumerate(corners):
            color = (255, 255, 0) if i == self.selected_corner else (0, 255, 255)
            radius = 10 if i == self.selected_corner else 6
            cv.circle(self.image, tuple(corner), radius, color, -1)
            cv.circle(self.image, tuple(corner), radius, (0, 0, 0), 2)
            cv.putText(self.image, str(i), (corner[0] + 12, corner[1] + 5),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Info text
        video_name = VIDEOS[self.current_video_idx] if self.current_video_idx < len(VIDEOS) else "?"
        info = f"{video_name} | Frame {self.current_frame_num} | Vehicle {self.current_bbox_idx + 1}/{len(self.bboxes_3d)}"
        cv.putText(self.image, info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv.putText(self.image, "S=Save L=Load R=Reset N=NextVehicle F=NextFrame V=NextVideo Q=Quit",
                  (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def get_projection_params(self) -> dict:
        """Calculate projection parameters from current corners"""
        if self.corners is None or not self.bboxes_3d:
            return {}

        bbox = self.bboxes_3d[self.current_bbox_idx]
        x1, y1, x2, y2 = bbox.bbox_2d.x1, bbox.bbox_2d.y1, bbox.bbox_2d.x2, bbox.bbox_2d.y2
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        c = self.corners

        # Front edge
        front_y = (c[0, 1] + c[1, 1]) / 2
        front_cx = (c[0, 0] + c[1, 0]) / 2
        front_width = c[1, 0] - c[0, 0]

        # Rear edge
        rear_y = (c[2, 1] + c[3, 1]) / 2
        rear_cx = (c[2, 0] + c[3, 0]) / 2
        rear_width = c[2, 0] - c[3, 0]

        # Top front
        top_front_y = (c[4, 1] + c[5, 1]) / 2

        params = {
            "depth_ratio": float(round((front_y - rear_y) / bbox_h, 3)),
            "side_offset_ratio": float(round((rear_cx - front_cx) / bbox_w, 3)),
            "height_ratio": float(round((front_y - top_front_y) / bbox_h, 3)),
            "shrink_ratio": float(round(rear_width / front_width, 3)) if front_width > 0 else 1.0,
            "front_inset": float(round((c[0, 0] - x1) / bbox_w, 3)),
        }
        return params

    def save_corners(self):
        """Save current corner positions to JSON"""
        if self.corners is None:
            print("No corners to save")
            return

        bbox = self.bboxes_3d[self.current_bbox_idx] if self.bboxes_3d else None

        # Load existing data
        data = {"adjustments": []}
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
            except:
                data = {"adjustments": []}

        # Create entry
        entry = {
            "video": VIDEOS[self.current_video_idx],
            "frame": self.current_frame_num,
            "vehicle_idx": self.current_bbox_idx,
            "bbox_2d": {
                "x1": int(bbox.bbox_2d.x1),
                "y1": int(bbox.bbox_2d.y1),
                "x2": int(bbox.bbox_2d.x2),
                "y2": int(bbox.bbox_2d.y2),
            } if bbox else None,
            "world_pos": {
                "x": float(bbox.center[0]),
                "y": float(bbox.center[1])
            } if bbox else None,
            "corners": [[int(c[0]), int(c[1])] for c in self.corners],
            "params": self.get_projection_params()
        }

        # Update or append
        found = False
        for i, adj in enumerate(data["adjustments"]):
            if (adj.get("video") == entry["video"] and
                adj.get("frame") == entry["frame"] and
                adj.get("vehicle_idx") == entry["vehicle_idx"]):
                data["adjustments"][i] = entry
                found = True
                break
        if not found:
            data["adjustments"].append(entry)

        # Save
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n*** SAVED to {self.save_path} ***")
        print(f"Projection parameters:")
        for k, v in entry["params"].items():
            print(f"  {k}: {v}")

    def load_corners(self):
        """Load saved corner positions"""
        if not self.save_path.exists():
            print("No saved corners found")
            return

        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)

            video_name = VIDEOS[self.current_video_idx]
            for adj in data.get("adjustments", []):
                if (adj.get("video") == video_name and
                    adj.get("frame") == self.current_frame_num and
                    adj.get("vehicle_idx") == self.current_bbox_idx):
                    self.corners = np.array(adj["corners"], dtype=np.float32)
                    print(f"Loaded corners for {video_name} frame {self.current_frame_num} vehicle {self.current_bbox_idx + 1}")
                    self.redraw()
                    return

            print(f"No saved corners for current vehicle")
        except Exception as e:
            print(f"Error loading corners: {e}")

    def reset_corners(self):
        """Reset to original positions"""
        if self.corners_original is not None:
            self.corners = self.corners_original.copy()
            self.redraw()
            print("Reset to original")

    def next_vehicle(self):
        """Go to next vehicle"""
        if not self.bboxes_3d:
            return
        self.current_bbox_idx = (self.current_bbox_idx + 1) % len(self.bboxes_3d)
        self.load_current_bbox()
        self.redraw()

    def next_frame(self):
        """Go to next frame in current video"""
        new_frame = self.current_frame_num + self.frame_step
        if self.load_frame_from_video(self.current_video_idx, new_frame):
            self.detect_vehicles()
            self.redraw()

    def prev_frame(self):
        """Go to previous frame in current video"""
        new_frame = max(0, self.current_frame_num - self.frame_step)
        if self.load_frame_from_video(self.current_video_idx, new_frame):
            self.detect_vehicles()
            self.redraw()

    def goto_frame(self):
        """Go to a specific frame number"""
        try:
            frame_num = int(input("Enter frame number: "))
            if self.load_frame_from_video(self.current_video_idx, frame_num):
                self.detect_vehicles()
                self.redraw()
        except ValueError:
            print("Invalid frame number")

    def next_video(self):
        """Go to next video"""
        new_video_idx = (self.current_video_idx + 1) % len(VIDEOS)
        if self.load_frame_from_video(new_video_idx, 100):
            self.detect_vehicles()
            self.redraw()

    def run(self):
        """Run the interactive tool"""
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.window_name, 1400, 800)
        cv.setMouseCallback(self.window_name, self.mouse_callback)

        # Load first frame
        if not self.load_frame_from_video(0, 100):
            print("Failed to load initial frame")
            return

        self.detect_vehicles()
        self.redraw()

        print("\n" + "=" * 60)
        print("INTERACTIVE 3D BBOX TOOL - Uni_west_1 Dataset")
        print("=" * 60)
        print("Drag corners to adjust the 3D box")
        print("Press S to SAVE this adjustment")
        print("Press X to SKIP/ignore this vehicle")
        print("Press L to LOAD saved positions")
        print("Press R to RESET to original")
        print("Press N for NEXT vehicle")
        print("Press F for FORWARD 30 frames")
        print("Press B for BACK 30 frames")
        print("Press G to GO TO specific frame")
        print("Press V for NEXT video")
        print("Press Q to QUIT")
        print("=" * 60)

        while True:
            cv.imshow(self.window_name, self.image)
            key = cv.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('s') or key == ord('S'):
                self.save_corners()
            elif key == ord('l') or key == ord('L'):
                self.load_corners()
            elif key == ord('r') or key == ord('R'):
                self.reset_corners()
            elif key == ord('x') or key == ord('X'):
                print("SKIPPED - Vehicle ignored, moving to next")
                self.next_vehicle()
            elif key == ord('n') or key == ord('N'):
                self.next_vehicle()
            elif key == ord('f') or key == ord('F'):
                self.next_frame()
            elif key == ord('b') or key == ord('B'):
                self.prev_frame()
            elif key == ord('g') or key == ord('G'):
                self.goto_frame()
            elif key == ord('v') or key == ord('V'):
                self.next_video()
            elif ord('0') <= key <= ord('7'):
                self.selected_corner = key - ord('0')
                self.redraw()

        cv.destroyAllWindows()


def main():
    print("=" * 60)
    print("Interactive 3D BBox Tool - Uni_west_1 Dataset")
    print("=" * 60)

    tool = Interactive3DBBoxTool(LOOKUP_TABLE)
    tool.run()


if __name__ == "__main__":
    main()
