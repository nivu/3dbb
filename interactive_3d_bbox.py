"""
Interactive 3D Bounding Box Adjustment Tool

Allows manual adjustment of 3D bounding box corners to fit vehicles accurately.
Saves corner pixel positions for easy recalibration.

Controls:
- Click near a corner to select it (highlighted in cyan)
- Drag to move the selected corner
- Press 1-8 to select corners directly
- Press 's' to save current corner positions
- Press 'l' to load previously saved corners
- Press 'r' to reset to original positions
- Press 'p' to print projection parameters
- Press 'n' to go to next vehicle
- Press 'q' to quit
"""

import numpy as np
import cv2 as cv
import json
from pathlib import Path
from typing import List, Tuple, Optional

# Try to import the generator
try:
    from bbox_3d_generator import BBox3DGenerator, BBox2D, BBox3D
    GENERATOR_AVAILABLE = True
except ImportError:
    GENERATOR_AVAILABLE = False
    print("bbox_3d_generator not found, running in standalone mode")


class Interactive3DBBoxTool:
    """Interactive tool for adjusting 3D bounding box corners"""

    CORNER_NAMES = [
        "0:front-left-bottom", "1:front-right-bottom",
        "2:rear-right-bottom", "3:rear-left-bottom",
        "4:front-left-top", "5:front-right-top",
        "6:rear-right-top", "7:rear-left-top"
    ]

    def __init__(self, image_path: str, lookup_table_path: str = None):
        self.image_path = image_path
        self.original_image = cv.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.image = self.original_image.copy()
        self.img_h, self.img_w = self.image.shape[:2]

        # Initialize generator if available
        self.generator = None
        if GENERATOR_AVAILABLE and lookup_table_path:
            self.generator = BBox3DGenerator(lookup_table_path)

        # State
        self.bboxes_3d: List[BBox3D] = []
        self.current_bbox_idx = 0
        self.corners: np.ndarray = None  # 8x2 array of current corners
        self.corners_original: np.ndarray = None
        self.selected_corner = None
        self.dragging = False

        # Save file path
        self.save_path = Path(image_path).parent / "corner_points.json"

        # Window setup
        self.window_name = "3D BBox Tool - Drag corners, S=save, L=load, R=reset, N=next, Q=quit"

    def detect_vehicles(self, conf_threshold: float = 0.3):
        """Detect vehicles using YOLO"""
        if self.generator is None:
            print("Generator not available")
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
        print(f"  2D Box: ({bbox.bbox_2d.x1}, {bbox.bbox_2d.y1}) - ({bbox.bbox_2d.x2}, {bbox.bbox_2d.y2})")
        print(f"  World: ({bbox.center[0]:.2f}, {bbox.center[1]:.2f}) m")
        print(f"{'='*50}")

    def set_manual_bbox(self, x1: int, y1: int, x2: int, y2: int):
        """Set a manual bounding box without YOLO detection"""
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        self.corners = np.array([
            [x1 + bbox_w * 0.05, y2],
            [x2 - bbox_w * 0.05, y2],
            [x2 - bbox_w * 0.15, y2 - bbox_h * 0.4],
            [x1 + bbox_w * 0.15, y2 - bbox_h * 0.4],
            [x1 + bbox_w * 0.05, y2 - bbox_h * 0.7],
            [x2 - bbox_w * 0.05, y2 - bbox_h * 0.7],
            [x2 - bbox_w * 0.15, y2 - bbox_h * 0.9],
            [x1 + bbox_w * 0.15, y2 - bbox_h * 0.9],
        ], dtype=np.float32)

        self.corners_original = self.corners.copy()

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
        info = f"Vehicle {self.current_bbox_idx + 1}/{len(self.bboxes_3d)} | Selected: {self.selected_corner}"
        cv.putText(self.image, info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv.putText(self.image, "S=Save L=Load R=Reset N=Next Q=Quit", (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def get_projection_params(self) -> dict:
        """Calculate projection parameters from current corners"""
        if self.corners is None or not self.bboxes_3d:
            return {}

        bbox = self.bboxes_3d[self.current_bbox_idx]
        x1, y1, x2, y2 = bbox.bbox_2d.x1, bbox.bbox_2d.y1, bbox.bbox_2d.x2, bbox.bbox_2d.y2
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_cx = (x1 + x2) / 2

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
        data = {"vehicles": []}
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
            except:
                data = {"vehicles": []}

        # Create entry for this vehicle
        entry = {
            "vehicle_idx": self.current_bbox_idx,
            "bbox_2d": {
                "x1": int(bbox.bbox_2d.x1),
                "y1": int(bbox.bbox_2d.y1),
                "x2": int(bbox.bbox_2d.x2),
                "y2": int(bbox.bbox_2d.y2),
            } if bbox else None,
            "corners": [[int(c[0]), int(c[1])] for c in self.corners],
            "params": self.get_projection_params()
        }

        # Update or append
        found = False
        for i, v in enumerate(data["vehicles"]):
            if v.get("vehicle_idx") == self.current_bbox_idx:
                data["vehicles"][i] = entry
                found = True
                break
        if not found:
            data["vehicles"].append(entry)

        # Save
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n*** SAVED to {self.save_path} ***")
        print(f"Corner positions (pixels):")
        for i, c in enumerate(self.corners):
            print(f"  {i}: ({int(c[0])}, {int(c[1])})")

        params = self.get_projection_params()
        print(f"\nProjection parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")

    def load_corners(self):
        """Load saved corner positions"""
        if not self.save_path.exists():
            print("No saved corners found")
            return

        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)

            for v in data.get("vehicles", []):
                if v.get("vehicle_idx") == self.current_bbox_idx:
                    self.corners = np.array(v["corners"], dtype=np.float32)
                    print(f"Loaded corners for vehicle {self.current_bbox_idx + 1}")
                    self.redraw()
                    return

            print(f"No saved corners for vehicle {self.current_bbox_idx + 1}")
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

    def run(self):
        """Run the interactive tool"""
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.window_name, 1400, 800)
        cv.setMouseCallback(self.window_name, self.mouse_callback)

        # Initial detection
        if self.generator:
            self.detect_vehicles()
        else:
            self.set_manual_bbox(100, 100, 400, 300)

        self.redraw()

        print("\n" + "=" * 50)
        print("INTERACTIVE 3D BBOX TOOL")
        print("=" * 50)
        print("Drag corners to adjust the 3D box")
        print("Press S to SAVE corner positions")
        print("Press L to LOAD saved positions")
        print("Press R to RESET")
        print("Press N for NEXT vehicle")
        print("Press Q to QUIT")
        print("=" * 50)

        while True:
            cv.imshow(self.window_name, self.image)
            key = cv.waitKey(30) & 0xFF

            if key != 255:  # A key was pressed
                print(f"Key pressed: {key} ('{chr(key) if 32 <= key < 127 else '?'}')")

            if key == ord('q') or key == 27:
                break
            elif key == ord('s') or key == ord('S'):
                print("Saving corners...")
                self.save_corners()
            elif key == ord('l') or key == ord('L'):
                self.load_corners()
            elif key == ord('r') or key == ord('R'):
                self.reset_corners()
            elif key == ord('n') or key == ord('N'):
                self.next_vehicle()
            elif key == ord('p') or key == ord('P'):
                params = self.get_projection_params()
                print("\nProjection params:", params)
            elif ord('0') <= key <= ord('7'):
                self.selected_corner = key - ord('0')
                self.redraw()
            elif key == 32:  # Spacebar as alternative save
                print("Spacebar - Saving corners...")
                self.save_corners()

        cv.destroyAllWindows()


def main():
    image_path = "/Users/navaneethmalingan/3D/Calibration/frame_14s.png"
    lookup_path = "/Users/navaneethmalingan/3D/Calibration/calibration-lookup-table.npy"

    tool = Interactive3DBBoxTool(image_path, lookup_path)
    tool.run()


if __name__ == "__main__":
    main()
