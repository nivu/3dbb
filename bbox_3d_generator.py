"""
3D Bounding Box Generator
Uses calibration lookup table to convert 2D vehicle detections to 3D world coordinates
and project 3D bounding boxes back onto the image.

Approach:
1. Detect vehicles using YOLO (2D bounding boxes)
2. Estimate tire contact points (bottom of bounding box on the street)
3. Convert tire contact points to world coordinates using calibration
4. Construct 3D bounding box constrained by 2D detection
"""

import numpy as np
import cv2 as cv
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


@dataclass
class BBox2D:
    """2D bounding box in image coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


@dataclass
class BBox3D:
    """3D bounding box"""
    corners_image: np.ndarray  # 8x2 array - corners in image coordinates
    corners_world: np.ndarray  # 8x3 array - corners in world coordinates
    center: np.ndarray         # [x, y, z] world coordinates
    length: float
    width: float
    height: float
    yaw: float
    bbox_2d: BBox2D
    confidence: float


class BBox3DGenerator:
    """Generates 3D bounding boxes from 2D detections using calibration data"""

    VEHICLE_DIMENSIONS = {
        'car': {'length': 4.5, 'width': 1.8, 'height': 1.5},
        'truck': {'length': 6.0, 'width': 2.5, 'height': 2.5},
        'bus': {'length': 12.0, 'width': 2.5, 'height': 3.0},
        'motorcycle': {'length': 2.2, 'width': 0.8, 'height': 1.2},
        'bicycle': {'length': 1.8, 'width': 0.6, 'height': 1.1},
        'person': {'length': 0.5, 'width': 0.5, 'height': 1.7},
    }

    VEHICLE_CLASSES = {
        2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 1: 'bicycle', 0: 'person',
    }

    def __init__(self, lookup_table_path: str, yolo_model: str = 'yolov8n.pt'):
        self.lookup_table = np.load(lookup_table_path)
        self.lt_width, self.lt_height, _ = self.lookup_table.shape
        print(f"Loaded lookup table: {self.lookup_table.shape}")

        # Store current image dimensions for scaling
        self.img_width = None
        self.img_height = None

        if YOLO_AVAILABLE:
            self.model = YOLO(yolo_model)
            print(f"Loaded YOLO model: {yolo_model}")
        else:
            self.model = None

    def set_image_size(self, width: int, height: int):
        """Set current image dimensions for coordinate scaling."""
        self.img_width = width
        self.img_height = height

    def pixel_to_world(self, x: int, y: int) -> Optional[np.ndarray]:
        """Convert pixel coordinates to world coordinates, with scaling if needed."""
        # Scale coordinates if image size differs from lookup table
        if self.img_width and self.img_height:
            scale_x = self.lt_width / self.img_width
            scale_y = self.lt_height / self.img_height
            x = int(x * scale_x)
            y = int(y * scale_y)

        if 0 <= x < self.lt_width and 0 <= y < self.lt_height:
            return self.lookup_table[x, y].copy()
        return None

    def detect_vehicles(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[BBox2D]:
        """Detect vehicles using YOLO."""
        if self.model is None:
            return []

        results = self.model(image, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if confidence < conf_threshold or class_id not in self.VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append(BBox2D(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                confidence=confidence, class_id=class_id,
                class_name=self.VEHICLE_CLASSES[class_id]
            ))

        return detections

    def estimate_3d_bbox(self, bbox_2d: BBox2D) -> Optional[BBox3D]:
        """
        Estimate 3D bounding box from 2D detection.

        The 3D box is constructed to fit within the 2D bounding box,
        with ground contact points mapped to world coordinates.
        """
        # 2D bbox dimensions
        x1, y1, x2, y2 = bbox_2d.x1, bbox_2d.y1, bbox_2d.x2, bbox_2d.y2
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Contact points (bottom of bbox, inset for tires)
        inset = 0.15
        left_x = int(x1 + bbox_w * inset)
        right_x = int(x2 - bbox_w * inset)
        bottom_y = y2

        # Get world coordinates for contact points
        left_world = self.pixel_to_world(left_x, bottom_y)
        right_world = self.pixel_to_world(right_x, bottom_y)

        if left_world is None or right_world is None:
            return None

        # Validate world coordinates
        if np.abs(left_world[0]) > 50 or np.abs(right_world[0]) > 50:
            return None

        # Get vehicle dimensions
        dims = self.VEHICLE_DIMENSIONS.get(bbox_2d.class_name, self.VEHICLE_DIMENSIONS['car'])

        # Calculate orientation from contact points
        # The direction from left to right contact point is perpendicular to the car's heading
        # So the car's heading (front direction) is 90° rotated from this
        lateral_direction = right_world[:2] - left_world[:2]
        # Car heading is perpendicular to the lateral direction
        # Rotate by -90° (turn left) to get the forward direction
        yaw = np.arctan2(lateral_direction[1], lateral_direction[0]) - np.pi/2

        # Center ground position
        center_ground = (left_world + right_world) / 2

        # Observed width
        observed_width = np.linalg.norm(right_world[:2] - left_world[:2])
        width = observed_width if 0.5 < observed_width < 3.0 else dims['width']
        length = dims['length']
        height = dims['height']

        # Build 3D corners in world coordinates
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        half_l, half_w = length / 2, width / 2

        local_corners = np.array([
            [ half_l, -half_w, 0],       # 0: front-left-bottom
            [ half_l,  half_w, 0],       # 1: front-right-bottom
            [-half_l,  half_w, 0],       # 2: rear-right-bottom
            [-half_l, -half_w, 0],       # 3: rear-left-bottom
            [ half_l, -half_w, height],  # 4: front-left-top
            [ half_l,  half_w, height],  # 5: front-right-top
            [-half_l,  half_w, height],  # 6: rear-right-top
            [-half_l, -half_w, height],  # 7: rear-left-top
        ])

        R = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])
        corners_world = (local_corners @ R.T) + center_ground

        # Project 3D box to 2D image - constrained by 2D bbox
        corners_image = self._project_constrained(corners_world, bbox_2d, yaw)

        center_3d = np.array([center_ground[0], center_ground[1], height / 2])

        return BBox3D(
            corners_image=corners_image,
            corners_world=corners_world,
            center=center_3d,
            length=length, width=width, height=height,
            yaw=yaw, bbox_2d=bbox_2d, confidence=bbox_2d.confidence
        )

    def _project_constrained(self, corners_world: np.ndarray, bbox_2d: BBox2D, yaw: float) -> np.ndarray:
        """
        Project 3D corners to 2D image coordinates using calibrated parameters.

        Parameters were calibrated from manual adjustments on reference vehicles.
        """
        x1, y1, x2, y2 = bbox_2d.x1, bbox_2d.y1, bbox_2d.x2, bbox_2d.y2
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        corners_image = np.zeros((8, 2), dtype=np.float32)

        # Calibrated parameters (averaged from manual adjustments on 4 vehicles)
        depth_ratio = 0.2433       # How far rear extends up (as ratio of bbox height)
        side_offset_ratio = 0.4670 # How far rear shifts right (positive = right)
        height_ratio = 0.85        # Box height vs bbox height
        shrink_ratio = 0.7405      # Rear width vs front width (perspective shrinking)

        # Front face positioning (calibrated from manual adjustments)
        front_left_offset = 0.0171   # Front left starts near bbox left edge
        front_right_offset = 0.45    # Front right is ~45% inset from bbox right edge

        # Calculate box dimensions
        box_height = bbox_h * height_ratio
        depth_up = bbox_h * depth_ratio

        # Front face (bottom of bbox, closest to camera)
        front_y = y2
        front_left_x = x1 + bbox_w * front_left_offset
        front_right_x = x2 - bbox_w * front_right_offset
        front_width = front_right_x - front_left_x
        front_cx = (front_left_x + front_right_x) / 2

        # Rear face (shifted up and to the right)
        rear_y = front_y - depth_up
        rear_width = front_width * shrink_ratio
        # Side offset is relative to front center
        side_offset = bbox_w * side_offset_ratio
        rear_cx = front_cx + side_offset
        rear_left_x = rear_cx - rear_width / 2
        rear_right_x = rear_cx + rear_width / 2

        # Bottom corners (ground plane)
        corners_image[0] = [front_left_x, front_y]      # front-left-bottom
        corners_image[1] = [front_right_x, front_y]     # front-right-bottom
        corners_image[2] = [rear_right_x, rear_y]       # rear-right-bottom
        corners_image[3] = [rear_left_x, rear_y]        # rear-left-bottom

        # Top corners - rear height slightly smaller due to perspective
        rear_height = box_height * 0.88  # Rear height slightly less than front
        corners_image[4] = [front_left_x, front_y - box_height]      # front-left-top
        corners_image[5] = [front_right_x, front_y - box_height]     # front-right-top
        corners_image[6] = [rear_right_x, rear_y - rear_height]      # rear-right-top
        corners_image[7] = [rear_left_x, rear_y - rear_height]       # rear-left-top

        return corners_image

    def process_image(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[BBox3D]:
        """Process image and generate 3D bounding boxes."""
        # Set image size for coordinate scaling
        h, w = image.shape[:2]
        self.set_image_size(w, h)

        detections_2d = self.detect_vehicles(image, conf_threshold)
        print(f"Detected {len(detections_2d)} vehicles")

        bboxes_3d = []
        for det in detections_2d:
            bbox_3d = self.estimate_3d_bbox(det)
            if bbox_3d is not None:
                bboxes_3d.append(bbox_3d)

        print(f"Generated {len(bboxes_3d)} 3D bounding boxes")
        return bboxes_3d

    def process_manual_bbox(self, x1: int, y1: int, x2: int, y2: int,
                           class_name: str = 'car') -> Optional[BBox3D]:
        """Process a manual bounding box."""
        bbox_2d = BBox2D(x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=1.0, class_id=2, class_name=class_name)
        return self.estimate_3d_bbox(bbox_2d)

    def draw_3d_box(self, image: np.ndarray, bbox: BBox3D,
                    color: Tuple[int, int, int] = (0, 0, 255),
                    thickness: int = 2) -> np.ndarray:
        """Draw 3D wireframe bounding box with proper face visibility."""
        corners = bbox.corners_image.astype(int)
        h, w = image.shape[:2]
        corners = np.clip(corners, [0, 0], [w-1, h-1])

        # Corner indices:
        # Bottom: 0=front-left, 1=front-right, 2=rear-right, 3=rear-left
        # Top:    4=front-left, 5=front-right, 6=rear-right, 7=rear-left

        # Determine which faces are visible based on corner positions
        # Front face visible if front corners are lower (closer to camera) than rear
        front_visible = corners[0, 1] > corners[3, 1] or corners[1, 1] > corners[2, 1]
        # Right face visible if right corners are more to the left than left corners (in perspective)
        right_visible = corners[1, 0] > corners[0, 0]
        # Left face visible
        left_visible = corners[0, 0] < corners[1, 0]

        # Draw back edges first (will be partially occluded)
        back_color = (100, 100, 100)  # Darker for back
        # Back face edges
        cv.line(image, tuple(corners[2]), tuple(corners[3]), back_color, thickness)
        cv.line(image, tuple(corners[6]), tuple(corners[7]), back_color, thickness)
        cv.line(image, tuple(corners[2]), tuple(corners[6]), back_color, thickness)
        cv.line(image, tuple(corners[3]), tuple(corners[7]), back_color, thickness)

        # Draw side edges (red)
        cv.line(image, tuple(corners[0]), tuple(corners[3]), color, thickness)
        cv.line(image, tuple(corners[1]), tuple(corners[2]), color, thickness)
        cv.line(image, tuple(corners[4]), tuple(corners[7]), color, thickness)
        cv.line(image, tuple(corners[5]), tuple(corners[6]), color, thickness)

        # Draw bottom face
        cv.line(image, tuple(corners[0]), tuple(corners[1]), color, thickness)
        cv.line(image, tuple(corners[1]), tuple(corners[2]), color, thickness)
        cv.line(image, tuple(corners[2]), tuple(corners[3]), color, thickness)
        cv.line(image, tuple(corners[3]), tuple(corners[0]), color, thickness)

        # Draw top face
        cv.line(image, tuple(corners[4]), tuple(corners[5]), color, thickness)
        cv.line(image, tuple(corners[5]), tuple(corners[6]), color, thickness)
        cv.line(image, tuple(corners[6]), tuple(corners[7]), color, thickness)
        cv.line(image, tuple(corners[7]), tuple(corners[4]), color, thickness)

        # Draw front face edges last (most visible) - in green
        front_color = (0, 255, 0)
        cv.line(image, tuple(corners[0]), tuple(corners[1]), front_color, thickness + 1)
        cv.line(image, tuple(corners[4]), tuple(corners[5]), front_color, thickness + 1)
        cv.line(image, tuple(corners[0]), tuple(corners[4]), front_color, thickness + 1)
        cv.line(image, tuple(corners[1]), tuple(corners[5]), front_color, thickness + 1)

        # Draw contact points at front bottom corners (yellow)
        cv.circle(image, tuple(corners[0]), 5, (0, 255, 255), -1)
        cv.circle(image, tuple(corners[1]), 5, (0, 255, 255), -1)

        return image

    def visualize(self, image: np.ndarray, bboxes_3d: List[BBox3D],
                  show_2d: bool = False, show_info: bool = True) -> np.ndarray:
        """Visualize 3D bounding boxes."""
        output = image.copy()

        for bbox in bboxes_3d:
            self.draw_3d_box(output, bbox, color=(0, 0, 255), thickness=2)

            if show_2d:
                cv.rectangle(output,
                            (bbox.bbox_2d.x1, bbox.bbox_2d.y1),
                            (bbox.bbox_2d.x2, bbox.bbox_2d.y2),
                            (100, 100, 100), 1)

            if show_info:
                info = f"{bbox.bbox_2d.class_name} | Pos:({bbox.center[0]:.1f},{bbox.center[1]:.1f})m"
                top_y = int(bbox.corners_image[4:8, 1].min()) - 5
                left_x = int(bbox.corners_image[:, 0].min())

                (tw, th), _ = cv.getTextSize(info, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv.rectangle(output, (left_x, top_y - th - 4), (left_x + tw + 4, top_y), (0, 0, 0), -1)
                cv.putText(output, info, (left_x + 2, top_y - 2),
                          cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return output

    def to_kitti_format(self, bboxes_3d: List[BBox3D]) -> str:
        """Export to KITTI format."""
        lines = []
        for bbox in bboxes_3d:
            line = (f"{bbox.bbox_2d.class_name} 0.0 0 0.0 "
                   f"{bbox.bbox_2d.x1} {bbox.bbox_2d.y1} {bbox.bbox_2d.x2} {bbox.bbox_2d.y2} "
                   f"{bbox.height:.2f} {bbox.width:.2f} {bbox.length:.2f} "
                   f"{bbox.center[0]:.2f} {bbox.center[1]:.2f} {bbox.center[2]:.2f} "
                   f"{bbox.yaw:.2f} {bbox.confidence:.2f}")
            lines.append(line)
        return '\n'.join(lines)


def main():
    print("=" * 60)
    print("3D BOUNDING BOX GENERATOR")
    print("=" * 60)

    calibration_dir = "/Users/navaneethmalingan/3D/Calibration"
    lookup_table_path = f"{calibration_dir}/calibration-lookup-table.npy"
    image_path = f"{calibration_dir}/1x.png"

    generator = BBox3DGenerator(lookup_table_path)

    image = cv.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    print(f"Loaded image: {image.shape}")

    if generator.model is not None:
        bboxes_3d = generator.process_image(image, conf_threshold=0.3)

        if bboxes_3d:
            print("\n" + "-" * 60)
            print("DETECTED VEHICLES (3D)")
            print("-" * 60)
            for i, bbox in enumerate(bboxes_3d):
                print(f"\nVehicle {i+1}: {bbox.bbox_2d.class_name}")
                print(f"  3D Center: ({bbox.center[0]:.2f}, {bbox.center[1]:.2f}) m")
                print(f"  Dimensions: {bbox.length:.1f}x{bbox.width:.1f}x{bbox.height:.1f} m")
                print(f"  Orientation: {np.degrees(bbox.yaw):.1f}°")

            output = generator.visualize(image, bboxes_3d, show_2d=False, show_info=True)
            output_path = f"{calibration_dir}/3d_bboxes_result.png"
            cv.imwrite(output_path, output)
            print(f"\nSaved to: {output_path}")

            with open(f"{calibration_dir}/detections_kitti.txt", 'w') as f:
                f.write(generator.to_kitti_format(bboxes_3d))

    print("\nDONE")


if __name__ == "__main__":
    main()
