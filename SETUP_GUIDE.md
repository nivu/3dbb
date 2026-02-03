# Vehicle Detection & Distance Measurement System
## Setup Guide

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Camera Mounting Requirements](#camera-mounting-requirements)
3. [Calibration Process](#calibration-process)
4. [Running the Analysis](#running-the-analysis)
5. [GUI Application](#gui-application)
6. [Output Description](#output-description)

---

## System Overview

This system detects vehicles from traffic camera footage and measures:
- **3D Bounding Boxes**: Visual representation of vehicle position and dimensions
- **Edge-to-Edge Lateral Distance**: Gap between vehicles side-to-side (from one car's edge to another car's edge)
- **Front-to-Back Longitudinal Distance**: Distance from the front of one car to the back of another
- **Vehicle Speed**: Estimated speed in km/h
- **Overtaking Events**: Automatic detection of overtaking maneuvers with safety metrics

### System Flow
```
Camera Video Input
        ↓
  YOLOv8 Detection
        ↓
 Homography Calibration
        ↓
  3D Box Projection
        ↓
  Vehicle Tracking
        ↓
 Speed & Distance Calculation
        ↓
  Results & Visualization
```

---

## Camera Mounting Requirements

### Height Specifications

| Parameter | Minimum | Recommended | Maximum |
|-----------|---------|-------------|---------|
| **Mounting Height** | 3.0 m | 4.0 - 6.0 m | 10.0 m |
| **Tilt Angle** | 15° | 25° - 35° | 45° |

### Critical Requirements

1. **Static Mount**: Camera must be fixed (no pan/tilt/zoom during recording)
2. **Flat Ground Plane**: System assumes flat road surface
3. **Unobstructed View**: Clear line of sight to the road surface
4. **Stable Position**: No vibration or movement during operation

### Optimal Setup

```
                    Camera
                       │
                       │ Height: 4-6m
                       │ Tilt: 25-35°
                       ▼
    ┌──────────────────────────────────┐
    │                                  │
    │     Calibration Zone             │
    │     (Best Accuracy)              │
    │                                  │
    └──────────────────────────────────┘
           Road Surface
```

### Coverage Area

Based on the recommended mounting:
- **Longitudinal Range**: 40-50 meters along road
- **Lateral Range**: Full road width (8-12 meters typical)
- **Optimal Zone**: Center 60% of frame has highest accuracy

---

## Calibration Process

### Step 1: Prepare Reference Points

You need **minimum 4 reference points** (recommended: 7-8 points) with known real-world coordinates.

**Methods to obtain coordinates:**
- Total Station survey
- GPS with RTK correction
- Measured from known landmarks

**Reference Point Requirements:**
- Points must be visible in camera view
- Spread points across the entire calibration area
- Include points at different distances from camera
- All points on the ground plane (Z = 0)

**Example Reference Points File (`Max_Pla.txt`):**
```
Point_ID    X(m)     Y(m)     Z(m)
TS0001      0.000    0.000    0.000   # Origin
TS0002      5.250    -2.150   0.000
TS0003      10.500   1.340    0.000
TS0004      15.750   -1.890   0.000
TS0005      -5.200   2.100    0.000
TS0006      20.300   0.450    0.000
TS0007      -10.150  -3.200   0.000
```

### Step 2: Mark Points in Image

1. Capture a clear frame from your camera
2. Mark each reference point location in pixels
3. Save pixel coordinates alongside world coordinates

**Coordinate Mapping:**
```
Image Point (pixels)  →  World Point (meters)
(x_pixel, y_pixel)    →  (X_world, Y_world, Z_world)
```

### Step 3: Generate Calibration Lookup Table

The system uses a homography-based lookup table that maps every pixel to world coordinates.

```bash
# Run calibration script
python utils/calibration.py --input reference_points.json --output calibration-lookup-table.npy
```

**Lookup Table Structure:**
```python
lookup_table.shape = (width, height, 3)  # e.g., (1920, 1080, 3)
lookup_table[x, y] = [world_x, world_y, world_z]
```

### Step 4: Verify Calibration

Check calibration accuracy by comparing known points:
- **Target accuracy**: < 10mm error on reference points
- **Acceptable accuracy**: < 50mm error

---

## Running the Analysis

### Prerequisites

```bash
# Install dependencies
pip install ultralytics opencv-python numpy torch
```

### Step 1: Process Video with 3D Detection

```bash
cd Uni_west_1
python run_3d_bbox.py
```

This generates detection frames with 3D bounding boxes.

### Step 2: Run Speed Estimation

```bash
# Process all videos
python speed_estimator.py

# Process specific video (by index)
python speed_estimator.py 0    # First video
python speed_estimator.py 1 2  # Videos 1 and 2
```

**Output:** `output/*_speed.mp4` - Video with speed overlays

### Step 3: Run Distance Analysis

```bash
# Process all videos
python distance_analyzer.py

# Process specific video
python distance_analyzer.py 0
```

**Output:**
- `output/*_distance.mp4` - Video with distance lines
- `output/*_distances.csv` - Distance data per frame
- `output/*_overtaking.csv` - Detected overtaking events

---

## GUI Application

### Overview

The GUI provides two main interfaces:

### Part A: Calibration Interface

**Features:**
1. Upload calibration image
2. Mark reference points by clicking on image
3. Enter real-world coordinates for each point
4. Generate calibration lookup table
5. Verify calibration accuracy

**Workflow:**
```
┌─────────────────────────────────────────────────────┐
│  CALIBRATION INTERFACE                              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [Upload Image]                                     │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │                                             │   │
│  │         (Click to mark points)              │   │
│  │              ● P1                           │   │
│  │                    ● P2                     │   │
│  │         ● P3              ● P4              │   │
│  │                                             │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Point List:                                        │
│  ┌───────┬────────────┬─────────────────────────┐  │
│  │ ID    │ Pixel (x,y)│ World (X, Y, Z)         │  │
│  ├───────┼────────────┼─────────────────────────┤  │
│  │ P1    │ (245, 380) │ [0.00, 0.00, 0.00]      │  │
│  │ P2    │ (890, 290) │ [5.25, -2.15, 0.00]     │  │
│  │ P3    │ (120, 520) │ [-5.20, 2.10, 0.00]     │  │
│  │ P4    │ (1450, 410)│ [10.50, 1.34, 0.00]     │  │
│  └───────┴────────────┴─────────────────────────┘  │
│                                                     │
│  [Calibrate]  [Save]  [Load]                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Part B: Analysis Interface

**Features:**
1. Upload video file
2. Automatic vehicle detection and tracking
3. Real-time 3D bounding box visualization
4. Distance measurement display (edge-to-edge and front-to-back)
5. Save annotated video and CSV data

**Workflow:**
```
┌─────────────────────────────────────────────────────┐
│  ANALYSIS INTERFACE                                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [Upload Video]  [Select Calibration]  [▶ Start]   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │                                             │   │
│  │    ┌─────┐         ┌─────┐                  │   │
│  │    │ Car │←─ 2.3m ─│ Car │                  │   │
│  │    │  A  │         │  B  │                  │   │
│  │    └─────┘         └─────┘                  │   │
│  │      ↑               ↑                      │   │
│  │    42 km/h         38 km/h                  │   │
│  │                                             │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ══════════════════════════════════════════════    │
│  Frame: 1234/5000  Time: 00:49.36                  │
│                                                     │
│  Distance Measurements:                             │
│  ┌──────────────────────────────────────────────┐  │
│  │ Vehicle A ↔ Vehicle B                        │  │
│  │   Lateral (edge-to-edge):  2.3 m            │  │
│  │   Longitudinal (front-back): 4.1 m          │  │
│  │   Total: 4.7 m                              │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  [⏸ Pause]  [Export CSV]  [Save Video]            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Output Description

### Video Output

**3D Bounding Boxes:**
- Each vehicle shown with projected 3D box
- Color-coded by vehicle type
- Speed displayed above each vehicle

**Distance Lines:**
- **Yellow lines**: Edge-to-edge lateral distance (side-to-side gap)
- **Cyan lines**: Front-to-back longitudinal distance
- Distance values displayed at line midpoint

### CSV Output

**Distance Data (`*_distances.csv`):**
```csv
frame,time,vehicle_a,vehicle_b,pos_a_x,pos_a_y,pos_b_x,pos_b_y,distance_longitudinal,distance_lateral,distance_total
1234,49.36,1,2,5.23,1.45,9.34,3.75,4.11,2.30,4.71
```

| Column | Description |
|--------|-------------|
| frame | Frame number |
| time | Timestamp in seconds |
| vehicle_a, vehicle_b | Vehicle tracking IDs |
| pos_a_x, pos_a_y | Vehicle A position (meters) |
| pos_b_x, pos_b_y | Vehicle B position (meters) |
| distance_longitudinal | Front-to-back distance (meters) |
| distance_lateral | Edge-to-edge side distance (meters) |
| distance_total | Euclidean distance (meters) |

**Overtaking Events (`*_overtaking.csv`):**
```csv
frame_start,frame_end,vehicle_a,vehicle_b,min_lateral_dist_m,min_total_dist_m,overtaking_vehicle
1200,1350,1,2,1.85,2.34,2
```

### Distance Types Explained

```
    LATERAL DISTANCE (Edge-to-Edge)
    ←─────────────────────────────→

    ┌─────────┐       ┌─────────┐
    │         │       │         │
    │  Car A  │  GAP  │  Car B  │
    │         │       │         │
    └─────────┘       └─────────┘


    LONGITUDINAL DISTANCE (Front-to-Back)

    ┌─────────┐
    │  Car A  │
    │  (rear) │
    └─────────┘
         ↑
         │ GAP
         ↓
    ┌─────────┐
    │ (front) │
    │  Car B  │
    └─────────┘
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Poor calibration accuracy | Insufficient reference points | Add more points (minimum 7-8) |
| Vehicles not detected | Low confidence threshold | Adjust detection threshold in config |
| Erratic speed values | Poor calibration in edge regions | Re-calibrate with more edge points |
| Missing distance data | Vehicles too far apart | Adjust proximity thresholds |

### Performance Tips

1. Use GPU acceleration (CUDA/Metal) for faster processing
2. Process videos at native resolution for best accuracy
3. Ensure calibration points cover the entire analysis area
4. Verify calibration before processing large batches

---

## File Structure

```
project/
├── calibration-lookup-table.npy   # Calibration data
├── Max_Pla.txt                    # Reference points
├── yolov8n.pt                     # Detection model
├── run_3d_bbox.py                 # 3D detection script
├── speed_estimator.py             # Speed calculation
├── distance_analyzer.py           # Distance analysis
├── interactive_bbox_tool.py       # Manual calibration tool
└── output/
    ├── *_speed.mp4                # Speed annotated videos
    ├── *_distance.mp4             # Distance annotated videos
    ├── *_distances.csv            # Distance data
    └── *_overtaking.csv           # Overtaking events
```

---

## Quick Start Checklist

- [ ] Mount camera at 4-6m height with 25-35° tilt
- [ ] Measure 7-8 reference points with Total Station or GPS
- [ ] Capture calibration image from camera
- [ ] Mark reference points and enter coordinates
- [ ] Generate calibration lookup table
- [ ] Verify calibration accuracy (< 50mm error)
- [ ] Run analysis on video
- [ ] Review output videos and CSV data

---

*For additional help or to report issues, visit the project repository.*
