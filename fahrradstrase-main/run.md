# Calibration Tool - User Guide

## Quick Start

```bash
python main.py -f <image_path>
```

## Usage Instructions

| Step | Action |
|------|--------|
| 1 | Click on **left image** to select a zoom area |
| 2 | Click on **right (zoomed) image** to mark a calibration point |
| 3 | Use **W/S** or **UP/DOWN** arrows to scroll through labels |
| 4 | Press **ENTER** to confirm the label selection |
| 5 | Press **C** or **Right-click** to cancel selection |
| 6 | **Right-click** to undo the last marked point |
| 7 | Press **ESC** when done to finish calibration |

> **Note:** Select at least 4 points for calibration to work.

**Output:** `calibration/calibration-lookup-table.npy`

---

## Understanding Max_Pla.txt

The `Max_Pla.txt` file contains **real-world coordinates** measured physically on the street using surveying equipment.

### How It Was Created

1. **"TS" prefix** = Total Station points - a surveying instrument that measures angles and distances with high precision using laser

2. **Physical measurement process:**
   - A surveyor places markers/targets at various points on the street
   - The Total Station measures each point's 3D coordinates (X, Y, Z in meters)
   - Results are exported to a file

3. **Coordinate system:**
   - Origin (0, 0, 0) is set at a reference point
   - X = lateral position (left/right)
   - Y = longitudinal position (forward/back along street)
   - Z = elevation (height above ground plane, mostly ~0 for street level)

---

## Mapping TS Points to Pixels

This is the key **manual step** in calibration - you need visual markers in the image.

### How It Works

The surveyor who created `Max_Pla.txt` would have:

1. Placed **physical markers** on the street (painted dots, targets, cones, tape marks)
2. Measured each marker's position with the Total Station → saved as TS0004, TS0005, etc.
3. Photographed/recorded the scene with markers visible

### Your Task When Calibrating

```
Image (pixels)                    Max_Pla.txt (meters)
     │                                  │
     ▼                                  ▼
┌─────────────┐                   TS0038: (6.46, 11.23)
│   ●  ●      │  ──── YOU ────►   TS0037: (10.50, 15.53)
│      ●      │     visually      TS0036: (23.23, 43.61)
│  ●      ●   │     match         ...
└─────────────┘
```

You look at the image, find a visible marker, click on it, then select which TS label it corresponds to.

---

## If No Markers Are Visible

If you don't have visible markers in the image, you need:

1. A **reference diagram** showing where each TS point is located on the street
2. The **original calibration image** with markers visible
3. **Documentation from the surveyor** explaining the marker positions

### Alternative: Use Identifiable Features

If you have a diagram showing approximate TS locations, use **fixed features** as reference points:

| Possible Reference Points | Example |
|---------------------------|---------|
| Corners of white lane markings | Where lines meet |
| Base of poles/signs | Traffic light pole base |
| Corners of red bike lane sections | Sharp edges |
| Road marking symbols | Arrow tips, bike symbol corners |

---

## Action Items

1. **Ask Julian for:**
   - The calibration diagram/map showing TS point positions
   - Or an annotated image showing where measurements were taken
   - Or Oleg's original calibration files

2. **Read Oleg's thesis** - it likely explains the calibration setup and which features were used as reference points

> **Important:** The `Max_Pla.txt` coordinates are useless without knowing their physical location relative to visible features in the image.
