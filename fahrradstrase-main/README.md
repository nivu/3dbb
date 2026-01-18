# Camera Calibration

Software package to perform camera calibration.
Maps the 2D image space to the 3D world space.

## Installation

1. Clone or download this repository
2. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Modules

The project consists or serveral modules:

- `app/main.py`: An app to mark calibration points in an image
- `calibration.py`: A calibration script to get the transformation
- `PositionEstimator.py`: A class to perform the transformation
- `stabilization.py`: A script for video stabilization

## Execution

For all modules enter `python <module.py> -h` to get information about its execution.

Open the Camera Calibration App with:

```bash
python app/main.py <image>
```

where `<image>` is the name of the calibration image file, for example `python app/main.py image.png`.

Enter the calibration points in the Camera Calibration App:

1. Set the zoom window by clicking in the overview on the left side.
2. Click an calibration point in the zoomed view on the right side.
3. Enter the world coordinate in the form `<id>,<x>,<y>,<z>`, for example `TS0004,13.2363,-8.4928,0.2282`.
4. Repeat for all points that should be used for calibration.

The Camera Calibration App generates a file `calibration-points.json` which contains all calibration points.

Call the calibration script to perform the calibration with these points:

```bash
python calibration.py
```

It will create a calibration file `calibration.json`.

Instantiate a `PositionEstimator` and load the calibration.
Call its method `get_world_points` to transform 2D image points to 3D world points.

You can also call `PositionEstimator.py` to estimate a single point.
See `python PositionEstimator.py -h` for more information.
