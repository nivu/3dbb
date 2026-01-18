from argparse import ArgumentParser
import json
import cv2 as cv
import numpy as np
from dataclasses import dataclass


@dataclass
class Plane:
    normal: np.ndarray
    origin: np.ndarray


@dataclass
class Axes:
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray

    def to_dict(self) -> dict:
        return {
            "origin": self.origin.tolist(),
            "u": self.u.tolist(),
            "v": self.v.tolist(),
        }

    @staticmethod
    def from_dict(axes: dict):
        return Axes(axes["origin"], axes["u"], axes["v"])


def load_calibration_points(file: str) -> tuple[np.ndarray, np.ndarray]:
    with open(file, "r") as fp:
        points = json.load(fp)
    world_points = np.array(
        [
            [point["world"]["x"], point["world"]["y"], point["world"]["z"]]
            for point in points
        ]
    )
    image_points = np.array(
        [[point["image"]["x"], point["image"]["y"]] for point in points]
    )
    return world_points, image_points


def approximate_plane(points: np.ndarray) -> Plane:
    """Fits a plane to a set of points
    points: nx3 array of row vectors

    https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
    """
    center = np.mean(points, axis=0, keepdims=True)
    svd = np.linalg.svd(np.transpose(points - center))
    normal = svd.U[:, -1]
    return Plane(normal.reshape(1, 3), center)


def move_to_plane(points: np.ndarray, plane: Plane) -> np.ndarray:
    """Moves the points to their closest point on the plane
    points: Array of row vectors
    plane: Plane to move the points to

    https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
    """
    v = points - plane.origin
    d = v @ plane.normal.transpose()
    return points - d * plane.normal


def create_axes(point: np.ndarray, plane: Plane) -> Axes:
    """Creates axes that span a coordinate system on the plane
    point: Point on the plane which must not be the plane's origin
    plane: Plane to create the axes on
    """
    u = point - plane.origin
    v = np.cross(u, plane.normal)
    return Axes(plane.origin, u / np.linalg.norm(u), v / np.linalg.norm(v))


def transform_to_axes(points: np.ndarray, axes: Axes) -> np.ndarray:
    """Transforms the points from the 3D world to the 2D plane coordinate space
    points: Array of row vectors of points on the plane
    axes: Axes on the plane
    """
    # Prepare for solving in form ax=b
    a = np.concatenate([axes.u, axes.v])
    a = a.transpose()

    b = points - axes.origin
    b = b.transpose()
    coordinates = np.linalg.solve(a[:2], b[:2])

    return coordinates.transpose()


def find_homography(image_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    """Finds the homography matrix H
    image_points: Array of row vectors of points in the image
    world_points: Array of row vectors of points in the world, transformed to the plane
    """
    return cv.findHomography(image_points, world_points)[0]


def calibrate(
    image_points: np.ndarray, world_points: np.ndarray
) -> tuple[np.ndarray, Axes]:
    """Calibrate the camera using the image and world points
    image_points: Array of row vectors of points in the image
    world_points: Array of row vectors of points in the world

    returns:
    H: Homography matrix
    axes: Axes of the plane coordinate system
    """
    plane = approximate_plane(world_points)
    world_points = move_to_plane(world_points, plane)
    axes = create_axes(world_points[0], plane)
    world_points = transform_to_axes(world_points, axes)
    H = find_homography(image_points, world_points)
    return H, axes


def get_world_points(image_points: np.ndarray, H: np.ndarray, axes: Axes) -> np.ndarray:
    """Estimate the 3D world points from given 2D image points
    image_points: nx2 array of points in the image space in form [[left, top], ...]
    H: homography matrix
    axes: axes in the plane
    """
    image_points_ones = np.concatenate(
        [image_points, np.ones([image_points.shape[0], 1])], axis=1
    )
    plane_points = np.matmul(H, image_points_ones.transpose()).transpose()
    plane_points = plane_points[:, :2] / plane_points[:, 2, np.newaxis]
    return np.matmul(plane_points, np.concatenate([axes.u, axes.v])) + axes.origin


def get_lookup_table(width: int, height: int, H: np.ndarray, axes: Axes) -> np.ndarray:
    """Computes the lookup table of image points -> world points
    width: width of the image
    height: height of the image
    H: homography matrix
    axes: axes in the plane
    """
    cols = np.arange(width).repeat(height)
    rows = np.array(list(range(height)) * width)

    image_points = np.column_stack([cols, rows])
    world_points = get_world_points(image_points, H, axes)
    return world_points.reshape(width, height, -1)


if __name__ == "__main__":
    parser = ArgumentParser(description="Calibrate the camera with calibration points")
    parser.add_argument(
        "-f",
        "--file",
        help="the file containing the calibration points as generated by the Camera Calibration App",
        default="calibration/calibration-points.json",
    )
    args = parser.parse_args()

    world_points, image_points = load_calibration_points(args.file)

    # Calibration, use these function in your own program
    H, axes = calibrate(image_points, world_points)
    lookup_table = get_lookup_table(width=1920, height=1080, H=H, axes=axes)

    np.save("calibration/calibration-lookup-table", lookup_table)

    calibration = {"axes": axes.to_dict(), "H": H.tolist()}
    with open("calibration/calibration.json", "w") as fp:
        json.dump(calibration, fp, indent=4)
