import cv2 as cv
from dataclasses import dataclass
import numpy as np


@dataclass
class Position_3d:
    label: str
    x: float
    y: float
    z: float


class PointSelector:
    def __init__(self, frame, sizeZoomArea=100, zoom_factor=30):
        self.points_3d = self.read_config()

        self.frame = frame
        self.height, self.width, self.channels = frame.shape
        self.sizeZoomArea = sizeZoomArea
        self.zoom_factor = zoom_factor

        self.size_zoomedArea = sizeZoomArea * zoom_factor

        self.cursor_pos = np.array([0, 0])
        self.selected_zoomArea_position = np.array([0, 0])
        self.selected_points = []

        self.selected_point_3d = []

        # For keyboard-based label selection
        self.pending_click = None
        self.current_label_index = 0
        self.selecting_label = False

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE:
            self.cursor_pos = np.array([x, y])

        if event == cv.EVENT_LBUTTONDOWN:
            if self.selecting_label:
                # Ignore clicks while selecting a label
                return

            if x > self.width and y < self.size_zoomedArea:
                pixel_x = (
                    (x - self.width) / self.zoom_factor
                    + self.selected_zoomArea_position[0]
                    - self.sizeZoomArea / 2
                )
                pixel_y = (
                    y / self.zoom_factor
                    + self.selected_zoomArea_position[1]
                    - self.sizeZoomArea / 2
                )

                # Store pending click and enter label selection mode
                self.pending_click = np.array([pixel_x, pixel_y])
                self.selecting_label = True
                self.current_label_index = 0
                print("\n--- SELECT LABEL ---")
                print("Use UP/DOWN arrows to navigate, ENTER to confirm, C to cancel")
                self._print_current_label()

            elif x < self.width and y < self.height:
                self.selected_zoomArea_position = np.array([x, y])

        if event == cv.EVENT_RBUTTONDOWN:
            if self.selecting_label:
                # Cancel label selection
                self.selecting_label = False
                self.pending_click = None
                print("Selection canceled")
                return

            if len(self.selected_points) > 0:
                self.points_3d.append(self.selected_point_3d[-1])

                self.selected_points.pop()
                self.selected_point_3d.pop()

                print("Point deleted")

    def _print_current_label(self):
        if len(self.points_3d) > 0:
            point = self.points_3d[self.current_label_index]
            print(f"  -> [{self.current_label_index + 1}/{len(self.points_3d)}] {point.label} ({point.x:.2f}, {point.y:.2f}, {point.z:.2f})")

    def handle_label_selection_key(self, key):
        if not self.selecting_label:
            return

        # UP arrow (various key codes)
        if key in [0, 82, 119]:  # up arrow, 'w'
            self.current_label_index = (self.current_label_index - 1) % len(self.points_3d)
            self._print_current_label()

        # DOWN arrow (various key codes)
        elif key in [1, 84, 115]:  # down arrow, 's'
            self.current_label_index = (self.current_label_index + 1) % len(self.points_3d)
            self._print_current_label()

        # ENTER to confirm
        elif key in [13, 10]:
            selected_point = self.points_3d[self.current_label_index]
            self.selected_point_3d.append(selected_point)
            self.points_3d.remove(selected_point)
            self.selected_points.append(self.pending_click)

            print(f"\nConfirmed: {selected_point.label}")
            print(f"Selected points: {[p.label for p in self.selected_point_3d]}")
            print(f"Remaining points: {len(self.points_3d)}")

            self.selecting_label = False
            self.pending_click = None

        # C to cancel
        elif key == ord('c'):
            print("Selection canceled")
            self.selecting_label = False
            self.pending_click = None

    def getZoomview(self):
        frame_copy = self.frame.copy()

        topLeftCornerZoomArea = (
            int(
                max(
                    self.sizeZoomArea / 2,
                    self.selected_zoomArea_position[0] - self.sizeZoomArea / 2,
                )
            ),
            int(
                max(
                    self.sizeZoomArea / 2,
                    self.selected_zoomArea_position[1] - self.sizeZoomArea / 2,
                )
            ),
        )

        bottomRightCornerZoomArea = (
            int(
                max(
                    topLeftCornerZoomArea[0] + self.sizeZoomArea,
                    self.selected_zoomArea_position[0] + self.sizeZoomArea / 2,
                )
            ),
            int(
                max(
                    topLeftCornerZoomArea[1] + self.sizeZoomArea,
                    self.selected_zoomArea_position[1] + self.sizeZoomArea / 2,
                )
            ),
        )

        zoomed_cropped = frame_copy[
            topLeftCornerZoomArea[1] : bottomRightCornerZoomArea[1],
            topLeftCornerZoomArea[0] : bottomRightCornerZoomArea[0],
        ]
        zoomed_region = cv.resize(
            zoomed_cropped,
            None,
            fx=self.zoom_factor,
            fy=self.zoom_factor,
            interpolation=cv.INTER_LINEAR,
        )

        return zoomed_region

    def selectPoints(
        self,
    ):
        print("\n" + "=" * 50)
        print("CALIBRATION POINT SELECTOR")
        print("=" * 50)
        print("1. Click on LEFT image to select zoom area")
        print("2. Click on RIGHT (zoomed) image to mark a point")
        print("3. Use W/S or UP/DOWN to select label, ENTER to confirm")
        print("4. Right-click to undo last point")
        print("5. Press ESC to finish")
        print("=" * 50 + "\n")

        cv.namedWindow("select_points", cv.WINDOW_NORMAL)
        cv.setMouseCallback("select_points", self.on_mouse_click)

        sizeMainWindow = (
            self.height + self.sizeZoomArea * self.zoom_factor,
            self.width + self.sizeZoomArea * self.zoom_factor,
            self.channels,
        )
        main_window = np.zeros(sizeMainWindow, dtype=np.uint8)

        while True:
            frame_copy = self.frame.copy()
            topLeftCornerZoomArea = (self.cursor_pos[0] - 50, self.cursor_pos[1] - 50)
            bottomRightCornerZoomArea = (
                self.cursor_pos[0] + 50,
                self.cursor_pos[1] + 50,
            )

            cv.rectangle(
                frame_copy,
                topLeftCornerZoomArea,
                bottomRightCornerZoomArea,
                (0, 255, 0),
                2,
            )
            for i in self.selected_points:
                cv.circle(frame_copy, (int(i[0]), int(i[1])), 5, (0, 255, 0), -1)

            zoomed_region = self.getZoomview()

            main_window[0 : self.height, 0 : self.width] = frame_copy.copy()

            main_window[
                0 : self.sizeZoomArea * self.zoom_factor,
                self.width : self.width + self.sizeZoomArea * self.zoom_factor,
            ] = zoomed_region.copy()

            if (
                self.cursor_pos[0] > self.width
                and self.cursor_pos[1] < self.sizeZoomArea * self.zoom_factor
            ):
                cv.circle(
                    main_window,
                    (int(self.cursor_pos[0]), int(self.cursor_pos[1])),
                    3,
                    (0, 255, 0),
                    -1,
                )

            # Show label selection overlay when in selection mode
            if self.selecting_label and len(self.points_3d) > 0:
                overlay = main_window.copy()
                cv.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
                cv.addWeighted(overlay, 0.7, main_window, 0.3, 0, main_window)

                cv.putText(main_window, "SELECT LABEL (W/S or arrows, ENTER to confirm)",
                           (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                point = self.points_3d[self.current_label_index]
                label_text = f"[{self.current_label_index + 1}/{len(self.points_3d)}] {point.label}"
                coord_text = f"({point.x:.2f}, {point.y:.2f}, {point.z:.2f})"

                cv.putText(main_window, label_text, (20, 70),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(main_window, coord_text, (20, 100),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv.putText(main_window, "Press C or Right-click to cancel", (20, 130),
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            cv.imshow("select_points", main_window)
            key = cv.waitKey(1) & 0xFF

            # Handle keyboard input for label selection
            if self.selecting_label:
                self.handle_label_selection_key(key)

            if key == 27:  # 27 is the ASCII value for the "Esc" key
                if self.selecting_label:
                    # Cancel label selection first
                    self.selecting_label = False
                    self.pending_click = None
                    print("Selection canceled")
                else:
                    print("Esc key pressed. Exiting.")
                    cv.destroyAllWindows()
                    return self.selected_point_3d, self.selected_points

            if len(self.points_3d) == 0:
                print("All points selected.")
                cv.destroyAllWindows()
                return self.selected_point_3d, self.selected_points

    def read_config(
        self,
    ):
        with open("Beispieldaten/Max_Pla.txt", "r") as file:
            lines = file.readlines()

        points_3d = []

        # Loop through each line and extract values
        for line in lines:
            # Split the line into name, x, y, and z using commas as separators
            data = line.strip().split(",")

            # Check if the line contains at least four values
            if len(data) >= 4:
                try:
                    # Extract values and append to respective lists
                    position_3d = Position_3d(
                        data[0], float(data[1]), float(data[2]), float(data[3])
                    )
                    points_3d.append(position_3d)
                except ValueError:
                    print("Invalid data found in line: %s" % line)

        return points_3d


if __name__ == "__main__":
    calibration_image = cv.imread("./Beispieldaten/example-street.png")

    pointSelector = PointSelector(calibration_image)
    selected_label, selected_points = pointSelector.selectPoints()

    print("Points selected")
