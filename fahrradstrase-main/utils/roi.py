import cv2
import numpy as np
import json
import sys


class Roi:
    
    def __init__(self): 
       self.image =None
       self.roi_points = []
      
    def save_roi_to_json(self, path_image):
        with open(path_image, 'w') as json_file:
            json.dump(self.roi_points, json_file)

    def load_roi_from_json(self,path_image):
        with open(path_image, 'r') as json_file:
            self.roi_points = json.load(json_file)
        return self.roi_points

    # Define a callback function for mouse events
    def callback_select_roi(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_points.append((x, y))
            if len(self.roi_points) > 1:
                # Draw a line between the last two points
                cv2.line(self.image, self.roi_points[-2], self.roi_points[-1], (0, 255, 0), 2)
                cv2.imshow("Select ROI", self.image)
    
            else:
                cv2.circle(self.image, self.roi_points[-1], 2, (0, 255, 0), 2)
                cv2.imshow("Select ROI", self.image)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.roi_points) > 0:
                # Remove the last point to undo the last drawing
                self.roi_points.pop()
                self.image = cv2.imread("calibration/stabilisation_image.png")
                for i in range(len(self.roi_points)):
                    if i > 0:
                        cv2.line(self.image, self.roi_points[i - 1], self.roi_points[i], (0, 255, 0), 2)
                    else:
                        cv2.circle(self.image, self.roi_points[i], 2, (0, 255, 0), 2)
                cv2.imshow("Select ROI", self.image)
                
    def defineRoi(self, image):
        self.image= image
        # Create a window and set the callback function
        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", self.callback_select_roi)      

        # Main loop
        while True:
            cv2.imshow("Select ROI", self.image)
            key = cv2.waitKey(1) & 0xFF

            # Break the loop when 'c' is pressed (complete ROI selection)
            if key == ord("s"):
                break

            # Break the loop when 'q' is pressed (exit without selecting ROI)
            elif key == ord("q"):
                cv2.destroyAllWindows()
                sys.exit()

        # Create a mask of the selected ROI
        mask = np.zeros_like(self.image)
        cv2.fillPoly(mask, [np.array(self.roi_points)], (255, 255, 255))

        # Bitwise AND operation to extract the region within the ROI
        roi_cutout = cv2.bitwise_and(self.image, mask)

        # Display the original image and the cutout ROI
        cv2.imshow("Original Image", self.image)
        cv2.imshow("Cutout ROI", roi_cutout)
        self.save_roi_to_json("calibration/roi.json")
        # Wait for a key press and then close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.roi_points

    def createRoi(self, roi_points, image):
        # Create a mask of the selected ROI
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [np.array(roi_points)], (255, 255, 255))

        # Bitwise AND operation to extract the region within the ROI
        roi_cutout = cv2.bitwise_and(image, mask)

        return roi_cutout
