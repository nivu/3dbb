# Stabilize the camera shake in a video using feature tracking
# See python stabilization.py --help

import argparse
import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
import json
import keyboard
import sys
from utils.roi import Roi


def load_roi_from_json(file_path):
    with open(file_path, 'r') as json_file:
        roi_points = json.load(json_file)
    return roi_points

class Stabilization:
      
    def __init__(self): 
       self.frame_index = 0
       self.calibration_frame = None
       self.track_features = []
       self.selected_track_features = []
       pass
     
    def get_calibration_frame(self, cap: cv.VideoCapture) -> np.ndarray:
        """Get the first frame of the video for calibration"""
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        success, self.calibration_frame = cap.read()
        if not success:
            print("Unable to read the first frame")
    
    def on_change_frame(self, pos):
            self.frame_index = pos
            
    def select_frame(self, cap: cv.VideoCapture):
        cv.namedWindow("Select Frame", cv.WINDOW_NORMAL)
        cv.setWindowProperty("Select Frame", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.createTrackbar("Frame Index", "Select Frame", 0, int(cap.get(cv.CAP_PROP_FRAME_COUNT)) - 1, self.on_change_frame)

        while True:
            cap.set(cv.CAP_PROP_POS_FRAMES, self.frame_index)
            ret, frame = cap.read()

            if not ret:
                print(f"Unable to read frame {self.frame_index}")
                break

            cv.imshow("Select Frame", frame)

            key = cv.waitKey(1)
            print("press s to select the frame or Esc to cancel")
            if key == ord('s'):  # Space key to select the current frame
                self.calibration_frame = frame
                break
            elif key == 27:  # Escape key to exit
                sys.exit()

        cv.destroyAllWindows()
        cv.imwrite("calibration\\stabilisation_image.png", self.calibration_frame)


    def get_tracking_features(self, num_features: int):
        """Get a list of feature coordinates that are easy to track"""
        frame_gray = cv.cvtColor(self.calibration_frame, cv.COLOR_BGR2GRAY)
        self.track_features= cv.goodFeaturesToTrack(
            frame_gray,
            maxCorners=num_features,
            qualityLevel=0.01,
            minDistance=20,
        )

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            # Check if the mouse click is near any tracking feature point
            for i, feature in enumerate(self.track_features):
                center_coordinate = (int(feature[0][0]), int(feature[0][1]))
                distance = np.sqrt((x - center_coordinate[0])**2 + (y - center_coordinate[1])**2)
                if distance <= 10:  # Check if the click is within a certain radius (e.g., 10 pixels) of the point
                    self.track_features = np.delete(self.track_features, i, 0)
                    self.selected_track_features.append(feature)
                   

    def select_tracking_features(self):
        cv.namedWindow("Select freature Points", cv.WINDOW_NORMAL)
        cv.setWindowProperty("Select freature Points", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.setMouseCallback("Select freature Points", self.on_mouse_click)
        print ("press s to save the points and continue")
        while True:
            frame_copy = self.calibration_frame.copy() # Create a copy of the frame to avoid modifying the original
            for i in self.track_features:
                center_coordinate = (int(i[0][0]), int(i[0][1]))
                cv.circle(frame_copy, center_coordinate, 5, (0, 0, 255), 4)
            if self.selected_track_features is not None:
                for i in self.selected_track_features:
                    center_coordinate = (int(i[0][0]), int(i[0][1]))
                    cv.circle(frame_copy, center_coordinate, 5, (0, 255, 0), 4)
            cv.imshow("Select freature Points", frame_copy)
            key = cv.waitKey(1) & 0xFF  # Use a small delay and update key handling to avoid freezing

            if key == ord('s'): # Press s to continue
                if self.selected_track_features <10:
                    print("Please select at least 10 points")
                    continue
                else:
                    break
        cv.destroyAllWindows()
    
    def save_stabilisation_calibration(self):
        if self.track_features is not None:
            with open("calibration\\track_features.json", "w") as json_file:
                json.dump(self.selected_track_features.tolist(), json_file)
                
    def load_stabilisation_calibration(self, filename="calibration\\track_features.json"):
        try:
            with open(filename, "r") as json_file:
                track_features_list = json.load(json_file)
                self.track_features = np.array(track_features_list, dtype=np.float32)
            self.calibration_frame = cv.imread("calibration\\stabilisation_image.png")
            
        except FileNotFoundError:
            print(f"File {filename} not found.")

    def stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Stabilize a video frame by matching the feature positions"""
        frame_features, status, _ = cv.calcOpticalFlowPyrLK(
            self.calibration_frame, frame, self.track_features, None
        )
        transformation, _ = cv.estimateAffine2D(
            frame_features[status == 1], self.track_features[status == 1]
        )
        return cv.warpAffine(frame, transformation, np.flip(frame.shape[0:2]))
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stabilize a video file using feature tracking."
    )
    parser.add_argument(
        "-f",
        "--file",
        help="the video file to stabilize",
        default= 'Beispieldaten\example-street.MP4'
        )
    parser.add_argument(
        "-n",
        "--nfeatures",
        default=80,
        type=int,
        metavar="N",
        help="number of features to track",
    )
    parser.add_argument(
        "-c",
        "--crop",
        metavar="DISTANCE",
        default=20,
        type=int,
        help="cropping distance on every side of the video",
    )
    max_contour_size = 600
    min_contour_size = 50
    args = parser.parse_args()

    # Open the video and get properties
    cap = cv.VideoCapture(args.file)
  
    if not cap.isOpened():
        print("Unable to open the video")
        
    video_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    frame_rate = int(cap.get(cv.CAP_PROP_FPS))
    # Set up the video writer
    out = cv.VideoWriter(
        "Beispieldaten\\stabilized_video.mp4", cv.VideoWriter_fourcc(*'XVID'),
        frame_rate, np.subtract(video_size, (2 * args.crop, 2 * args.crop)) )
    
    stabilization = Stabilization()
    roi = Roi()
    
    
    ################# calibration stabilisation #######################
    
    
    stabilization.select_frame(cap)
    #roi_points = roi.defineRoi(stabilization.calibration_frame)
    #stabilization.calibration_frame = roi.createRoi(roi_points, stabilization.calibration_frame)
    stabilization.get_tracking_features(args.nfeatures)
    stabilization.select_tracking_features()
    stabilization.save_stabilisation_calibration()
    
     ################# calibration stabilisation #######################

    # Determine the desired number of frames of the output file

    stabilization.load_stabilisation_calibration()
    roi_points = roi.load_roi_from_json("calibration\\roi.json")
    
    # Stabilize and write every frame
    stabilization.calibration_frame = roi.createRoi(roi_points, stabilization.calibration_frame)
    image_street = stabilization.calibration_frame[args.crop : -args.crop, args.crop : -args.crop]
    
    gray_image_street = cv.cvtColor(image_street, cv.COLOR_BGR2GRAY)
    denoised_image_street = cv.fastNlMeansDenoising(gray_image_street, None, 10, 10, 7)   
    #blurred_image_street = cv.GaussianBlur(gray_image_street, (5, 5), 0)
    #medianblur_image_street = cv.medianBlur(gray_image_street, 5)
    
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print(f"Unable to read frame")
            continue
        frame = roi.createRoi(roi_points, frame)
        frame = stabilization.stabilize_frame(frame)
        frame = frame[args.crop : -args.crop, args.crop : -args.crop]  # cropes the edges which are black
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        denoised_frame = cv.fastNlMeansDenoising(gray_frame, None, 10, 10, 7)   
        #blurred_frame = cv.GaussianBlur(gray_frame, (5, 5), 0)
        #medianblur_frame = cv.medianBlur(gray_frame, 5)
        
        FrameDifference = cv.absdiff(denoised_frame, denoised_image_street)
        edges = cv.Canny(gray_frame, 50, 150)
        circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=50, minRadius=10, maxRadius=100)
        if circles is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            circles = np.uint16(np.around(circles))
            # Draw the circles on the image
            for circle in circles[0, :]:
                # circle[0]: x-coordinate, circle[1]: y-coordinate, circle[2]: radius
                cv.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
        cv.imshow("edges", edges)
        
        cv.imshow("frame", frame)
        

        cv.waitKey(1)
        
        #out.write(frame)
    # Release the video reader and writer
    cap.release()
    out.release()
