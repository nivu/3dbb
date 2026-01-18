"""
Run 3D Bounding Box Detection on Uni_west_1 Dataset
"""

import sys
import os
sys.path.insert(0, '/Users/navaneethmalingan/3D')

import cv2 as cv
import numpy as np
from bbox_3d_generator import BBox3DGenerator

# Dataset paths
DATASET_DIR = "/Users/navaneethmalingan/3D/Uni_west_1"
LOOKUP_TABLE = f"{DATASET_DIR}/calibration-lookup-table.npy"
OUTPUT_DIR = f"{DATASET_DIR}/output"

# Video files
VIDEOS = [
    "GOPR0574.MP4",
    "GOPR0575.MP4",
    "GOPR0581.MP4"
]

def process_video(generator, video_path, output_dir, sample_interval=30):
    """Process video and save frames with 3D bounding boxes."""
    video_name = os.path.basename(video_path).replace('.MP4', '')
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    print(f"\nProcessing: {video_name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}, FPS: {fps:.1f}")
    print(f"  Sampling every {sample_interval} frames")

    frame_idx = 0
    processed = 0
    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            # Process frame
            bboxes_3d = generator.process_image(frame, conf_threshold=0.3)

            if bboxes_3d:
                # Visualize and save
                output_frame = generator.visualize(frame, bboxes_3d, show_2d=True, show_info=True)
                output_path = os.path.join(video_output_dir, f"frame_{frame_idx:06d}.jpg")
                cv.imwrite(output_path, output_frame)

                # Store detections
                for bbox in bboxes_3d:
                    all_detections.append({
                        'frame': frame_idx,
                        'class': bbox.bbox_2d.class_name,
                        'x': bbox.center[0],
                        'y': bbox.center[1],
                        'length': bbox.length,
                        'width': bbox.width,
                        'yaw': np.degrees(bbox.yaw)
                    })

                processed += 1
                if processed % 10 == 0:
                    print(f"  Processed {processed} frames with detections...")

        frame_idx += 1

    cap.release()

    # Save detections to file
    if all_detections:
        det_path = os.path.join(video_output_dir, "detections.txt")
        with open(det_path, 'w') as f:
            f.write("frame,class,x_m,y_m,length_m,width_m,yaw_deg\n")
            for d in all_detections:
                f.write(f"{d['frame']},{d['class']},{d['x']:.2f},{d['y']:.2f},"
                       f"{d['length']:.2f},{d['width']:.2f},{d['yaw']:.1f}\n")
        print(f"  Saved {len(all_detections)} detections to {det_path}")

    print(f"  Done! Saved {processed} frames to {video_output_dir}")
    return all_detections


def process_single_frame(generator, video_path, frame_num=0):
    """Extract and process a single frame for quick testing."""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return None

    cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: Could not read frame {frame_num}")
        return None

    return frame


def main():
    print("=" * 60)
    print("3D BOUNDING BOX - Uni_west_1 Dataset")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize generator
    print(f"\nLoading calibration from: {LOOKUP_TABLE}")
    generator = BBox3DGenerator(LOOKUP_TABLE)

    # Quick test on first frame of first video
    print("\n" + "-" * 60)
    print("QUICK TEST - First frame of GOPR0574.MP4")
    print("-" * 60)

    test_video = os.path.join(DATASET_DIR, VIDEOS[0])
    frame = process_single_frame(generator, test_video, frame_num=100)

    if frame is not None:
        bboxes_3d = generator.process_image(frame, conf_threshold=0.3)

        if bboxes_3d:
            print(f"\nFound {len(bboxes_3d)} vehicles:")
            for i, bbox in enumerate(bboxes_3d):
                print(f"  {i+1}. {bbox.bbox_2d.class_name}: "
                      f"Position ({bbox.center[0]:.2f}, {bbox.center[1]:.2f})m, "
                      f"Yaw: {np.degrees(bbox.yaw):.1f}°")

            # Save test output
            output = generator.visualize(frame, bboxes_3d, show_2d=True, show_info=True)
            test_output = os.path.join(OUTPUT_DIR, "test_frame.jpg")
            cv.imwrite(test_output, output)
            print(f"\nTest frame saved to: {test_output}")
        else:
            # Save the frame anyway for inspection
            test_output = os.path.join(OUTPUT_DIR, "test_frame_no_detection.jpg")
            cv.imwrite(test_output, frame)
            print(f"\nNo vehicles detected. Frame saved to: {test_output}")

    # Ask user if they want to process all videos
    print("\n" + "=" * 60)
    response = input("Process all videos? (y/n): ").strip().lower()

    if response == 'y':
        for video in VIDEOS:
            video_path = os.path.join(DATASET_DIR, video)
            if os.path.exists(video_path):
                process_video(generator, video_path, OUTPUT_DIR, sample_interval=30)
            else:
                print(f"WARNING: {video} not found")

    print("\nDONE!")


if __name__ == "__main__":
    main()
