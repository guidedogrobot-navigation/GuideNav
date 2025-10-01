#!/usr/bin/env python3
"""
Capture and save an image from /dev/video0 using OpenCV
Requires: pip install opencv-python
"""

import cv2
import datetime
import sys

def capture_image(device_path="/dev/video0", output_filename=None):
    """
    Capture an image from video device and save it
    
    Args:
        device_path: Path to video device (default: /dev/video0)
        output_filename: Output filename (default: auto-generated with timestamp)
    """
    
    # Extract device number from path (e.g., /dev/video0 -> 0)
    try:
        device_num = int(device_path.split('video')[-1])
    except (ValueError, IndexError):
        print(f"Error: Invalid device path '{device_path}'")
        return False
    
    # Initialize camera
    cap = cv2.VideoCapture(device_num)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera at {device_path}")
        return False
    
    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    try:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            return False
        
        # Generate filename if not provided
        if output_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"captured_image_{timestamp}.jpg"
        
        # Save image
        success = cv2.imwrite(output_filename, frame)
        
        if success:
            print(f"Image saved successfully: {output_filename}")
            print(f"Image dimensions: {frame.shape[1]}x{frame.shape[0]}")
            return True
        else:
            print("Error: Failed to save image")
            return False
            
    except Exception as e:
        print(f"Error during capture: {e}")
        return False
        
    finally:
        # Release camera
        cap.release()

def main():
    """Main function with command line argument support"""
    device_path = "/dev/video0"
    output_file = None
    
    # Simple command line argument parsing
    if len(sys.argv) > 1:
        device_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Attempting to capture from: {device_path}")
    success = capture_image(device_path, output_file)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()