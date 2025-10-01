import cv2
import os
import re
import numpy as np
from pathlib import Path

def create_video_from_keyframes(input_dir, output_path, fps=30, duration_per_frame=1.0):
    """
    Creates a video from keyframe images in a directory.
    
    Args:
        input_dir (str): Directory containing keyframe images
        output_path (str): Output video file path (e.g., 'output.mp4')
        fps (int): Frames per second for output video
        duration_per_frame (float): How long each keyframe should be displayed (seconds)
    """
    
    # Get all keyframe files and sort them numerically
    input_path = Path(input_dir)
    keyframe_files = []
    
    # Find all keyframe files
    for file in input_path.glob('keyframe_*.jpg'):
        # Extract frame number from filename
        match = re.search(r'keyframe_(\d+)\.jpg', file.name)
        if match:
            frame_num = int(match.group(1))
            keyframe_files.append((frame_num, str(file)))
    
    # Sort by frame number
    keyframe_files.sort(key=lambda x: x[0])
    
    if not keyframe_files:
        raise ValueError(f"No keyframe files found in {input_dir}")
    
    print(f"Found {len(keyframe_files)} keyframes")
    
    # Read first image to get dimensions
    first_frame = cv2.imread(keyframe_files[0][1])
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {keyframe_files[0][1]}")
    
    height, width, channels = first_frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate how many times to repeat each frame
    frames_per_keyframe = int(fps * duration_per_frame)
    
    try:
        for frame_num, file_path in keyframe_files:
            print(f"Processing keyframe {frame_num}: {file_path}")
            
            # Read the keyframe
            frame = cv2.imread(file_path)
            if frame is None:
                print(f"Warning: Could not read {file_path}, skipping...")
                continue
            
            # Resize frame if needed
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            # Write the frame multiple times to create duration
            for _ in range(frames_per_keyframe):
                out.write(frame)
        
        print(f"Video created successfully: {output_path}")
        print(f"Total duration: {len(keyframe_files) * duration_per_frame:.2f} seconds")
        
    finally:
        # Release the video writer
        out.release()
        cv2.destroyAllWindows()

def create_video_with_transitions(input_dir, output_path, fps=30, keyframe_duration=2.0, transition_duration=0.5):
    """
    Creates a video with smooth transitions between keyframes.
    
    Args:
        input_dir (str): Directory containing keyframe images
        output_path (str): Output video file path
        fps (int): Frames per second
        keyframe_duration (float): How long each keyframe is displayed
        transition_duration (float): Duration of transition between keyframes
    """
    
    input_path = Path(input_dir)
    keyframe_files = []
    
    # Find and sort keyframe files
    for file in input_path.glob('keyframe_*.jpg'):
        match = re.search(r'keyframe_(\d+)\.jpg', file.name)
        if match:
            frame_num = int(match.group(1))
            keyframe_files.append((frame_num, str(file)))
    
    keyframe_files.sort(key=lambda x: x[0])
    
    if len(keyframe_files) < 2:
        raise ValueError("Need at least 2 keyframes for transitions")
    
    # Read all keyframes
    keyframes = []
    for frame_num, file_path in keyframe_files:
        frame = cv2.imread(file_path)
        if frame is not None:
            keyframes.append(frame)
    
    if not keyframes:
        raise ValueError("No valid keyframes found")
    
    # Get dimensions from first frame
    height, width = keyframes[0].shape[:2]
    
    # Resize all frames to same size
    for i, frame in enumerate(keyframes):
        if frame.shape[:2] != (height, width):
            keyframes[i] = cv2.resize(frame, (width, height))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    keyframe_frames = int(fps * keyframe_duration)
    transition_frames = int(fps * transition_duration)
    
    try:
        for i in range(len(keyframes)):
            # Write keyframe
            for _ in range(keyframe_frames):
                out.write(keyframes[i])
            
            # Create transition to next frame (if not last frame)
            if i < len(keyframes) - 1:
                for t in range(transition_frames):
                    alpha = t / transition_frames
                    # Blend current and next frame
                    blended = cv2.addWeighted(keyframes[i], 1-alpha, keyframes[i+1], alpha, 0)
                    out.write(blended)
        
        print(f"Video with transitions created: {output_path}")
        
    finally:
        out.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Basic video creation
    input_directory= "/media/hochul/2TB/IJRR_Topomaps/topo_gail2library_1m_topogen_blip"
    # input_directory = "path/to/your/keyframes"  # Change this to your directory
    output_video = "/media/hochul/2TB/IJRR_Topomaps/topo_gail2library_1m_topogen_blip/output_video.mp4"
    
    try:
        # Simple version - each keyframe shown for 1 second
        create_video_from_keyframes(input_directory, output_video, fps=30, duration_per_frame=1.0)
        
        # Version with smooth transitions
        # create_video_with_transitions(input_directory, "output_with_transitions.mp4", 
        #                             fps=30, keyframe_duration=2.0, transition_duration=0.5)
        
    except Exception as e:
        print(f"Error: {e}")