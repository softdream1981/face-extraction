import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

def extract_unique_faces(video_path, output_dir="extracted_faces", min_face_size=(30, 30), detection_interval=30):
    """
    Extract unique faces from a video file
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted faces
        min_face_size: Minimum face size to detect (width, height)
        detection_interval: Process every Nth frame
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Video has {frame_count} frames at {fps} FPS")
    print(f"Processing every {detection_interval}th frame (detection_interval={detection_interval})")
    
    # Initialize variables
    known_face_encodings = []
    known_face_images = []
    frame_number = 0
    faces_found = 0
    last_progress_time = datetime.now()
    
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Process only every Nth frame to speed things up
            if frame_number % detection_interval != 0:
                frame_number += 1
                continue
            
            # Print progress every 2 seconds
            current_time = datetime.now()
            if (current_time - last_progress_time).total_seconds() >= 2:
                print(f"Processing frame {frame_number}/{frame_count} ({frame_number/frame_count*100:.1f}%)")
                last_progress_time = current_time
            
            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find all face locations
            face_locations = face_recognition.face_locations(rgb_frame)
            
            for face_location in face_locations:
                top, right, bottom, left = face_location
                
                # Skip small faces
                face_width = right - left
                face_height = bottom - top
                if face_width < min_face_size[0] or face_height < min_face_size[1]:
                    continue
                
                # Expand the face region slightly for better results
                top = max(0, top - int(face_height * 0.1))
                bottom = min(frame.shape[0], bottom + int(face_height * 0.1))
                left = max(0, left - int(face_width * 0.1))
                right = min(frame.shape[1], right + int(face_width * 0.1))
                
                # Get face encoding
                face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]
                
                # Check if this face matches any we've seen before
                is_new_face = True
                if known_face_encodings:
                    # Compare with tolerance (lower = more strict)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                    if True in matches:
                        is_new_face = False
                
                if is_new_face:
                    # Save the encoding and image of this new face
                    known_face_encodings.append(face_encoding)
                    
                    # Extract face image
                    face_image = frame[top:bottom, left:right]
                    known_face_images.append(face_image)
                    
                    # Save face image with improved quality
                    faces_found += 1
                    filename = f"{output_dir}/face_{faces_found}.jpg"
                    cv2.imwrite(filename, face_image)
                    print(f"Saved new unique face #{faces_found} to {filename}")
            
            frame_number += 1
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    finally:
        video.release()
        print(f"Processing complete. Found {faces_found} unique faces.")

if __name__ == "__main__":
    print("Starting face extraction...")
    extract_unique_faces("faces.mp4", detection_interval=30)
