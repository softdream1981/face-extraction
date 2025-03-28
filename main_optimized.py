import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import time
import argparse
import logging
import multiprocessing
from functools import partial
import json
from pathlib import Path
import shutil

# Setup logging
def setup_logging(log_file=None, log_level=logging.INFO):
    """Set up logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger()

def calculate_image_quality(image, min_size=(50, 50)):
    """
    Calculate quality metrics for a face image
    Returns a score from 0 (poor) to 1 (excellent)
    """
    # Check if image is too small
    h, w = image.shape[:2]
    if h < min_size[0] or w < min_size[1]:
        return 0.0
    
    # Convert to grayscale for calculations
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Calculate sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Calculate brightness
    brightness = np.mean(gray)
    
    # Calculate contrast
    contrast = np.std(gray)
    
    # Normalize scores
    sharpness_score = min(1.0, sharpness / 500.0)  # Normalize sharpness
    brightness_score = 1.0 - abs((brightness - 128) / 128)  # Penalize too dark/bright
    contrast_score = min(1.0, contrast / 80)  # Normalize contrast
    
    # Combine scores (weights can be adjusted)
    quality_score = (0.5 * sharpness_score + 
                     0.3 * brightness_score + 
                     0.2 * contrast_score)
    
    return quality_score

def process_frame_chunk(chunk_info, video_path, detection_interval, resize_factor, min_face_size, 
                        quality_threshold, model, tolerance, output_dir, encoding_file):
    """Process a chunk of frames from the video"""
    
    start_frame, end_frame, chunk_id = chunk_info
    logger = logging.getLogger()
    
    logger.info(f"Processing chunk {chunk_id}: frames {start_frame} to {end_frame}")
    
    # Create a temporary directory for this chunk
    chunk_dir = os.path.join(output_dir, f"chunk_{chunk_id}")
    os.makedirs(chunk_dir, exist_ok=True)
    
    # Open the video
    video = cv2.VideoCapture(video_path)
    
    # Seek to start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Initialize variables
    known_face_encodings = []
    known_face_metadata = []
    faces_found = 0
    frame_number = start_frame
    
    # Load existing encodings if available
    if os.path.exists(encoding_file):
        try:
            with open(encoding_file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    known_face_encodings.append(np.array(entry['encoding']))
                    known_face_metadata.append(entry['metadata'])
        except Exception as e:
            logger.warning(f"Could not load existing encodings: {e}")
    
    try:
        while frame_number <= end_frame:
            frame_number += 1
            
            # Process only every Nth frame to speed things up
            if (frame_number - 1) % detection_interval != 0:
                ret = video.grab()  # Just grab frame without decoding for faster skipping
                if not ret:
                    break
                continue
            
            # Read frame
            ret, frame = video.read()
            if not ret:
                break
            
            # Resize frame for faster processing
            if resize_factor != 1.0:
                h, w = frame.shape[:2]
                small_frame = cv2.resize(frame, (int(w * resize_factor), int(h * resize_factor)))
            else:
                small_frame = frame
            
            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all face locations
            face_locations = face_recognition.face_locations(rgb_frame, model=model)
            
            # Scale face locations back to original size
            if resize_factor != 1.0:
                face_locations = [(int(top/resize_factor), int(right/resize_factor), 
                                 int(bottom/resize_factor), int(left/resize_factor)) 
                                for top, right, bottom, left in face_locations]
            
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
                
                # Extract face for encoding
                face_image = frame[top:bottom, left:right]
                
                # Check face quality
                quality_score = calculate_image_quality(face_image)
                if quality_score < quality_threshold:
                    logger.debug(f"Skipping low quality face (score: {quality_score:.2f}) at frame {frame_number}")
                    continue
                
                # Create a smaller version just for encoding (faster)
                small_face = cv2.resize(face_image, (150, 150))
                small_face_rgb = cv2.cvtColor(small_face, cv2.COLOR_BGR2RGB)
                
                # Get face encoding
                try:
                    face_encoding = face_recognition.face_encodings(small_face_rgb)[0]
                    
                    # Check if this face matches any we've seen before
                    is_new_face = True
                    if known_face_encodings:
                        # Compare with tolerance (lower = more strict)
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
                        if True in matches:
                            is_new_face = False
                    
                    if is_new_face:
                        # Save the encoding and image of this new face
                        faces_found += 1
                        
                        # Create a timestamp-based subfolder
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        face_dir = os.path.join(chunk_dir, f"face_{faces_found}_{timestamp}")
                        os.makedirs(face_dir, exist_ok=True)
                        
                        # Save face image
                        filename = os.path.join(face_dir, f"face_{faces_found}_frame_{frame_number}.jpg")
                        cv2.imwrite(filename, face_image)
                        
                        # Save metadata
                        metadata = {
                            'face_id': faces_found,
                            'frame': frame_number,
                            'timestamp': timestamp,
                            'location': {'top': top, 'right': right, 'bottom': bottom, 'left': left},
                            'quality_score': quality_score,
                            'chunk_id': chunk_id,
                            'filename': filename
                        }
                        
                        # Save metadata to JSON
                        with open(os.path.join(face_dir, 'metadata.json'), 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        # Add to known faces
                        known_face_encodings.append(face_encoding)
                        known_face_metadata.append(metadata)
                        
                        logger.info(f"Saved new unique face #{faces_found} to {filename} (frame {frame_number}, quality: {quality_score:.2f})")
                        
                        # Save incremental encodings for long videos (every 5 new faces)
                        if faces_found % 5 == 0:
                            chunk_encoding_file = os.path.join(chunk_dir, f"encodings_chunk_{chunk_id}.json")
                            data_to_save = []
                            for i, (enc, meta) in enumerate(zip(known_face_encodings, known_face_metadata)):
                                data_to_save.append({
                                    'encoding': enc.tolist(),
                                    'metadata': meta
                                })
                            with open(chunk_encoding_file, 'w') as f:
                                json.dump(data_to_save, f)
                            logger.debug(f"Saved {len(data_to_save)} face encodings to {chunk_encoding_file}")
                        
                except IndexError:
                    # Sometimes face_encodings fails if the face is too small or unclear
                    pass
    
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id}: {e}")
    finally:
        video.release()
        
    # Save final results for this chunk
    chunk_results = {
        'chunk_id': chunk_id,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'faces_found': faces_found,
        'metadata': known_face_metadata
    }
    
    chunk_result_file = os.path.join(chunk_dir, f"results_chunk_{chunk_id}.json")
    with open(chunk_result_file, 'w') as f:
        json.dump(chunk_results, f, indent=2)
    
    return chunk_results

def merge_results(output_dir):
    """Merge results from multiple chunks"""
    logger = logging.getLogger()
    
    # Find all chunk directories
    chunks = [d for d in os.listdir(output_dir) if d.startswith('chunk_')]
    
    if not chunks:
        logger.warning("No chunk directories found to merge")
        return
    
    logger.info(f"Merging results from {len(chunks)} chunks")
    
    # Create a final directory
    final_dir = os.path.join(output_dir, "final_faces")
    os.makedirs(final_dir, exist_ok=True)
    
    # Load all results
    all_metadata = []
    for chunk in chunks:
        chunk_dir = os.path.join(output_dir, chunk)
        result_files = [f for f in os.listdir(chunk_dir) if f.startswith('results_chunk_')]
        
        for result_file in result_files:
            with open(os.path.join(chunk_dir, result_file), 'r') as f:
                chunk_results = json.load(f)
                all_metadata.extend(chunk_results['metadata'])
    
    # Copy all faces to final directory
    for i, metadata in enumerate(all_metadata, 1):
        src_file = metadata['filename']
        if os.path.exists(src_file):
            # Create a new subfolder
            face_dir = os.path.join(final_dir, f"face_{i}")
            os.makedirs(face_dir, exist_ok=True)
            
            # Copy the face image
            dest_file = os.path.join(face_dir, f"face_{i}.jpg")
            shutil.copy2(src_file, dest_file)
            
            # Update metadata and save it
            metadata['final_id'] = i
            metadata['final_filename'] = dest_file
            with open(os.path.join(face_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Merged face #{i} (from chunk {metadata['chunk_id']})")
    
    # Save final summary
    summary = {
        'total_faces': len(all_metadata),
        'chunks_processed': len(chunks),
        'faces': all_metadata
    }
    
    with open(os.path.join(output_dir, 'final_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Merged {len(all_metadata)} faces into {final_dir}")
    return summary

def extract_unique_faces(video_path, output_dir="extracted_faces", min_face_size=(30, 30), 
                        detection_interval=60, resize_factor=0.5, tolerance=0.6,
                        quality_threshold=0.5, model="hog", num_processes=None,
                        chunk_size=None, log_file=None, log_level=logging.INFO):
    """
    Extract unique faces from a video file with optimizations for speed and quality
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted faces
        min_face_size: Minimum face size to detect (width, height)
        detection_interval: Process every Nth frame
        resize_factor: Resize input frames by this factor (smaller = faster)
        tolerance: Face recognition tolerance (lower = more strict)
        quality_threshold: Minimum quality score threshold (0.0-1.0)
        model: Face detection model ('hog' for CPU, 'cnn' for GPU)
        num_processes: Number of processes to use (None = auto)
        chunk_size: Number of frames per chunk (None = auto)
        log_file: Path to log file (None = no file logging)
        log_level: Logging level (default = INFO)
    """
    # Setup logging
    logger = setup_logging(log_file, log_level)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file to get properties
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        logger.error(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    
    logger.info(f"Video has {frame_count} frames at {fps} FPS")
    logger.info(f"Processing every {detection_interval}th frame (detection_interval={detection_interval})")
    logger.info(f"Using {model} face detection model with resize factor {resize_factor}")
    logger.info(f"Face quality threshold: {quality_threshold}")
    
    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    num_processes = min(num_processes, multiprocessing.cpu_count())
    logger.info(f"Using {num_processes} processes")
    
    # Determine chunk size
    if chunk_size is None:
        # Make sure each process gets at least some frames
        chunk_size = max(detection_interval * 10, frame_count // (num_processes * 2))
    logger.info(f"Processing in chunks of {chunk_size} frames")
    
    # Create chunks
    chunks = []
    for i in range(0, frame_count, chunk_size):
        start_frame = i
        end_frame = min(i + chunk_size - 1, frame_count - 1)
        chunks.append((start_frame, end_frame, len(chunks) + 1))
    
    # Create an encodings file
    encoding_file = os.path.join(output_dir, "face_encodings.json")
    if not os.path.exists(encoding_file):
        with open(encoding_file, 'w') as f:
            json.dump([], f)
    
    # Process chunks in parallel
    start_time = time.time()
    if num_processes > 1:
        # Multiprocessing approach
        process_func = partial(
            process_frame_chunk,
            video_path=video_path,
            detection_interval=detection_interval,
            resize_factor=resize_factor,
            min_face_size=min_face_size,
            quality_threshold=quality_threshold,
            model=model,
            tolerance=tolerance,
            output_dir=output_dir,
            encoding_file=encoding_file
        )
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_func, chunks)
            
    else:
        # Single process approach
        results = []
        for chunk in chunks:
            result = process_frame_chunk(
                chunk,
                video_path=video_path,
                detection_interval=detection_interval,
                resize_factor=resize_factor,
                min_face_size=min_face_size,
                quality_threshold=quality_threshold,
                model=model,
                tolerance=tolerance,
                output_dir=output_dir,
                encoding_file=encoding_file
            )
            results.append(result)
    
    # Merge results
    summary = merge_results(output_dir)
    
    # Calculate timing
    elapsed = time.time() - start_time
    logger.info(f"Processing complete in {elapsed:.1f} seconds.")
    if summary:
        logger.info(f"Found {summary['total_faces']} unique faces.")
        logger.info(f"Average processing time per frame: {elapsed/(frame_count/detection_interval):.3f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract unique faces from a video file (Improved version)')
    parser.add_argument('video', type=str, help='Path to the video file')
    parser.add_argument('--output-dir', '-o', type=str, default='extracted_faces', 
                        help='Directory to save extracted faces')
    parser.add_argument('--interval', '-i', type=int, default=60, 
                        help='Process every Nth frame (higher = faster)')
    parser.add_argument('--resize', '-r', type=float, default=0.5, 
                        help='Resize factor (smaller = faster, 0.5 = half size)')
    parser.add_argument('--min-size', '-s', type=int, default=30, 
                        help='Minimum face size to detect (width and height)')
    parser.add_argument('--tolerance', '-t', type=float, default=0.6, 
                        help='Face recognition tolerance (lower = more strict, 0.6 recommended)')
    parser.add_argument('--quality', '-q', type=float, default=0.5,
                        help='Minimum face quality threshold (0.0-1.0)')
    parser.add_argument('--model', '-m', type=str, default='hog', choices=['hog', 'cnn'],
                        help='Face detection model (hog for CPU, cnn for GPU)')
    parser.add_argument('--processes', '-p', type=int, default=None,
                        help='Number of processes to use (default: auto)')
    parser.add_argument('--chunk-size', '-c', type=int, default=None,
                        help='Number of frames per chunk (default: auto)')
    parser.add_argument('--log-file', '-l', type=str, default=None,
                        help='Path to log file (default: no file logging)')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    print("Starting advanced face extraction...")
    extract_unique_faces(
        args.video,
        output_dir=args.output_dir,
        detection_interval=args.interval,
        resize_factor=args.resize,
        min_face_size=(args.min_size, args.min_size),
        tolerance=args.tolerance,
        quality_threshold=args.quality,
        model=args.model,
        num_processes=args.processes,
        chunk_size=args.chunk_size,
        log_file=args.log_file,
        log_level=log_level
    ) 