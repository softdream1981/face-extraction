# Face Extraction from Videos

A Python tool to extract unique faces from video files with high performance and quality control.

## Overview

This project provides scripts to detect, extract, and organize unique faces from video files. It's useful for:

- Creating datasets of faces for machine learning
- Cataloging people appearing in videos
- Automated face extraction for media analysis

## Features

### Basic Version (`main.py`)
- Face detection and extraction from video files
- Unique face identification to avoid duplicates
- Simple command-line interface

### Optimized Version (`main_optimized.py`)
- **Parallel Processing**: Utilizes multiple CPU cores for faster processing
- **GPU Support**: Optional CNN-based model when GPU is available
- **Quality Control**: Filters out blurry or poor-quality face images
- **Memory Management**: Chunk-based processing for handling long videos
- **Structured Output**: Organized directory structure with metadata
- **Logging**: Detailed logs with timestamps and progress tracking
- **Configurable Parameters**: Fine-tune the extraction process

## Installation

1. Clone this repository:
```bash
git clone https://github.com/softdream1981/face-extraction.git
cd face-extraction
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Process a video file with default settings:

```bash
python main.py your_video.mp4
```

### Optimized Usage

Process a video with advanced options:

```bash
python main_optimized.py your_video.mp4
```

### Command Line Options

The optimized version supports numerous command-line options:

```
usage: main_optimized.py [-h] [--output-dir OUTPUT_DIR] [--interval INTERVAL]
                        [--resize RESIZE] [--min-size MIN_SIZE]
                        [--tolerance TOLERANCE] [--quality QUALITY]
                        [--model {hog,cnn}] [--processes PROCESSES]
                        [--chunk-size CHUNK_SIZE] [--log-file LOG_FILE]
                        [--debug]
                        video

Extract unique faces from a video file (Improved version)

positional arguments:
  video                 Path to the video file

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory to save extracted faces
  --interval INTERVAL, -i INTERVAL
                        Process every Nth frame (higher = faster)
  --resize RESIZE, -r RESIZE
                        Resize factor (smaller = faster, 0.5 = half size)
  --min-size MIN_SIZE, -s MIN_SIZE
                        Minimum face size to detect (width and height)
  --tolerance TOLERANCE, -t TOLERANCE
                        Face recognition tolerance (lower = more strict, 0.6 recommended)
  --quality QUALITY, -q QUALITY
                        Minimum face quality threshold (0.0-1.0)
  --model {hog,cnn}, -m {hog,cnn}
                        Face detection model (hog for CPU, cnn for GPU)
  --processes PROCESSES, -p PROCESSES
                        Number of processes to use (default: auto)
  --chunk-size CHUNK_SIZE, -c CHUNK_SIZE
                        Number of frames per chunk (default: auto)
  --log-file LOG_FILE, -l LOG_FILE
                        Path to log file (default: no file logging)
  --debug, -d           Enable debug logging
```

## Performance Optimization

Adjust these parameters to balance speed vs. accuracy:

- **For maximum speed**: `--interval 150 --resize 0.3 --quality 0.3`
- **For best quality**: `--interval 30 --resize 0.8 --quality 0.7 --model cnn`
- **Balanced performance**: `--interval 60 --resize 0.5 --quality 0.5`

## Output Structure

The optimized script creates a structured output directory:

```
extracted_faces/
├── chunk_1/            # Processing chunks
├── chunk_2/
├── ...
├── final_faces/        # Final organized faces
│   ├── face_1/
│   │   ├── face_1.jpg  # Extracted face image
│   │   └── metadata.json  # Face metadata
│   ├── face_2/
│   └── ...
├── face_encodings.json  # Face recognition data
└── final_summary.json   # Processing summary
```

## Face Quality Assessment

The quality score (0.0-1.0) is calculated based on:

- **Sharpness**: Measures image clarity (50% weight)
- **Brightness**: Penalizes too dark/bright faces (30% weight)
- **Contrast**: Evaluates color/detail separation (20% weight)

## Requirements

- Python 3.7+
- OpenCV
- face_recognition
- numpy

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
