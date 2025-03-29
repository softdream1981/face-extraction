# Face Extraction Tool

A high-performance Python tool for extracting unique faces from video files.

## Features

- **Parallel Processing**: Utilizes multiple CPU cores for faster processing
- **Quality Control**: Filters out blurry or poor-quality face images
- **Deduplication**: Identifies and extracts only unique faces
- **Model Options**: Supports both CPU (HOG) and GPU (CNN) detection models
- **Configurable Parameters**: Fine-tune extraction settings via command-line arguments

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main_optimized.py your_video.mp4
```

### Advanced Options
```bash
python main_optimized.py your_video.mp4 --output-dir faces --interval 30 --resize 0.5 --min-size 50 --quality 0.6 --model hog
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `video` | Path to the video file | (Required) |
| `--output-dir`, `-o` | Directory to save extracted faces | extracted_faces |
| `--interval`, `-i` | Process every Nth frame | 60 |
| `--resize`, `-r` | Resize factor (smaller = faster) | 0.5 |
| `--min-size`, `-s` | Minimum face size to detect | 30 |
| `--tolerance`, `-t` | Face recognition tolerance | 0.6 |
| `--quality`, `-q` | Minimum face quality threshold | 0.5 |
| `--model`, `-m` | Face detection model (hog/cnn) | hog |
| `--processes`, `-p` | Number of processes to use | auto |
| `--chunk-size`, `-c` | Number of frames per chunk | auto |
| `--log-file`, `-l` | Path to log file | None |
| `--debug`, `-d` | Enable debug logging | False |

## Output Structure

The tool creates an organized directory structure:
- `extracted_faces/`: Top-level output directory
  - `final_faces/`: Contains subdirectories for each unique face
  - `final_summary.json`: Summary of extraction results
  - `face_encodings.json`: Face encodings data
  - `chunk_X/`: Temporary processing directories

## Requirements

- Python 3.6+
- OpenCV
- face_recognition
- NumPy

## License

MIT License 