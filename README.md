# Violation Detection System

This system detects parking and speed violations in video footage using computer vision and deep learning. It supports both video file processing and real-time camera detection.

## Features

- **Parking Violation Detection**: Detects vehicles parked in no-parking zones
- **Speed Violation Detection**: Measures vehicle speeds and detects violations
- **Real-time Detection**: Live camera input support
- **Interactive Zone Configuration**: GUI tools to define parking zones and speed measurement lines
- **Violation Storage**: Saves violations to MongoDB with video clips and images

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Detection Zones:**
   ```bash
   python launcher.py zones    # Configure parking zones
   python launcher.py lines    # Configure speed measurement lines
   ```

3. **Run Detection:**
   ```bash
   # Live camera detection
   python launcher.py live

   # Video file detection
   python launcher.py video path/to/video.mp4

   # Direct command (alternative)
   python main_detection.py --live 0
   ```

## Setup

### Model Preparation
- Place your YOLO model file (e.g., `yolov8n.pt`) in the project root
- The model should be trained to detect vehicles (cars, trucks, buses, motorcycles)

### Zone Configuration

#### Parking Zones
```bash
python parking_violation_detection/zone_detection.py
```
- Interactive GUI to draw parking zones
- Define "Zebra Zone" (no-parking area) and "Buffer Zone"
- Configuration saved to `config/parking_config.json`

#### Speed Measurement Lines
```bash
python speed_violation_detection/line_draw_ui.py
```
- Interactive GUI to draw speed detection lines
- Define Entry and Exit lines for speed measurement
- Configuration saved to `config/speed_config.json`

## Usage

### Using the Launcher

The `launcher.py` script provides easy access to all components:

```bash
# Configure zones and lines
python launcher.py zones          # Parking zone setup
python launcher.py lines          # Speed line setup

# Run detection
python launcher.py live           # Live camera (default camera 0)
python launcher.py live 1         # Live camera (camera 1)
python launcher.py video myvid.mp4 # Process video file
```

### Command Line Options

For direct control, use `main_detection.py`:

```bash
python main_detection.py [options]

Options:
  --live CAMERA_INDEX, -l CAMERA_INDEX
                        Use live camera input (e.g., 0 for default camera)
  --video PATH, -v PATH  Use video file input
  --config {speed,parking,both}, -c {speed,parking,both}
                        Configuration to use (default: speed)
```

### Examples

**Live Detection with Camera:**
```bash
python main_detection.py --live 0
```

**Process Video File:**
```bash
python main_detection.py --video videos/traffic.mp4
```

**Parking Detection Only:**
```bash
python main_detection.py --video videos/parking_lot.mp4 --config parking
```

## Configuration Files

### Parking Config (`config/parking_config.json`)
```json
{
  "video_path": "videos/parking.mp4",
  "zebra_zone": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "buffer_zone": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "parking_threshold": 10
}
```

### Speed Config (`config/speed_config.json`)
```json
{
  "video_path": "videos/speed.mp4",
  "speed_limit_kmph": 50,
  "line1": [[x1,y1], [x2,y2]],
  "line2": [[x1,y1], [x2,y2]],
  "distance_meters": 6
}
```

## Output

- **Video Clips**: Saved to `violations_storage/videos/`
- **Images**: Saved to MongoDB GridFS
- **Logs**: Violation events printed to console
- **Database**: Violations stored in MongoDB collection

## Files Structure

```
violation-detection/
├── main_detection.py          # Main unified detection script
├── config/
│   ├── parking_config.json    # Parking zone configuration
│   └── speed_config.json      # Speed detection configuration
├── parking_violation_detection/
│   ├── zone_detection.py      # Interactive parking zone setup
│   ├── detect_parking.py      # Parking detection logic
│   ├── tracker.py            # Object tracking
│   └── utils.py              # Utility functions
├── speed_violation_detection/
│   ├── line_draw_ui.py       # Interactive speed line setup
│   ├── detect_vehicle.py     # Speed detection logic
│   ├── speed_estimator.py    # Speed calculation
│   ├── tracker.py           # Object tracking
│   └── utils.py             # Utility functions
└── violations_storage/       # Output directory
```

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics YOLO
- DeepSORT tracker
- NumPy
- PyMongo (for database storage)

## Troubleshooting

### Camera Issues
- Ensure camera index is correct (usually 0 for built-in camera)
- Check camera permissions on your system
- Try different camera indices if multiple cameras are available

### Video Processing
- Ensure video file exists and is readable
- Check video codec compatibility
- For large videos, processing may take time

### Configuration
- Run zone/line setup tools before detection
- Verify coordinates are within video frame bounds
- Check JSON syntax in config files

### Parking Detection:
- **Model not found**: Ensure `models/best.pt` exists
- **Video not found**: Check `vids/` directory and config file
- **No violations detected**: Adjust parking threshold in config
- **Wrong zone**: Run `zone_detection.py` again to redefine the zone
- **TFLite/TensorFlow model fails to run**: `.tflite` and `.pb` models need `tensorflow` or `tflite-runtime`. On Windows, use a Python version supported by TensorFlow. If you are on Python 3.14, create a new environment with Python 3.10-3.13 and install `tensorflow`.

### Speed Detection:
- **Model not found**: Ensure `models/best.pt` exists
- **Video not found**: Check `vids/` directory and config file
- **No speed calculations**: Check that vehicles are crossing both detection lines
- **Inaccurate speeds**: Adjust the `distance_meters` in config to match real-world distance

Video link: https://www.youtube.com/watch?v=wqctLW0Hb_0
