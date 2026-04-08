# Parking Violation Detection System

This system detects illegal parking violations in video footage using computer vision and deep learning.

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   - If you plan to use TensorFlow exports such as `.tflite` or `.pb`, also install TensorFlow in a supported Python environment.

2. **Prepare Model:**
   - Place your trained YOLO model file as `models/best.pt`
   - The model should be trained to detect vehicles (cars, trucks, buses, motorcycles)
   - If you instead use `bestv1_tf/bestv1_float32.tflite` or another TensorFlow export, keep reading the troubleshooting notes below.

3. **Prepare Video:**
   - Place your video file in the `vids/` directory
   - Update the video path in `config/parking_config.json` if needed

4. **Configure Parking Zone:**
   ```bash
   cd parking-violation-detection
   python zone_detection.py
   ```
   - Click 4 points to define the no-parking zone
   - The configuration will be automatically saved

## Usage

Run the parking violation detection:
```bash
cd parking-violation-detection
python detect_parking.py
```

## Speed Violation Detection

The system also includes speed violation detection:

### Setup for Speed Detection

1. **Configure Speed Detection Lines:**
   - Update `config/speed_config.json` with your video path and speed detection parameters
   - Set the Y coordinates for the two speed detection lines (LINE1 and LINE2)
   - Adjust the distance between lines in meters

2. **Run Speed Detection:**
   ```bash
   cd speed-violation-detection
   python detect_vehicle.py
   ```

### Speed Detection Configuration

The speed detection uses `config/speed_config.json`:

- `video_path`: Path to the video file
- `speed_limit_kmph`: Speed limit threshold for violations
- `line1_y`, `line2_y`: Y coordinates of the two speed detection lines
- `distance_meters`: Real-world distance between the two lines
- Output paths for results and violation logs

## Configuration

### Parking Detection (`config/parking_config.json`):
- `video_path`: Path to the video file (relative to project root)
- `zone_polygon`: 4 points defining the no-parking zone [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
- `parking_threshold`: Time in seconds a vehicle must be stationary to be considered illegally parked

### Speed Detection (`config/speed_config.json`):
- `video_path`: Path to the video file
- `speed_limit_kmph`: Speed limit threshold
- `line1_y`, `line2_y`: Y coordinates of speed detection lines
- `distance_meters`: Distance between detection lines in meters

## Files

### Parking Violation Detection:
- `detect_parking.py`: Main detection script
- `zone_detection.py`: Interactive tool to define parking zones
- `tracker.py`: Object tracking using DeepSORT
- `utils.py`: Utility functions for coordinate calculations

### Speed Violation Detection:
- `detect_vehicle.py`: Main speed detection script
- `speed_estimator.py`: Speed calculation logic
- `tracker.py`: Object tracking using DeepSORT
- `utils.py`: Utility functions for coordinate calculations

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics YOLO
- DeepSORT tracker
- NumPy

## Troubleshooting

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
