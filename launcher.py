#!/usr/bin/env python3
"""
Violation Detection System Launcher
Provides easy access to all system components
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_zone_detection():
    """Run the parking zone detection UI"""
    script_path = Path(__file__).parent / "ui" / "calibrator_ui.py"
    print("Starting calibration UI...")
    print("Choose parking zone mode in the app and save your changes there.")
    subprocess.run([sys.executable, str(script_path)])

def run_line_ui():
    """Run the speed line drawing UI"""
    script_path = Path(__file__).parent / "ui" / "calibrator_ui.py"
    print("Starting calibration UI...")
    print("Choose speed line mode in the app and save your changes there.")
    subprocess.run([sys.executable, str(script_path)])

def run_detection(live=None, video=None, config="speed", model=None, tpu=False, output_dir=None):
    """Run the main detection system"""
    script_path = Path(__file__).parent / "main_detection.py"
    cmd = [sys.executable, str(script_path)]

    if live is not None:
        cmd.extend(["--live", str(live)])
        print(f"Starting live detection with camera {live}...")
    elif video:
        cmd.extend(["--video", video])
        print(f"Starting detection with video: {video}...")
    else:
        print("Starting detection with default configuration...")

    cmd.extend(["--config", config])
    if model:
        cmd.extend(["--model", model])
    if tpu:
        cmd.append("--tpu")
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="Violation Detection System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py zones          # Configure parking zones
  python launcher.py lines          # Configure speed lines
  python launcher.py live           # Live detection with camera 0
  python launcher.py live 1         # Live detection with camera 1
  python launcher.py video myvid.mp4 # Process video file
        """
    )

    parser.add_argument("command", choices=["zones", "lines", "live", "video"],
                       help="Command to run")
    parser.add_argument("argument", nargs="?", help="Additional argument (camera index or video path)")
    parser.add_argument("--config", "-c", choices=["speed", "parking", "both"],
                       default="speed", help="Configuration type (default: speed)")
    parser.add_argument("--model", "-m", help="Path to a .pt or .tflite model")
    parser.add_argument("--tpu", action="store_true", help="Use the Edge TPU delegate for TFLite models")
    parser.add_argument("--output-dir", help="Directory for saved violation clips")

    args = parser.parse_args()

    if args.command == "zones":
        run_zone_detection()
    elif args.command == "lines":
        run_line_ui()
    elif args.command == "live":
        camera_index = int(args.argument) if args.argument else 0
        run_detection(
            live=camera_index,
            config=args.config,
            model=args.model,
            tpu=args.tpu,
            output_dir=args.output_dir
        )
    elif args.command == "video":
        if not args.argument:
            parser.error("Video path required for 'video' command")
        run_detection(
            video=args.argument,
            config=args.config,
            model=args.model,
            tpu=args.tpu,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()
