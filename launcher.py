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
    script_path = Path(__file__).parent / "parking_violation_detection" / "zone_detection.py"
    print("Starting Parking Zone Detection UI...")
    print("Instructions:")
    print("- Draw ZEBRA zone first (no-parking area)")
    print("- Press N to switch to BUFFER zone")
    print("- Press S to save both zones")
    print("- Press ESC to quit")
    subprocess.run([sys.executable, str(script_path)])

def run_line_ui():
    """Run the speed line drawing UI"""
    script_path = Path(__file__).parent / "speed_violation_detection" / "line_draw_ui.py"
    print("Starting Speed Line Drawing UI...")
    print("Instructions:")
    print("- Click and drag to draw lines")
    print("- Press N to confirm each line")
    print("- Press S to save when you have 2 lines")
    print("- Press ESC to quit")
    subprocess.run([sys.executable, str(script_path)])

def run_detection(live=None, video=None, config="speed"):
    """Run the main detection system"""
    cmd = [sys.executable, "-m", "violation_detection.main_detection"]

    if live is not None:
        cmd.extend(["--live", str(live)])
        print(f"Starting live detection with camera {live}...")
    elif video:
        cmd.extend(["--video", video])
        print(f"Starting detection with video: {video}...")
    else:
        print("Starting detection with default configuration...")

    cmd.extend(["--config", config])
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

    args = parser.parse_args()

    if args.command == "zones":
        run_zone_detection()
    elif args.command == "lines":
        run_line_ui()
    elif args.command == "live":
        camera_index = int(args.argument) if args.argument else 0
        run_detection(live=camera_index, config=args.config)
    elif args.command == "video":
        if not args.argument:
            parser.error("Video path required for 'video' command")
        run_detection(video=args.argument, config=args.config)

if __name__ == "__main__":
    main()