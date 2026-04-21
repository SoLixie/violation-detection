#!/usr/bin/env python3

"""Compatibility wrapper for legacy speed-engine entry points.

This module intentionally delegates to the unified detection pipeline so
future integrations only have one production path to maintain.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_detection import main as unified_main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Legacy speed engine wrapper. Delegates to main_detection.py."
    )
    parser.add_argument("--live", type=int, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--conf", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    forwarded_argv = [sys.argv[0], "--config", "speed"]
    if args.live is not None:
        forwarded_argv.extend(["--live", str(args.live)])
    if args.video:
        forwarded_argv.extend(["--video", args.video])
    if args.model:
        forwarded_argv.extend(["--model", args.model])
    if args.imgsz is not None:
        forwarded_argv.extend(["--imgsz", str(args.imgsz)])
    if args.max_width is not None:
        forwarded_argv.extend(["--max-width", str(args.max_width)])
    if args.output_dir:
        forwarded_argv.extend(["--output-dir", args.output_dir])
    if args.conf is not None:
        forwarded_argv.extend(["--conf", str(args.conf)])

    sys.argv = forwarded_argv
    unified_main()


if __name__ == "__main__":
    main()
