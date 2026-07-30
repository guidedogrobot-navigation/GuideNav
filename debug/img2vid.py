#!/usr/bin/env python3
"""Render the frames saved by save_data.py into a video.

Point --input at a directory of debug frames (either a single run directory, or
a parent containing nav_run_* subdirectories, in which case one video is made
per run).

Requires ffmpeg on PATH.

Examples:
    # one run directory -> one video
    python debug/img2vid.py --input debug_results/nav_run_20260709_012222

    # every nav_run_* under a parent directory
    python debug/img2vid.py --input debug_results --all
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile


def natural_key(name):
    """Sort key that orders embedded numbers numerically (000002 < 000010)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def has_frames(directory):
    return any(f.lower().endswith((".jpg", ".jpeg", ".png"))
               for f in os.listdir(directory))


def find_run_dirs(input_dir):
    """Return the run directories to render.

    If input_dir holds nav_run_* subdirectories, return those in chronological
    (name) order; otherwise treat input_dir itself as a single run.
    """
    runs = sorted(
        (os.path.join(input_dir, d) for d in os.listdir(input_dir)
         if d.startswith("nav_run_") and os.path.isdir(os.path.join(input_dir, d))),
        key=lambda p: natural_key(os.path.basename(p)),
    )
    return runs if runs else [input_dir]


def make_video(run_dir, output_path, fps):
    frames = sorted(
        (f for f in os.listdir(run_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))),
        key=natural_key,
    )
    if not frames:
        print(f"  No frames in {run_dir}, skipping.")
        return False

    # Use ffmpeg's concat demuxer so we don't depend on a fixed numeric pattern.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listfile:
        list_path = listfile.name
        for f in frames:
            abs_path = os.path.abspath(os.path.join(run_dir, f))
            listfile.write(f"file '{abs_path}'\n")
            listfile.write(f"duration {1.0 / fps}\n")
        # Repeat the last frame so its duration is honored by the demuxer.
        listfile.write(f"file '{os.path.abspath(os.path.join(run_dir, frames[-1]))}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-vsync", "vfr",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        # Ensure even dimensions (required by yuv420p / libx264).
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        output_path,
    ]
    print(f"  {len(frames)} frames -> {output_path}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    os.unlink(list_path)

    if result.returncode != 0:
        print(result.stderr.decode(errors="replace"))
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True,
                    help="run directory of frames, or a parent of nav_run_* directories")
    ap.add_argument("--output-dir", "-o", default=".",
                    help="where to write the video(s) (default: current directory)")
    ap.add_argument("--fps", type=int, default=10, help="output frame rate (default: 10)")
    ap.add_argument("--all", action="store_true",
                    help="render every nav_run_* subdirectory, not just the first")
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        sys.exit(f"Input directory not found: {args.input}")

    run_dirs = find_run_dirs(args.input)
    if not args.all:
        run_dirs = [d for d in run_dirs if has_frames(d)][:1]
        if not run_dirs:
            sys.exit(f"No image frames found under {args.input}")

    os.makedirs(args.output_dir, exist_ok=True)

    for run_dir in run_dirs:
        name = os.path.basename(os.path.normpath(run_dir))
        output_path = os.path.join(args.output_dir, f"{name}.mp4")
        print(f"Processing {name}")
        make_video(run_dir, output_path, args.fps)

    print("Done.")


if __name__ == "__main__":
    main()
