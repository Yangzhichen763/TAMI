"""
Convert image frame folders into looping MP4 videos (forward + backward) with optional filtering.
"""

import glob

import cv2
import os
import argparse
from typing import List

from basic.utils.io import glob_packed_files, IMG_EXTENSIONS
from basic.utils.console.pbar import PbarContext
from basic.utils.parallelNsync.multiproc import multi_thread_process


def create_video(image_paths: List[str], output_path: str, fps: int = 24):
    """
    Create a forward+backward looping video from a list of images (n + n-1 frames)

    Args:
        image_paths: List of paths to input images
        output_path: Path for the output video
        fps: Frame rate for the output video
    """
    if not image_paths:
        print(f"Warning: No images to process, skipping {output_path}")
        return

    # Read first image to get dimensions
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        print(f"Error: Failed to read first image {image_paths[0]}")
        return

    height, width, _ = first_image.shape
    size = (width, height)

    # Create output directory if it doesn't exist
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize video writer (using MP4V codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    # Write frames in forward order
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Skipping unreadable image {img_path}")
            continue
        out.write(img)

    # Write frames in reverse order (skip first frame to avoid duplicate)
    for img_path in reversed(image_paths[1:]):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Skipping unreadable image {img_path}")
            continue
        out.write(img)

    out.release()
    # print(f"Video saved: {output_path}")


def any_excluded(path: str, exclude_patterns: List[str]) -> bool:
    for exclude_pattern in exclude_patterns:
        # Handle both relative and absolute exclude patterns
        if (os.path.normpath(exclude_pattern) == path or
                os.path.normpath(os.path.join(os.getcwd(), exclude_pattern)) == path):
            return True

        # glob pattern matching
        if glob.fnmatch.fnmatch(path, exclude_pattern):
            return True

    return False


def process_frames(input_path: str, output_path: str, fps: int = 24, exclude_patterns: List[str] = None):
    """
    Process image frames in the input path or its subdirectories

    Args:
        input_path: Path containing frames or subdirectories of frames
        output_path: Base directory for output videos
        fps: Frame rate for output videos
    """
    dir_path = os.path.dirname(input_path)
    files_paths = glob_packed_files(f"{input_path}/**/*", IMG_EXTENSIONS)
    relative_frames_paths = [[os.path.relpath(f, dir_path) for f in fs] for fs in files_paths]

    output_dir = os.path.join(os.path.dirname(dir_path), output_path)

    with PbarContext(total=len(files_paths)) as pbar:
        for file_paths, relative_frames_path in zip(files_paths, relative_frames_paths):
            common_dir = os.path.commonprefix(file_paths)
            common_dir = '/'.join(common_dir.split('/')[:-1])

            # Check if this input path should be excluded
            if any_excluded(common_dir, exclude_patterns):
                pbar.update(1)
                continue

            relative_video_path = os.path.commonprefix(relative_frames_path)
            output_video_path = os.path.join(output_dir, os.path.dirname(relative_video_path) + '.mp4')
            create_video(file_paths, output_video_path, fps)
            print(f"Processed {common_dir} -> {output_video_path}")

            pbar.update(1)

def main():
    # Configure command line argument parser
    parser = argparse.ArgumentParser(description='Convert image frames to forward+backward looping videos')
    parser.add_argument('input_paths', type=str, nargs='+', help='Input directory containing frames or subdirectories')
    parser.add_argument('--output_path', '-o', default='results_video_clips', type=str, help='Output directory for video files')
    parser.add_argument('--fps', type=int, default=24, help='Frame rate for output videos (default: 24)')
    parser.add_argument('--exclude', '-e', type=str, nargs='*', default=[],
                       help='Paths or patterns to exclude (supports multiple)')

    args = parser.parse_args()

    def process_input(input_path):
        input_path = os.path.normpath(input_path)
        if not os.path.exists(input_path):
            print(f"Error: Input path does not exist: {input_path}")
            return

        # Check if this input path should be excluded
        if any_excluded(input_path, args.exclude):
            print(f"Skipping excluded path: {input_path}")
            return

        output_path = os.path.normpath(args.output_path)

        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        process_frames(input_path, output_path, args.fps, exclude_patterns=args.exclude)

    def on_result(input_path, result):
        pass

    multi_thread_process(args.input_paths, process_input, on_result)

if __name__ == "__main__":
    main()