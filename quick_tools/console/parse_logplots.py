#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解析 logplot 保存的 .npz 文件，执行其中保存的绘图操作。

功能：
1. 输入文件夹路径
2. 递归读取文件夹及子文件夹中的所有 .npz 文件路径
3. 使用 plotter 中的 parse_action 方法解析这些文件
"""

import os
import sys
import argparse
from pathlib import Path
from shutil import get_terminal_size

sys.path.append('.')
from basic.utils.console.logplot import get_root_plotter
from basic.utils.io import glob_single_files


def parse_logplots(folder_path, plotter_root=None):
    """
    Recursively parse all .npz files under a folder.

    Args:
        folder_path (str): The folder path to search.
        plotter_root (str, optional): The plotter root directory for saving the generated figures.
                                      If None, uses a default path.
    """
    # Check folder existence
    if not os.path.exists(folder_path):
        print(f"Error: folder path does not exist: {folder_path}")
        return

    if not os.path.isdir(folder_path):
        print(f"Error: path is not a folder: {folder_path}")
        return

    # Initialize plotter
    plotter = get_root_plotter(plot_root=plotter_root)
    plotter.set_save_only(False)  # ensure actual plotting, not only saving actions

    # Search for .npz files
    print(f"Searching folder: {folder_path}")
    npz_files = glob_single_files(folder_path, 'npz')

    if not npz_files:
        print("No .npz files found.")
        return

    print(f"Found {len(npz_files)} .npz files")
    max_width = get_terminal_size().columns - 2
    print("-" * max_width)

    success_count = 0
    fail_count = 0

    # Parse files sequentially
    for i, npz_path in enumerate(npz_files, 1):
        print(f"[{i}/{len(npz_files)}] Parsing: {npz_path}")
        try:
            plotter.parse_action(npz_path)
            success_count += 1
            print("  ✓ Success")
        except Exception as e:
            fail_count += 1
            print(f"  ✗ Failed: {str(e)}")

    print("-" * max_width)
    print(f"Finished parsing: {success_count} succeeded, {fail_count} failed")


def main():
    parser = argparse.ArgumentParser(
        description='Recursively parse all .npz files under a folder and execute the stored plotting actions.'
    )
    parser.add_argument(
        'folder_path',
        type=str,
        help='Folder path to search.'
    )
    parser.add_argument(
        '--plotter-root', '-r',
        type=str,
        default=None,
        help='Root directory for plotter output (default: {folder_path}_results/)'
    )

    args = parser.parse_args()

    if args.plotter_root is None:
        if args.folder_path.endswith('/'):
            args.folder_path = args.folder_path[:-1]
        args.plotter_root = f"{args.folder_path}_results"

    parse_logplots(args.folder_path, args.plotter_root)


if __name__ == '__main__':
    main()
