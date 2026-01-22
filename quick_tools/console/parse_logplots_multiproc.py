import os
import sys
import argparse
from multiprocessing import Pool
from shutil import get_terminal_size

sys.path.append('.')
from basic.utils.console.logplot import get_root_plotter
from basic.utils.io import glob_single_files


def parse_single_file(args):
    """
    Worker function for processing a single .npz file in a separate process.

    Args:
        args: Tuple (npz_path, plotter_root)
    """
    npz_path, plotter_root = args

    try:
        plotter = get_root_plotter(plot_root=plotter_root)
        plotter.set_save_only(False)
        plotter.parse_action(npz_path, any_original_func=True)
        return True, npz_path
    except Exception as e:
        return False, (npz_path, str(e))


def parse_logplots_mp(folder_path, plotter_root=None, num_workers=8):
    """
    Parse logplots using multiprocessing. Each .npz file is executed in a separate process.

    Args:
        folder_path (str): Folder path to search.
        plotter_root (str): Plot output root path.
        num_workers (int): Number of parallel worker processes.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: invalid folder: {folder_path}")
        return

    npz_files = glob_single_files(folder_path, 'npz')
    if not npz_files:
        print("No .npz files found.")
        return

    print(f"Found {len(npz_files)} files, using {num_workers} workers")
    max_width = get_terminal_size().columns - 2
    print("-" * max_width)

    tasks = [(path, plotter_root) for path in npz_files]

    success_count = 0
    fail_count = 0

    with Pool(num_workers) as pool:
        for ok, result in pool.imap_unordered(parse_single_file, tasks):
            if ok:
                print(f"✓ Success: {result}")
                success_count += 1
            else:
                npz_path, err = result
                print(f"✗ Failed: {npz_path}\n    Error: {err}")
                fail_count += 1

    print("-" * max_width)
    print(f"Finished: {success_count} succeeded, {fail_count} failed")


def main():
    parser = argparse.ArgumentParser(
        description='Parse all .npz logplot files using multiprocessing.')
    parser.add_argument('folder_path', type=str)
    parser.add_argument('--plotter-root', '-r', type=str, default=None)
    parser.add_argument('--workers', '-w', type=int, default=os.cpu_count(),
                        help='Number of worker processes')

    args = parser.parse_args()

    if args.plotter_root is None:
        fp = args.folder_path.rstrip('/')
        args.plotter_root = f"{fp}_results"

    parse_logplots_mp(args.folder_path, args.plotter_root, args.workers)


if __name__ == '__main__':
    main()
