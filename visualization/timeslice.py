import cv2
import numpy as np
import os
import math
import sys
import argparse
from glob import glob
from collections import defaultdict

sys.path.append('.')
sys.path.append('..')

try:
    from utils.logger import Logger  # type: ignore

    log = Logger()
except Exception:
    class _FallbackLogger:
        def _print(self, prefix, msg):
            print(f"[{prefix}] {msg}")

        def debug(self, msg):
            self._print('DEBUG', msg)

        def info(self, msg):
            self._print('INFO', msg)

        def success(self, msg):
            self._print('OK', msg)

        def warn(self, msg):
            self._print('WARN', msg)

        def error(self, msg):
            self._print('ERROR', msg)

        def note(self, msg):
            self._print('NOTE', msg)

        def banner(self, title):
            line = '=' * max(10, len(str(title)))
            print(f"\n{line}\n{title}\n{line}\n")

        # compare.py-style helpers (no-op styling in fallback)
        def style_num(self, s):
            return str(s)

        def style_mode(self, s):
            return str(s)

        def style_key(self, s):
            return str(s)

        def style_path(self, s):
            return str(s)

        def set_color_enabled(self, enabled):
            return

        def set_level(self, level):
            return

    log = _FallbackLogger()

try:
    from natsort import natsorted
except Exception:
    natsorted = sorted

try:
    # Prefer project-provided helper if available
    from basic.utils.io import glob_single_files  # type: ignore
except Exception:
    class PathHandler:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        @staticmethod
        def get_vanilla_path(path):
            return path

        @staticmethod
        def get_basename(path):
            return os.path.basename(path)

        @staticmethod
        def remove_extension(path):
            filename, _extension = os.path.splitext(path)
            return filename

        def get_dir_removed_path(self, path):
            return os.path.relpath(path, self.dirname)


IMG_EXTS = ['png', 'jpg', 'jpeg']


def is_hidden_path(path):
    parts = os.path.normpath(path or '').split(os.sep)
    return any(part.startswith('.') for part in parts if part not in ('', '.', '..'))


def filter_hidden(paths):
    return [p for p in paths if not is_hidden_path(p)]


def _parse_exclude_methods(raw):
    if not raw:
        return set()
    return {p.strip() for p in str(raw).replace(',', ' ').split() if p.strip()}


def _apply_exclude(methods, exclude_set):
    if not exclude_set:
        return methods
    return [m for m in methods if m not in exclude_set]


if 'glob_single_files' not in globals():
    def glob_single_files(directory, file_extensions, path_handler=PathHandler.get_vanilla_path):
        if isinstance(file_extensions, str):
            file_extensions = [file_extensions]

        raw_paths = []
        for file_extension in file_extensions:
            pattern = os.path.join(directory, f"**/*.{file_extension}")
            raw_paths += natsorted(glob(pattern, recursive=True))
        raw_paths = [os.path.normpath(path) for path in raw_paths]
        file_paths = [path_handler(path) for path in raw_paths if not is_hidden_path(path)]
        return file_paths


def has_images(directory, exts=None):
    exts = exts or IMG_EXTS
    try:
        return len(filter_hidden(glob_single_files(directory, exts))) > 0
    except Exception:
        return False


def resolve_group_folder(method_root, target_group):
    """Resolve a group folder allowing hyphen/underscore mismatch."""
    tg = (target_group or '').replace('-', '').replace('_', '')
    if not tg:
        return None
    try:
        for d in os.listdir(method_root):
            if d.startswith('.'):
                continue
            full = os.path.join(method_root, d)
            if not os.path.isdir(full):
                continue
            if d.replace('-', '').replace('_', '') == tg:
                return d
    except Exception:
        return None
    return None


def discover_method_path(method_root, group=None, dataset=None, pair=None, structure='auto'):
    """Return the first path that contains images for a method.

    structure choices:
      - auto: try all known layouts in order
      - group-dataset-pair: <method>/<group>/<dataset>/<pair>
      - group-dataset: <method>/<group>/<dataset>
      - dataset-only: <method>/<dataset>
      - flat: <method>/ (images directly under method)
    """
    candidates = []
    resolved_group = resolve_group_folder(method_root, group) if group else None
    group_name = resolved_group or group
    if structure == 'group-dataset-pair':
        if group_name and dataset and pair:
            candidates.append(os.path.join(method_root, group_name, dataset, pair))
    elif structure == 'group-dataset':
        if group_name and dataset:
            # Prefer per-pair folder if present
            if pair:
                candidates.append(os.path.join(method_root, group_name, dataset, pair))
            candidates.append(os.path.join(method_root, group_name, dataset))
    elif structure == 'dataset-only':
        if dataset:
            # Prefer per-pair folder if present
            if pair:
                candidates.append(os.path.join(method_root, dataset, pair))
            candidates.append(os.path.join(method_root, dataset))
    elif structure == 'flat':
        # Prefer per-pair folder if present
        if pair:
            candidates.append(os.path.join(method_root, pair))
        candidates.append(method_root)
    else:  # auto
        if group_name and dataset and pair:
            candidates.append(os.path.join(method_root, group_name, dataset, pair))
        if group_name and dataset:
            candidates.append(os.path.join(method_root, group_name, dataset))
        # Also support <method>/<dataset>/<pair> when group is not used
        if dataset and pair:
            candidates.append(os.path.join(method_root, dataset, pair))
        if dataset:
            candidates.append(os.path.join(method_root, dataset))
        candidates.append(method_root)

    for cand in candidates:
        if cand and os.path.isdir(cand) and has_images(cand):
            return cand
    return None


def discover_shared_folder_methods(root):
    """Handle layout where each image folder contains <method>.png/.jpg files.

    Example:
        root/
          img1/
            method1.png
            method2.png
          img2/
            method1.png
            method2.png

    Returns a mapping of method -> ordered list of files.
    """
    if not os.path.isdir(root):
        return {}
    subdirs = [d for d in os.listdir(root) if not d.startswith('.') and os.path.isdir(os.path.join(root, d))]
    subdirs = sorted(subdirs)
    if not subdirs:
        return {}

    method_names = None
    for sd in subdirs:
        cur = os.path.join(root, sd)
        files = [f for f in os.listdir(cur) if not f.startswith('.') and os.path.isfile(os.path.join(cur, f))]
        files = [f for f in files if os.path.splitext(f)[1].lstrip('.').lower() in IMG_EXTS]
        if not files:
            continue
        names = [os.path.splitext(f)[0] for f in files]
        if method_names is None:
            method_names = set(names)
        else:
            method_names |= set(names)

    if not method_names:
        return {}

    out = {}
    for method in sorted(method_names):
        paths = []
        for sd in subdirs:
            cur = os.path.join(root, sd)
            pattern = os.path.join(cur, f"{method}.*")
            matches = natsorted([p for p in glob(pattern) if os.path.splitext(p)[1].lstrip('.').lower() in IMG_EXTS])
            paths.extend(matches)
        if paths:
            out[method] = filter_hidden(paths)
    return out


def discover_local_inputs(root, methods, group=None, dataset=None, pair=None, structure='auto'):
    """Return mapping of method -> folder (or file list) for local mode."""
    if structure == 'shared':
        return discover_shared_folder_methods(root)

    inputs = {}
    for m in methods:
        method_root = os.path.join(root, m)
        cand = discover_method_path(method_root, group=group, dataset=dataset, pair=pair, structure=structure)
        if cand:
            inputs[m] = cand
    if inputs:
        return inputs

    # fallback: shared-folder layout
    shared = discover_shared_folder_methods(root)
    if shared:
        return shared
    return {}


class EventDispatcher:
    """Simple event dispatcher to register and dispatch handlers."""

    def __init__(self):
        self.handlers = {}

    def register(self, key_code, handler):
        self.handlers.setdefault(key_code, []).append(handler)

    def dispatch(self, key_code, *args, **kwargs):
        handled = False
        for h in self.handlers.get(key_code, []):
            try:
                h(*args, **kwargs)
                handled = True
            except Exception:
                handled = True
        return handled


class UndoManager:
    """Manage undoable actions using a stack of callables."""

    def __init__(self, capacity=200):
        self.undo_stack = []
        self.redo_stack = []
        self.capacity = capacity

    def record(self, undo_fn, redo_fn=None, desc=""):
        if undo_fn is None:
            return
        self.undo_stack.append((undo_fn, redo_fn, desc))
        if len(self.undo_stack) > self.capacity:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return
        undo_fn, redo_fn, _desc = self.undo_stack.pop()
        try:
            undo_fn()
        except Exception:
            pass
        if redo_fn is not None:
            self.redo_stack.append((redo_fn, undo_fn, _desc))

    def redo(self):
        if not self.redo_stack:
            return
        redo_fn, undo_fn, _desc = self.redo_stack.pop()
        try:
            redo_fn()
        except Exception:
            pass
        self.undo_stack.append((undo_fn, redo_fn, _desc))


def print_usage_instructions(pair_info=""):
    """Print short runtime instructions."""
    if pair_info:
        log.note(f"Sequence: {pair_info}")
    log.note(
        "Keys: "
        + f"{log.style_key('n')}/{log.style_key('p')} frame, "
        + f"{log.style_key('m')}/{log.style_key('b')} sequence, "
        + f"{log.style_key('Enter')} switch dataset/group, "
        + f"{log.style_key('Space')} jump to image, "
        + f"{log.style_key('s')} save, "
        + f"{log.style_key('r')} toggle direction, "
        + f"{log.style_key('z')}/{log.style_key('y')} undo/redo, "
        + f"{log.style_key('q')}/{log.style_key('ESC')} quit"
    )


class InteractiveLineExtractor:
    def __init__(
            self,
            input_folder, output_folder,
            cache_images=True, window_independent=True, horizontal=False,
            columns=None, inf_length=False, line_width=5, save_wide_slice=True,
            pair_info="", dataset_name=None, pair_name=None
    ):
        self.window_independent = window_independent
        self.horizontal = horizontal

        self.columns = int(columns) if columns is not None else None
        self.inf_length = inf_length
        self.save_wide_slice = bool(save_wide_slice)
        self.pair_info = pair_info

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.dataset_name = dataset_name
        self.pair_name = pair_name
        self.pending_group = None
        self.pending_dataset = None
        
        # Load images: value can be a directory (str) or an explicit file list (list/tuple)
        self.method_roots = {}
        self.image_files = {}
        for name, src in self.input_folder.items():
            if isinstance(src, (list, tuple)):
                files = filter_hidden(list(src))
            else:
                files = filter_hidden(glob_single_files(src, IMG_EXTS))
            self.image_files[name] = files
            if name not in self.method_roots:
                self.method_roots[name] = self._infer_method_root(name, src, files)
        
        # Use GT as reference if available; otherwise use the first available key
        ref_key = 'GT' if 'GT' in self.image_files else list(self.image_files.keys())[0]
        self.reference_key = ref_key
        self.num_frames = len(self.image_files[ref_key])
        
        if self.num_frames == 0:
            raise ValueError(f"No reference images found (key={ref_key})")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Read the first frame to determine image size
        sample = cv2.imread(self.image_files[ref_key][0])
        if sample is None:
            raise ValueError(f"Failed to read image: {self.image_files[ref_key][0]}")
        
        self.height, self.width = sample.shape[:2]

        self.x_position = self.width // 2
        self.y_position = self.height // 2
        if self.inf_length:
            self.line_length = max(self.height * 2, self.width * 2) * 2
        else:
            self.line_length = self.height // 2
        try:
            self.line_width = max(1, int(line_width))
        except Exception:
            self.line_width = 5
        self.window_name = "Interactive Line Extractor"
        self.timeline_name = "Timeline View"
        self.current_frame = 0
        self.dragging = False
        self._pre_drag_state = None
        self._pre_wheel_state = None

        self._windows_initialized = False
        self._method_windows = set()

        self.dispatcher = EventDispatcher()
        self.undo_manager = UndoManager()
        self.needs_update = True
        self._register_keybindings()

        # Optional .srt-based method sorting for grid display (same behavior as compare.py)
        self._method_srt_stems = {}
        self._all_methods_have_srt = False
        self._refresh_method_grid_sorting()

        # Initialize timeline images
        self.reference_image = cv2.imread(self.image_files[ref_key][0])
        self.timelines = {
            name: np.zeros((self.line_length, self.num_frames, 3), dtype=np.uint8)
            for name, path in self.input_folder.items()
        }

        # Cache images (optional)
        self.cached_images = {}
        if cache_images:
            for name, image_paths in self.image_files.items():
                self.cached_images[name] = []
                for image_path in image_paths:
                    img = cv2.imread(image_path)
                    if img is not None:
                        self.cached_images[name].append(img)
                    else:
                        log.warn(f"Failed to cache image: {image_path}")

    def jump_to_image_by_name(self, name):
        # Match by stem or filename (compare.py behavior)
        files = self.image_files.get(self.reference_key, [])
        target = None
        for idx, path in enumerate(files):
            base = os.path.basename(path)
            stem = base.rsplit('.', 1)[0]
            if name == base or name == stem:
                target = idx
                break
        if target is None:
            try:
                cur = os.path.basename(files[self.current_frame]) if files else 'N/A'
                log.warn(f"Image {log.style_path(name)} not found; staying on {log.style_path(cur)}")
            except Exception:
                log.warn(f"Image {name} not found")
            return False
        self.current_frame = target
        try:
            log.success(f"Jumped to image {log.style_path(os.path.basename(files[self.current_frame]))}")
        except Exception:
            log.success(f"Jumped to image {os.path.basename(files[self.current_frame])}")
        return True

    def request_update(self):
        self.needs_update = True

    # ---- Grid ordering helpers (ported from compare.py) ----
    @staticmethod
    def _is_input_key(key):
        try:
            return str(key).lower() == 'input'
        except Exception:
            return False

    @staticmethod
    def _is_gt_key(key):
        try:
            return str(key).lower() == 'gt'
        except Exception:
            return False

    def _ensure_windows(self):
        if getattr(self, '_windows_initialized', False):
            return
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        if not self.window_independent:
            cv2.namedWindow(self.timeline_name)
        self._windows_initialized = True

    def _ensure_method_windows(self):
        if not self.window_independent:
            return
        desired = set(self.image_files.keys())
        for name in natsorted(list(desired)):
            win = f"{self.timeline_name} {name}"
            if win in self._method_windows:
                continue
            try:
                cv2.namedWindow(win)
            except Exception:
                pass
            self._method_windows.add(win)

        # Close windows for methods that no longer exist in this pair
        for win in list(self._method_windows):
            method_name = win.replace(f"{self.timeline_name} ", "", 1)
            if method_name not in desired:
                try:
                    cv2.destroyWindow(win)
                except Exception:
                    pass
                self._method_windows.discard(win)

    def reload_pair(self, input_folder, cache_images=True, dataset_name=None, pair_name=None, pair_info=""):
        """Reload data for a new pair/sequence without creating new windows."""
        self.input_folder = input_folder
        self.dataset_name = dataset_name
        self.pair_name = pair_name
        self.pair_info = pair_info or ""

        # Load images: value can be a directory (str) or an explicit file list (list/tuple)
        self.method_roots = {}
        self.image_files = {}
        for name, src in self.input_folder.items():
            if isinstance(src, (list, tuple)):
                files = filter_hidden(list(src))
            else:
                files = filter_hidden(glob_single_files(src, IMG_EXTS))
            self.image_files[name] = files
            self.method_roots[name] = self._infer_method_root(name, src, files)

        ref_key = 'GT' if 'GT' in self.image_files else list(self.image_files.keys())[0]
        self.reference_key = ref_key
        self.num_frames = len(self.image_files[ref_key])
        if self.num_frames == 0:
            raise ValueError(f"No reference images found (key={ref_key})")

        sample = cv2.imread(self.image_files[ref_key][0])
        if sample is None:
            raise ValueError(f"Failed to read image: {self.image_files[ref_key][0]}")

        old_w, old_h = int(self.width), int(self.height)
        self.height, self.width = sample.shape[:2]

        self.current_frame = max(0, min(int(self.current_frame), self.num_frames - 1))
        if old_w != int(self.width) or old_h != int(self.height):
            self.x_position = int(self.width) // 2
            self.y_position = int(self.height) // 2
        else:
            self.x_position = max(0, min(int(self.x_position), int(self.width) - 1))
            self.y_position = max(0, min(int(self.y_position), int(self.height) - 1))

        if self.inf_length:
            self.line_length = max(int(self.height) * 2, int(self.width) * 2) * 2
        else:
            max_len = max(2, int(self.width) if self.horizontal else int(self.height))
            self.line_length = max(2, min(int(self.line_length), max_len))

        self._refresh_method_grid_sorting()

        self.reference_image = cv2.imread(self.image_files[ref_key][0])
        self.timelines = {
            name: np.zeros((self.line_length, self.num_frames, 3), dtype=np.uint8)
            for name, _path in self.input_folder.items()
        }

        self.cached_images = {}
        if cache_images:
            for name, image_paths in self.image_files.items():
                self.cached_images[name] = []
                for image_path in image_paths:
                    img = cv2.imread(image_path)
                    if img is not None:
                        self.cached_images[name].append(img)
                    else:
                        log.warn(f"Failed to cache image: {image_path}")

        self._ensure_windows()
        self._ensure_method_windows()
        if self.pair_info:
            try:
                log.note(f"Sequence: {self.pair_info}")
            except Exception:
                pass
        self.request_update()

    def _infer_method_root(self, method_key, src, files):
        """Best-effort method root inference for .srt-based sorting."""
        # 1) If src is a path, try to locate the segment equal to method name
        if isinstance(src, str) and src:
            try:
                norm = os.path.normpath(src)
                parts = norm.split(os.sep)
                for i in range(len(parts) - 1, -1, -1):
                    if parts[i] == method_key:
                        cand = os.sep.join(parts[:i + 1])
                        if os.path.isdir(cand):
                            return cand
            except Exception:
                pass
            try:
                if os.path.isdir(src):
                    # common case: .../<method>/<dataset>/<pair>
                    cur = os.path.normpath(src)
                    for _ in range(6):
                        base = os.path.basename(cur)
                        if base == method_key:
                            return cur
                        parent = os.path.dirname(cur)
                        if not parent or parent == cur:
                            break
                        cur = parent
            except Exception:
                pass
        # 2) If we have file list, use their common path
        if files:
            try:
                common = os.path.commonpath(files)
                # walk up a bit to find directory named method_key
                cur = os.path.normpath(common)
                for _ in range(6):
                    if os.path.basename(cur) == method_key and os.path.isdir(cur):
                        return cur
                    parent = os.path.dirname(cur)
                    if not parent or parent == cur:
                        break
                    cur = parent
            except Exception:
                pass
        return None

    def _refresh_method_grid_sorting(self):
        """If every method folder contains a .srt file, use its stem for grid sorting."""
        self._method_srt_stems = {}

        keys = list(self.input_folder.keys()) if isinstance(self.input_folder, dict) else []
        methods = [k for k in keys if (not self._is_input_key(k)) and (not self._is_gt_key(k))]
        if not methods:
            self._all_methods_have_srt = False
            return

        def _list_srt_files(folder):
            try:
                files = [
                    f for f in os.listdir(folder)
                    if (not f.startswith('.')) and f.lower().endswith('.srt') and os.path.isfile(os.path.join(folder, f))
                ]
                return natsorted(files)
            except Exception:
                return []

        def _method_dir_from_src(method_key, src_path):
            if not isinstance(src_path, str) or not src_path:
                return None
            try:
                norm = os.path.normpath(src_path)
                parts = norm.split(os.sep)
                idx = None
                for i in range(len(parts) - 1, -1, -1):
                    if parts[i] == method_key:
                        idx = i
                        break
                if idx is not None:
                    cand = os.sep.join(parts[:idx + 1])
                    if os.path.isdir(cand):
                        return cand
            except Exception:
                return None
            return None

        all_have = True
        for k in methods:
            method_dir = None

            # 1) Try recorded method root
            try:
                mr = (self.method_roots or {}).get(k)
                if isinstance(mr, str) and os.path.isdir(mr):
                    method_dir = mr
            except Exception:
                method_dir = None

            # 2) Try deriving from the input folder path
            if method_dir is None:
                src = self.input_folder.get(k)
                method_dir = _method_dir_from_src(k, src)

            # 3) Fallback: walk up from src to find a directory containing .srt
            if method_dir is None:
                src = self.input_folder.get(k)
                if isinstance(src, str) and os.path.isdir(src):
                    cur = os.path.normpath(src)
                    for _ in range(6):
                        if os.path.isdir(cur) and _list_srt_files(cur):
                            method_dir = cur
                            break
                        parent = os.path.dirname(cur)
                        if not parent or parent == cur:
                            break
                        cur = parent

            if method_dir is None or (not os.path.isdir(method_dir)):
                all_have = False
                break

            srt_files = _list_srt_files(method_dir)
            if not srt_files:
                all_have = False
                break

            stem = os.path.splitext(srt_files[0])[0]
            self._method_srt_stems[k] = stem

        if all_have:
            try:
                log.info(
                    "Using .srt-based sorting for grid display of methods: "
                    + ", ".join(f"{log.style_path(k)}->{v}" for k, v in self._method_srt_stems.items())
                )
            except Exception:
                pass
        else:
            missing = [k for k in methods if k not in self._method_srt_stems]
            try:
                log.warn(
                    "Not all methods have .srt files. Methods missing .srt files: "
                    + ", ".join(log.style_path(k) for k in missing)
                )
            except Exception:
                pass
        self._all_methods_have_srt = bool(all_have and len(self._method_srt_stems) == len(methods))

    def _ordered_method_keys_for_grid(self):
        """Order keys for grid tiles: Input first, GT last, optional .srt stem sorting."""
        keys = list(self.image_files.keys())
        if not keys:
            return []

        input_keys = [k for k in keys if self._is_input_key(k)]
        gt_keys = [k for k in keys if self._is_gt_key(k)]
        middle = [k for k in keys if (k not in input_keys) and (k not in gt_keys)]

        if self._all_methods_have_srt and middle:
            try:
                if all(k in self._method_srt_stems for k in middle):
                    middle = sorted(middle, key=lambda k: (self._method_srt_stems.get(k, ''), str(k)))
            except Exception:
                pass

        return input_keys + middle + gt_keys

    def _snapshot_state(self):
        return (
            int(self.current_frame),
            int(self.x_position),
            int(self.y_position),
            int(self.line_length),
            bool(self.horizontal),
        )

    def _restore_state(self, snapshot):
        (self.current_frame, self.x_position, self.y_position, self.line_length, self.horizontal) = snapshot
        self.current_frame = max(0, min(self.num_frames - 1, int(self.current_frame)))
        self.x_position = max(0, min(self.width - 1, int(self.x_position)))
        self.y_position = max(0, min(self.height - 1, int(self.y_position)))
        self.line_length = max(2, int(self.line_length))
        self.horizontal = bool(self.horizontal)
        self.request_update()

    def _record_state_change(self, before, after, desc=""):
        if before == after:
            return
        self.undo_manager.record(
            lambda b=before: self._restore_state(b),
            lambda a=after: self._restore_state(a),
            desc,
        )

    def _register_keybindings(self):
        self.dispatcher.register(ord('n'), self._cmd_next_frame)
        self.dispatcher.register(ord('p'), self._cmd_prev_frame)
        self.dispatcher.register(ord('r'), self._cmd_toggle_orientation)
        self.dispatcher.register(ord('s'), self._cmd_save)
        self.dispatcher.register(ord('z'), self._cmd_undo)
        self.dispatcher.register(ord('y'), self._cmd_redo)

    def _cmd_next_frame(self):
        before = self._snapshot_state()
        self.current_frame = min(self.current_frame + 1, self.num_frames - 1)
        after = self._snapshot_state()
        self._record_state_change(before, after, "next frame")
        self.request_update()

    def _cmd_prev_frame(self):
        before = self._snapshot_state()
        self.current_frame = max(self.current_frame - 1, 0)
        after = self._snapshot_state()
        self._record_state_change(before, after, "prev frame")
        self.request_update()

    def _cmd_toggle_orientation(self):
        before = self._snapshot_state()
        self.horizontal = not self.horizontal
        after = self._snapshot_state()
        self._record_state_change(before, after, "toggle orientation")
        self.request_update()

    def _cmd_undo(self):
        self.undo_manager.undo()
        self.request_update()

    def _cmd_redo(self):
        self.undo_manager.redo()
        self.request_update()

    def _cmd_save(self):
        # Preserve old behavior but centralize behind keybinding.
        self._save_current_outputs()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self._pre_drag_state = self._snapshot_state()
            self.x_position = x
            self.y_position = y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.x_position = x
                self.y_position = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.x_position = x
            self.y_position = y
            if self._pre_drag_state is not None:
                after = self._snapshot_state()
                self._record_state_change(self._pre_drag_state, after, "drag line")
                self._pre_drag_state = None
        elif event == cv2.EVENT_MOUSEWHEEL:  # Mouse wheel adjusts line length
            # Allow wheel to switch from "infinite" to adjustable length.
            if self._pre_wheel_state is None:
                self._pre_wheel_state = self._snapshot_state()

            max_len = int(self.width) if self.horizontal else int(self.height)
            max_len = max(2, max_len)

            if self.inf_length:
                self.inf_length = False
                self.line_length = min(int(self.line_length), max_len)

            step = 10
            if flags > 0:  # Scroll up
                self.line_length = min(int(self.line_length) + step, max_len)
            else:  # Scroll down
                self.line_length = max(int(self.line_length) - step, 2)

            after = self._snapshot_state()
            self._record_state_change(self._pre_wheel_state, after, "adjust length")
            self._pre_wheel_state = None

        self.x_position = max(0, min(self.width - 1, self.x_position))
        self.y_position = max(0, min(self.height - 1, self.y_position))
        self.request_update()

    def _save_current_outputs(self):
        def _make_thin_timeline(timeline_img: np.ndarray) -> np.ndarray:
            """Convert a wide timeline (striped by self.line_width) into a 1px-per-frame timeline."""
            if timeline_img is None:
                return None
            if int(self.line_width) <= 1:
                return timeline_img

            lw = int(self.line_width)
            pick = lw // 2

            # horizontal: stacked stripes along height
            if self.horizontal:
                # shape: (num_frames*lw, width, 3) -> (num_frames, width, 3)
                h = int(timeline_img.shape[0])
                num_frames = max(1, h // lw)
                rows = [min((t * lw) + pick, h - 1) for t in range(num_frames)]
                return timeline_img[rows, :, :]

            # vertical: striped along width
            # shape: (height, num_frames*lw, 3) -> (height, num_frames, 3)
            w = int(timeline_img.shape[1])
            num_frames = max(1, w // lw)
            cols = [min((t * lw) + pick, w - 1) for t in range(num_frames)]
            return timeline_img[:, cols, :]

        # Current frame name (used to create an output subfolder)
        ref_key = 'GT' if 'GT' in self.image_files else list(self.image_files.keys())[0]
        current_frame_path = self.image_files[ref_key][self.current_frame]
        current_frame_name = os.path.splitext(os.path.basename(current_frame_path))[0]

        dataset_name = self.dataset_name
        pair_name = self.pair_name
        if not dataset_name or not pair_name:
            # Fallback: infer from input_folder (reliable only when values are paths)
            first_src = list(self.input_folder.values())[0]
            if isinstance(first_src, str):
                pair_name = os.path.basename(first_src.rstrip('/\\'))
                dataset_name = os.path.basename(os.path.dirname(first_src).rstrip('/\\'))
            else:
                dataset_name = dataset_name or 'dataset'
                pair_name = pair_name or 'pair'

        # Build output directory: output_folder/dataset/pair/frame_name/
        save_dir = os.path.join(self.output_folder, dataset_name, pair_name, current_frame_name)
        os.makedirs(save_dir, exist_ok=True)

        # Slice line parameters
        x = self.x_position
        y = self.y_position
        l = self.line_length
        if self.horizontal:
            x_start = max(0, x - l // 2)
            x_end = min(self.width, x + l // 2)
            y_start = y
            y_end = y
        else:
            x_start = x
            x_end = x
            y_start = max(0, y - l // 2)
            y_end = min(self.height, y + l // 2)

        line_color = (0, 0, 255)

        for name, timeline in self.timelines.items():
            if self.cached_images and name in self.cached_images and len(self.cached_images[name]) > self.current_frame:
                frame = self.cached_images[name][self.current_frame].copy()
            else:
                frame = cv2.imread(self.image_files[name][self.current_frame])
                if frame is None:
                    log.error(f"Failed to read image: {self.image_files[name][self.current_frame]}")
                    continue

            image_raw_path = os.path.join(save_dir, f"{name}.png")
            cv2.imwrite(image_raw_path, frame)

            frame_with_line = frame.copy()
            cv2.line(frame_with_line, (x_start, y_start), (x_end, y_end), line_color, 2)

            image_output_path = os.path.join(save_dir, f"{name}_line.png")
            cv2.imwrite(image_output_path, frame_with_line)

            # Save thin slice (1px-per-frame) as the default output
            thin_timeline = _make_thin_timeline(timeline)
            slice_output_path = os.path.join(save_dir, f"{name}_slice.png")
            if thin_timeline is not None:
                cv2.imwrite(slice_output_path, thin_timeline)

            # Optionally save wide slice (line_width px-per-frame)
            if bool(self.save_wide_slice) and int(self.line_width) > 1:
                slice_output_path_wide = os.path.join(save_dir, f"{name}_slice_w{int(self.line_width)}.png")
                cv2.imwrite(slice_output_path_wide, timeline)

        log.success(f"Saved outputs to: {save_dir}")
        print_usage_instructions(self.pair_info)

    def update_display(self):
        gap = 2
        line_color = (0, 0, 255)
        text_color = (0, 255, 0)
        
        # Reference frame selection: GT first, otherwise first available
        ref_key = 'GT' if 'GT' in self.image_files else list(self.image_files.keys())[0]
        
        if self.cached_images and ref_key in self.cached_images and len(self.cached_images[ref_key]) > self.current_frame:
            frame = self.cached_images[ref_key][self.current_frame]
        else:
            frame = cv2.imread(self.image_files[ref_key][self.current_frame])
        
        frame_with_line = frame.copy()

        x = self.x_position
        y = self.y_position
        l = self.line_length
        if self.horizontal:
            x_start = max(0, x - l // 2)
            x_end = min(self.width, x + l // 2)
            y_start = y
            y_end = y
        else:
            x_start = x
            x_end = x
            y_start = max(0, y - l // 2)
            y_end = min(self.height, y + l // 2)

        # Draw the slice line
        cv2.line(frame_with_line, (x_start, y_start), (x_end, y_end), line_color, 2)
        # Draw slice info
        frame_with_text = frame_with_line.copy()
        cv2.putText(frame_with_text, f"Line Length: {l}px | Slice Width: {int(self.line_width)}px",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Update timelines
        height = self.num_frames * self.line_width if self.horizontal else max(1, y_end - y_start)
        width = max(1, x_end - x_start) if self.horizontal else self.num_frames * self.line_width
        self.timelines = {
            name: np.zeros((height, width, 3), dtype=np.uint8)
            for name, path in self.input_folder.items()
        }
        for name, timeline in self.timelines.items():
            for t in range(self.num_frames):
                if name not in self.image_files or t >= len(self.image_files[name]):
                    continue
                try:
                    if self.cached_images and name in self.cached_images and len(self.cached_images[name]) > t:
                        frame = self.cached_images[name][t]
                    else:
                        frame = cv2.imread(self.image_files[name][t])
                except Exception as e:
                    log.warn(f"Failed to load image {self.image_files[name][t]}: {e}")
                    continue

                if frame is None:
                    continue

                if self.horizontal:
                    column = frame[y, x_start:x_end, :]
                    timeline[t * self.line_width:(t + 1) * self.line_width, :, :] = (
                        np.tile(column[np.newaxis, :, :], (self.line_width, 1, 1)))
                else:
                    column = frame[y_start:y_end, x, :]
                    timeline[:, t * self.line_width:(t + 1) * self.line_width, :] = (
                        np.tile(column[:, np.newaxis, :], (1, self.line_width, 1)))

        # Display the current frame and timelines
        self.reference_image = frame_with_line
        cv2.imshow(self.window_name, frame_with_text)
        method_keys = self._ordered_method_keys_for_grid()
        if self.window_independent:
            for name in method_keys:
                timeline = self.timelines.get(name)
                if timeline is None:
                    continue
                cv2.imshow(f"{self.timeline_name} {name}", timeline)
        else:
            # Compose timelines into a grid
            num_methods = len(self.timelines)
            if self.columns is None:
                num_cols = (num_methods + 1) // 2 if num_methods >= 9 else max(num_methods, 1)
            else:
                num_cols = max(int(self.columns), 1)
            num_rows = max(int(math.ceil(len(self.timelines) / num_cols)), 1)
            timeline = np.zeros((height * num_rows + gap * (num_rows - 1), width * num_cols + gap * (num_cols - 1), 3),
                                dtype=np.uint8)
            ordered_items = [(k, self.timelines.get(k)) for k in method_keys if k in self.timelines]
            for i, (name, timeline_img) in enumerate(ordered_items):
                if timeline_img is None:
                    continue
                row = i // num_cols
                col = i % num_cols
                row_gap = row * gap
                col_gap = col * gap
                timeline[row * height + row_gap:(row + 1) * height + row_gap, col * width + col_gap:(
                                                                                                                col + 1) * width + col_gap, :] = timeline_img

                # Top-left label
                cv2.putText(timeline, name, (5 + col * width + col_gap, 15 + row * height + row_gap),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            cv2.imshow(self.timeline_name, timeline)

    def run(self, on_next_pair=None, on_prev_pair=None, on_switch_dataset=None):
        log.banner("Interactive Timeline Slice Extractor")
        log.info("Mouse: left-drag moves the slice line; mouse wheel adjusts length (if not infinite).")
        log.info(
            "Keys: "
            + f"{log.style_key('n')}/{log.style_key('p')} next/prev frame, "
            + f"{log.style_key('m')}/{log.style_key('b')} next/prev sequence, "
            + f"{log.style_key('r')} toggle {log.style_mode('horizontal')} / {log.style_mode('vertical')} line, "
            + f"{log.style_key('s')} save outputs, "
            + f"{log.style_key('z')} undo, "
            + f"{log.style_key('y')} redo, "
            + f"{log.style_key('Enter')} switch dataset/group, "
            + f"{log.style_key('Space')} jump to image, "
            + f"{log.style_key('q')}/{log.style_key('ESC')} quit"
        )

        # Create windows once and keep them open across pair switches
        self._ensure_windows()
        self._ensure_method_windows()

        # Print short per-sequence hint
        if self.pair_info:
            try:
                log.note(f"Sequence: {self.pair_info}")
            except Exception:
                pass

        self.request_update()

        while True:
            if self.needs_update:
                self.update_display()
                self.needs_update = False

            key = cv2.waitKeyEx(10)
            key = (key & 0xFF) if key >= 0 else key

            if key in (255, -1):
                continue

            # sequence navigation / quit
            if key == ord('m'):
                if callable(on_next_pair):
                    try:
                        on_next_pair()
                        self._ensure_method_windows()
                        self.request_update()
                    except Exception as e:
                        log.error(f"Failed to switch to next sequence: {e}")
                else:
                    return 'next_pair'
                continue

            if key == ord('b'):
                if callable(on_prev_pair):
                    try:
                        on_prev_pair()
                        self._ensure_method_windows()
                        self.request_update()
                    except Exception as e:
                        log.error(f"Failed to switch to previous sequence: {e}")
                else:
                    return 'prev_pair'
                continue

            if key == 27 or key == ord('q'):
                return 'quit'

            # Enter: prompt to switch dataset (optionally group/dataset)
            if key in (13, 10):
                try:
                    text = input("Enter dataset (or group/dataset): ").strip()
                except Exception:
                    text = ""
                if text:
                    if '/' in text:
                        ng, nd = text.split('/', 1)
                    else:
                        ng, nd = (""), text
                    self.pending_group = (ng or None)
                    self.pending_dataset = (nd or None)
                    if not self.pending_dataset:
                        log.info("Dataset switch cancelled (empty input)")
                        continue
                    if callable(on_switch_dataset):
                        try:
                            on_switch_dataset(self.pending_group, self.pending_dataset)
                            self._ensure_method_windows()
                            self.request_update()
                        except Exception as e:
                            log.error(f"Failed to switch dataset: {e}")
                        continue
                    return 'switch_dataset'
                log.info("Dataset switch cancelled (empty input)")
                continue

            # Space: prompt to jump to image by name
            if key == 32:
                try:
                    text = input("Enter image name to jump (stem or filename): ").strip()
                except Exception:
                    text = ""
                if text:
                    if self.jump_to_image_by_name(text):
                        self.request_update()
                else:
                    log.info("Image jump cancelled (empty input)")
                continue

            # dispatch other keys
            handled = self.dispatcher.dispatch(key)
            if handled:
                self.request_update()

        return 'quit'


if __name__ == "__main__":
    log.banner("LLIE Results - Timeline Tool")

    parser = argparse.ArgumentParser(
        description='Timeline slice extractor for comparing LLIE method results across time.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # --- Data source & paths ---
    g_data = parser.add_argument_group(
        title='Data Source & Paths',
        description='Root directory and output location for timeline slices.'
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.join(script_dir, 'examples')
    default_output = os.path.join(script_dir, 'timeline_slice')

    g_data.add_argument('--root', '-r', default=default_root, type=str,
                        help='Root directory containing method folders (default: <this_dir>/examples)')
    g_data.add_argument('--output', '-o', default=default_output, type=str,
                        help='Output directory for saved slices (default: <this_dir>/timeline_slice)')
    
    # --- Dataset selection ---
    g_dataset = parser.add_argument_group(
        title='Dataset Selection',
        description='Specify dataset and pair/sequence to process.'
    )
    g_dataset.add_argument('--group', '-g', default=None, type=str,
                           help='Optional dataset group folder under each method (e.g., SDSD-indoor+). Hyphens/underscores auto-matched.')
    g_dataset.add_argument('--dataset', '-ds', default='SDSD-indoor', type=str,
                           help='Dataset name (default: SDSD-indoor).')
    g_dataset.add_argument('--pair', '-p', default=None, type=str,
                           help='Pair/sequence name (default: first available). Example: pair21')

    g_dataset.add_argument('--structure', default='auto',
                           choices=['auto', 'group-dataset-pair', 'group-dataset', 'dataset-only', 'flat', 'shared'],
                           help='Folder structure layout: auto (default), group-dataset-pair, group-dataset, dataset-only, flat, or shared.')

    g_dataset.add_argument('--exclude', '-x', default=None, type=str,
                           help='Comma/space separated method names to exclude.')

    g_dataset.add_argument('--methods', default=None, type=str,
                           help='Comma/space separated method names to include (overrides directory scan / methods.txt).')

    g_dataset.add_argument('--methods-file', default=os.path.join(script_dir, 'methods.txt'), type=str,
                           help='Path to methods.txt (default: <this_dir>/methods.txt). Used if --methods not set.')
    
    # --- Extraction settings ---
    g_extract = parser.add_argument_group(
        title='Extraction Settings',
        description='Configure timeline slice extraction parameters.'
    )
    g_extract.add_argument('--columns', '-c', default=None, type=lambda x: int(x) if x else None,
                           help='Number of columns for grid layout (default: auto).')
    g_extract.add_argument('--inf-length', '-il', action='store_true', default=True,
                           help='Use infinite line length (default: True)')
    g_extract.add_argument('--no-inf-length', dest='inf_length', action='store_false',
                           help='Disable infinite line length')
    g_extract.add_argument('--horizontal', action='store_true', default=False,
                           help='Start with horizontal slice direction (default: vertical)')
    g_extract.add_argument('--window-independent', action='store_true', default=False,
                           help='Show timeline windows independently (default: grid layout)')

    g_extract.add_argument('--no-cache', dest='cache_images', action='store_false', default=True,
                           help='Disable image caching to reduce RAM usage')

    g_extract.add_argument('--line-width', '-lw', default=5, type=int,
                           help='Timeline slice width in pixels (default: 5).')

    g_extract.add_argument('--save-wide-slice', dest='save_wide_slice', action='store_true', default=True,
                           help='Also save a wide timeline slice image using --line-width (default: True).')
    g_extract.add_argument('--no-save-wide-slice', dest='save_wide_slice', action='store_false',
                           help='Only save the thin (1px-per-frame) timeline slice image.')

    # --- Logging ---
    g_log = parser.add_argument_group(
        title='Logging',
        description='Control log output (color on/off and verbosity level).'
    )
    g_log.add_argument('--no-color', action='store_true',
                       help='Disable ANSI colored logs (use plain text).')
    g_log.add_argument('--log-level', default='info', choices=['debug', 'info', 'warn', 'error'],
                       help='Logging level: debug|info|warn|error (default: info).')
    
    args = parser.parse_args()

    # configure logger (best-effort)
    try:
        log.set_color_enabled(not bool(getattr(args, 'no_color', False)))
    except Exception:
        pass
    try:
        log.set_level(getattr(args, 'log_level', 'info'))
    except Exception:
        pass

    root_dir = os.path.abspath(args.root)
    output_dir = os.path.abspath(args.output)
    dataset = args.dataset
    group = args.group
    structure = args.structure
    exclude_methods = _parse_exclude_methods(args.exclude)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # == Discover available methods and pairs ==
    if not os.path.exists(root_dir):
        log.error(f"Root directory does not exist: {root_dir}")
        sys.exit(1)

    # methods priority: --methods > --methods-file > scan directories
    all_methods = []
    if args.methods:
        all_methods = [m.strip() for m in str(args.methods).replace(',', ' ').split() if m.strip()]
    elif args.methods_file and os.path.exists(args.methods_file):
        try:
            with open(args.methods_file, 'r', encoding='utf-8') as f:
                all_methods = [line.strip() for line in f.readlines() if line.strip()]
        except Exception:
            all_methods = []
    else:
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                all_methods.append(item)

    all_methods = natsorted(sorted(set(all_methods)))
    all_methods = _apply_exclude(all_methods, exclude_methods)
    log.info(f"Discovered {len(all_methods)} methods: {all_methods}")

    if not all_methods:
        log.error(f"No method folders found under: {root_dir}")
        sys.exit(1)

    def _list_pairs_under_dataset(dataset_root):
        try:
            pairs = [d for d in os.listdir(dataset_root)
                     if not d.startswith('.') and os.path.isdir(os.path.join(dataset_root, d))]
            return natsorted(pairs)
        except Exception:
            return []

    def _compute_pairs_for_dataset(cur_group, cur_dataset, cur_structure, pair_hint=None):
        """Compute available pairs for given dataset/group/structure (same logic as initial scan)."""
        if cur_structure in ('group-dataset', 'dataset-only', 'flat', 'shared'):
            # If the dataset folder contains per-pair subfolders with images, list them.
            # Otherwise treat it as a single-sequence dataset.
            def _list_pair_like_subdirs(dataset_root):
                try:
                    subs = [
                        d for d in os.listdir(dataset_root)
                        if (not d.startswith('.')) and os.path.isdir(os.path.join(dataset_root, d))
                    ]
                except Exception:
                    return []
                out = []
                for d in natsorted(subs):
                    p = os.path.join(dataset_root, d)
                    if has_images(p):
                        out.append(d)
                return out

            gt_root = os.path.join(root_dir, 'GT')
            gt_group = resolve_group_folder(gt_root, cur_group) if cur_group else None
            gt_group_name = gt_group or cur_group
            gt_dataset_path = os.path.join(gt_root, gt_group_name, cur_dataset) if gt_group_name else os.path.join(gt_root, cur_dataset)
            pairs = _list_pair_like_subdirs(gt_dataset_path) if os.path.exists(gt_dataset_path) else []
            if pairs:
                return pairs
            return [pair_hint or cur_dataset]

        pairs = []
        gt_root = os.path.join(root_dir, 'GT')
        gt_group = resolve_group_folder(gt_root, cur_group) if cur_group else None
        gt_group_name = gt_group or cur_group
        gt_dataset_path = os.path.join(gt_root, gt_group_name, cur_dataset) if gt_group_name else os.path.join(gt_root, cur_dataset)
        if os.path.exists(gt_dataset_path):
            pairs = _list_pairs_under_dataset(gt_dataset_path)

        if not pairs:
            for method in all_methods:
                method_root = os.path.join(root_dir, method)
                resolved_group = resolve_group_folder(method_root, cur_group) if cur_group else None
                group_name = resolved_group or cur_group
                method_dataset_path = os.path.join(method_root, group_name, cur_dataset) if group_name else os.path.join(method_root, cur_dataset)
                if os.path.exists(method_dataset_path):
                    pairs = _list_pairs_under_dataset(method_dataset_path)
                    if pairs:
                        break
        return pairs

    all_pairs = _compute_pairs_for_dataset(group, dataset, structure, pair_hint=args.pair)

    if not all_pairs:
        log.error(f"No pairs found for dataset: {dataset}")
        sys.exit(1)

    log.info(f"Discovered {len(all_pairs)} pairs: {all_pairs}")

    # Determine starting pair
    if args.pair is None:
        current_pair_idx = 0
    else:
        if args.pair in all_pairs:
            current_pair_idx = all_pairs.index(args.pair)
        else:
            log.warn(f"Pair '{args.pair}' not found; starting from the first pair.")
            current_pair_idx = 0


    def get_input_folder_for_pair(pair, *, dataset_name, group_name):
        """Build the input_folder mapping for a given pair (directory or file-list values)."""
        # For shared layout, pair is ignored and root is used as-is
        input_folder = discover_local_inputs(
            root_dir,
            all_methods,
            group=group_name,
            dataset=dataset_name,
            pair=pair,
            structure=structure,
        )

        if not input_folder:
            raise ValueError(f"No valid method paths found for pair {pair}")
        return input_folder


    output_folder = output_dir

    def _validate_input_folder(input_folder):
        # Validate inputs (align with compare.py)
        for name, source in input_folder.items():
            if isinstance(source, (list, tuple)):
                imgs = list(source)
                if len(imgs) == 0:
                    log.error(f"No images found for {name} in shared-folder layout")
                    raise ValueError(f"No images found for {name}")
                log.info(f"{name}: {len(imgs)} images")
            else:
                if not os.path.exists(source):
                    log.error(f"Folder not exist: {name} -> {source}")
                    raise ValueError(f"Folder not exist: {name} -> {source}")
                imgs = filter_hidden(glob_single_files(source, IMG_EXTS))
                if len(imgs) == 0:
                    log.error(f"Folder {source} has no images (png/jpg/jpeg)")
                    raise ValueError(f"Folder {source} has no images (png/jpg/jpeg)")

    # Initialize extractor once, then keep switching pairs inside extractor.run()
    state = {
        'dataset': dataset,
        'group': group,
        'all_pairs': all_pairs,
        'current_pair_idx': current_pair_idx,
        'current_pair': None,
    }

    pair = state['all_pairs'][state['current_pair_idx']]
    state['current_pair'] = pair
    input_folder = get_input_folder_for_pair(pair, dataset_name=state['dataset'], group_name=state['group'])
    _validate_input_folder(input_folder)
    pair_info = f"{state['current_pair_idx'] + 1}/{len(state['all_pairs'])}: {pair}"
    extractor = InteractiveLineExtractor(
        input_folder,
        output_folder,
        cache_images=args.cache_images,
        window_independent=args.window_independent,
        horizontal=args.horizontal,
        columns=args.columns,
        inf_length=args.inf_length,
        line_width=args.line_width,
        save_wide_slice=args.save_wide_slice,
        pair_info=pair_info,
        dataset_name=state['dataset'],
        pair_name=pair,
    )

    def _reload_to_current_pair():
        pair = state['all_pairs'][state['current_pair_idx']]
        state['current_pair'] = pair

        input_folder = get_input_folder_for_pair(pair, dataset_name=state['dataset'], group_name=state['group'])
        _validate_input_folder(input_folder)
        pair_info = f"{state['current_pair_idx'] + 1}/{len(state['all_pairs'])}: {pair}"
        extractor.reload_pair(
            input_folder,
            cache_images=args.cache_images,
            dataset_name=state['dataset'],
            pair_name=pair,
            pair_info=pair_info,
        )

    def _on_next_pair():
        state['current_pair_idx'] = (state['current_pair_idx'] + 1) % len(state['all_pairs'])
        _reload_to_current_pair()

    def _on_prev_pair():
        state['current_pair_idx'] = (state['current_pair_idx'] - 1) % len(state['all_pairs'])
        _reload_to_current_pair()

    def _on_switch_dataset(new_group, new_dataset):
        prev_pair = extractor.pair_name or state.get('current_pair')

        state['dataset'] = new_dataset or state['dataset']
        state['group'] = new_group if new_group is not None else state['group']

        state['all_pairs'] = _compute_pairs_for_dataset(state['group'], state['dataset'], structure, pair_hint=prev_pair)
        if not state['all_pairs']:
            raise ValueError(f"No pairs found for dataset: {state['dataset']}")

        state['current_pair_idx'] = state['all_pairs'].index(prev_pair) if prev_pair in state['all_pairs'] else 0
        _reload_to_current_pair()

    extractor.run(
        on_next_pair=_on_next_pair,
        on_prev_pair=_on_prev_pair,
        on_switch_dataset=_on_switch_dataset,
    )

    cv2.destroyAllWindows()
    log.info("Exited.")
