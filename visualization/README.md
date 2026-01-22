# ✨ Interactive Timeline Slice Extractor (timeslice)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/Status-Research%20Tool-informational)

</div>

An interactive tool to extract a **temporal “timeline slice”** from a sequence of images for multiple methods.
It is designed for low-light enhancement / restoration result analysis where you want to compare **how a specific line (row/column) evolves over time** across different methods.

This README documents the usage of:

- `visualization/timeslice.py`

---

## Overview 📌

### Features ✨

- Interactive slice-line selection (mouse drag + wheel length control)
- Compare multiple methods side-by-side over time (grid or per-method windows)
- Fast navigation: next/prev frame, next/prev sequence (pair), jump by image name
- One-key export: saves raw frame, line overlay, and timeline slice images
- Flexible workspace discovery with multiple folder structures (`--structure auto`)

## Contents 📑

- Getting Started 🚀
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Workspace Layout](#workspace-layout)
- User Guidance 🧭
  - [Interaction](#interaction)
  - [Saving Outputs](#saving-outputs)
- Reference ⚙️
  - [CLI Options](#cli-options)
  - [Supported Folder Structures](#supported-folder-structures)
  - [Troubleshooting](#troubleshooting)

---

## Getting Started 🚀

<a id="installation"></a>
### Installation 🧰

- Python 3.8+ (Python 3.10 recommended)
- Core dependencies: `opencv-python`, `numpy`, `natsort`

```bash
# Conda (recommended)
conda create -n timeslice python=3.10 -y
conda activate timeslice
pip install -r requirements.txt
```

<details>
<summary>venv alternative</summary>

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

</details>

<a id="quick-start"></a>
### Quick Start 🚀

```bash
python visualization/timeslice.py
```

- Minimal two-step flow for local reviews:
1) Put your method results under a common root (default: `visualization/examples/`).
2) Run `python visualization/timeslice.py`, move the slice line, then press `s` to save.
- If your layout differs, set `--root` / `--structure` (and optionally `--group`, `--dataset`, `--pair`).

<details>
<summary>Common custom run (dataset + pair)</summary>

```bash
python visualization/timeslice.py \
    --root visualization/examples \
    --dataset SDSD-indoor \
    --pair pair21
```

</details>

---

## Workspace Layout 🗂️

The tool assumes your results are organized by **method** under a common root. Each method contains datasets (and optionally groups/pairs).

Minimal (dataset-only) layout:

```text
<root>/
  <methodA>/
    <dataset>/
      0001.png
      0002.png
      ...
  <methodB>/
    <dataset>/
      0001.png
      0002.png
      ...
```

If you have per-sequence folders (pairs):

```text
<root>/
  <methodA>/
    <dataset>/
      <pair>/
        0001.png
        0002.png
```

The default example folder is:

- `visualization/examples/`

---

## Interaction 🧭

### Mouse

- **Left click + drag**: move the slice line (sets the center position).
- **Mouse wheel**: adjust line length.
    - If you started with “infinite length”, the first wheel action will switch to adjustable length.

### Keyboard

- `n` / `p`: next / previous **frame**
- `m` / `b`: next / previous **sequence (pair)**
- `r`: toggle slice orientation (**vertical** ↔ **horizontal**)
- `s`: save outputs for the current frame
- `z` / `y`: undo / redo
- `Enter`: switch dataset (input `dataset` or `group/dataset`)
- `Space`: jump to an image by name (stem or full filename)
- `q` / `ESC`: quit

Display modes:

- Default: show all method timelines in a **grid** window.
- `--window-independent`: open **one window per method**.

---

## Saving Outputs 💾

Press `s` to save three files per method into:

```text
<output>/<dataset>/<pair>/<frame_name>/
  <method>.png
  <method>_line.png
  <method>_slice.png
  <method>_slice_w{line_width}.png  (optional)
```

Where:

- `<method>.png` is the original frame.
- `<method>_line.png` overlays the slice line on the frame.
- `<method>_slice.png` is the extracted timeline image (thin version: 1px per frame).
- `<method>_slice_w{line_width}.png` is the wide timeline image (saved when `--save-wide-slice` and `--line-width > 1`).

The default output directory is:

- `visualization/timeline_slice/`

---

## CLI Options ⚙️

Run `python visualization/timeslice.py -h` to see full help.

Common options:

- `--root, -r`: root directory containing method folders (default: `visualization/examples`)
- `--output, -o`: output directory for saved slices (default: `visualization/timeline_slice`)
- `--group, -g`: optional group folder under each method
- `--dataset, -ds`: dataset name (default: `SDSD-indoor`)
- `--pair, -p`: pair/sequence name (default: first discovered)
- `--structure`: folder layout (default: `auto`)
- `--methods`: comma/space separated method names to include
- `--exclude, -x`: comma/space separated method names to exclude
- `--methods-file`: path to `methods.txt` (default: `visualization/methods.txt`, used if `--methods` not set)

Extraction/UI:

- `--horizontal`: start with horizontal slice direction (default: vertical)
- `--inf-length, -il / --no-inf-length`: start with “infinite” line length (default: enabled; use `--no-inf-length` to disable)
- `--columns, -c`: number of columns for grid layout (default: auto)
- `--line-width, -lw`: timeline slice width in pixels (default: 5)
- `--save-wide-slice / --no-save-wide-slice`: also save a wide timeline image using `--line-width` (default: enabled)
- `--window-independent`: open one timeline window per method
- `--no-cache`: disable image caching (lower RAM usage; slower disk reads)

Logging:

- `--no-color`: disable ANSI colored logs
- `--log-level`: `debug|info|warn|error`

---

## Supported Folder Structures 🗂️

Select with `--structure`:

- `auto` (default): tries the known layouts in order
- `group-dataset-pair`: `<root>/<method>/<group>/<dataset>/<pair>/<img>`
- `group-dataset`: `<root>/<method>/<group>/<dataset>/<img>`
- `dataset-only`: `<root>/<method>/<dataset>/<img>`
- `flat`: `<root>/<method>/<img>`
- `shared`: `<root>/<image-id>/<method>.<ext>`

Notes:

- Hidden files/folders (starting with `.`) are ignored.
- For `shared`, each subfolder under `<root>` is treated as a time step; the tool collects `<method>.*` files under each step.

---

## Troubleshooting 🛠️

### 1) “Import cv2/numpy/natsort could not be resolved” in VS Code

- This usually means your VS Code interpreter is not the same environment where packages are installed.
- Ensure your Python environment is selected correctly, then run `pip install -r requirements.txt`.

### 2) “No reference images found” / empty timelines

- The tool uses `GT` as reference if present; otherwise it uses the first discovered key.
- Verify that each method folder contains images under the selected `--dataset/--pair` (and `--group` if used).

### 3) Pair switching does not find folders

- Try setting `--structure` explicitly (e.g. `dataset-only` or `group-dataset-pair`).
- If your group naming differs by hyphen/underscore, `--group` will auto-match them.

### 4) Too slow / too much RAM

- Use `--no-cache` to reduce RAM usage.
- Reduce the number of methods by using `--methods` or `--exclude`.

---

## Call to Action ⭐

- If this tool helps your research workflow, consider starring the repo.
- PRs and issues for bugs / UX improvements are welcome.
