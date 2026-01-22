"""
Pretty-print a hierarchical view of PyTorch checkpoint keys with weight/bias suffix compression.
"""

import torch
import argparse


def insert_nested(tree, key_path):
    """
    Insert keys into a nested dict based on token list.
    """
    current = tree
    for k in key_path:
        if k not in current:
            current[k] = {}
        current = current[k]


def build_tree_from_flat_keys(flat_keys):
    """
    Convert flat keys into a hierarchical tree.
    Also track weight/bias flags for node merging.
    Removes: total_ops, total_params
    """
    tree = {}
    weight_bias_map = {}

    for key in flat_keys:
        # skip unwanted keys
        if key.endswith("total_ops") or key.endswith("total_params"):
            continue

        parts = key.split(".")
        last = parts[-1]

        insert_nested(tree, parts)

        # track w/b for merging
        parent_key = ".".join(parts[:-1])
        if parent_key not in weight_bias_map:
            weight_bias_map[parent_key] = {"w": False, "b": False}

        if last == "weight":
            weight_bias_map[parent_key]["w"] = True
        elif last == "bias":
            weight_bias_map[parent_key]["b"] = True

    # Remove raw weight/bias leaf nodes
    def prune_weight_bias(node, parent_path=""):
        keys = list(node.keys())

        for k in keys:
            full_path = parent_path + "." + k if parent_path else k

            if isinstance(node[k], dict):
                prune_weight_bias(node[k], full_path)

            # weight/bias removal
            if k in ("weight", "bias"):
                if parent_path in weight_bias_map:
                    del node[k]

            # also remove leftover total_ops/params if any
            if k in ("total_ops", "total_params"):
                del node[k]

    prune_weight_bias(tree)
    return tree, weight_bias_map


def print_tree(node, weight_bias_map, prefix="", parent_path=""):
    """
    Recursively print hierarchical tree with (w), (b), or (w, b) suffixes.
    """
    keys = list(node.keys())

    for i, k in enumerate(keys):
        full_path = parent_path + "." + k if parent_path else k
        is_last = (i == len(keys) - 1)
        connector = "└── " if is_last else "├── "

        suffix = ""
        if full_path in weight_bias_map:
            w = weight_bias_map[full_path]["w"]
            b = weight_bias_map[full_path]["b"]
            if w and b:
                suffix = " (w, b)"
            elif w:
                suffix = " (w)"
            elif b:
                suffix = " (b)"

        print(prefix + connector + k + suffix)

        new_prefix = prefix + ("    " if is_last else "│   ")
        print_tree(node[k], weight_bias_map, new_prefix, full_path)


def load_and_print_keys(pth_path):
    """
    Load .pth checkpoint and print hierarchical key structure.
    """
    print(f"\n🔍 Loading: {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu")

    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint does not contain a dictionary-like object.")

    flat_keys = list(ckpt.keys())
    tree, weight_bias_map = build_tree_from_flat_keys(flat_keys)

    print("\n📂 Keys Hierarchy:\n")
    print_tree(tree, weight_bias_map)
    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hierarchical keys with weight/bias merging")
    parser.add_argument("path", help="Path to the .pth file")
    args = parser.parse_args()

    load_and_print_keys(args.path)
