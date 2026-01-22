"""
Print a compact hierarchical tree of parameter keys in a PyTorch state_dict/checkpoint.
"""

import torch
import argparse
from collections import defaultdict


def build_tree(keys):
    """Build a hierarchical tree structure with parameter compression.

    Args:
        keys: List of dot-separated parameter names from state_dict

    Returns:
        Nested dictionary representing the model structure
    """
    tree = defaultdict(dict)
    param_groups = defaultdict(list)

    # First pass: identify all parameter groups
    for key in keys:
        parts = key.split('.')
        if len(parts) >= 2:
            parent = '.'.join(parts[:-1])  # Get parent module path
            param_name = parts[-1]  # Get parameter name
            param_groups[parent].append(param_name)

    # Second pass: build the tree structure
    for key in keys:
        parts = key.split('.')
        parent = '.'.join(parts[:-1])

        # Handle parameter groups
        if parent in param_groups and parent not in tree:
            # Generate compressed parameter notation
            prefixes = []
            for param in param_groups[parent]:
                if param == 'weight':
                    prefixes.append('w')
                elif param == 'bias':
                    prefixes.append('b')
                else:
                    prefixes.append(param)  # Use first 3 chars for other params

            # Add to tree with compressed notation
            node = tree
            for part in parent.split('.')[:-1]:
                node = node.setdefault(part, {})
            node[parts[-2]] = {'_params': ', '.join(sorted(set(prefixes)))}
            continue

        # Normal case: non-parameter-group nodes
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        if parts[-1] not in node:  # Avoid overwriting existing nodes
            node[parts[-1]] = None

    return tree


def print_tree(tree, indent=0):
    """Print the hierarchical tree with proper indentation.

    Args:
        tree: Nested dictionary structure
        indent: Current indentation level
    """
    for key, subtree in sorted(tree.items()):
        if isinstance(subtree, dict) and '_params' in subtree:
            # Parameter group node
            print(" " * indent + f"{key} ({subtree['_params']})")
        elif isinstance(subtree, dict):
            # Normal module node
            print(" " * indent + f"{key}")
            print_tree(subtree, indent + 2)
        else:
            # Leaf node
            print(" " * indent + f"{key}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Visualize .pth file contents with hierarchical structure and parameter compression")
    parser.add_argument("path", help="Path to the .pth file")
    args = parser.parse_args()

    # Load model data
    data = torch.load(args.path, map_location="cpu")
    print(f"\nContents of '{args.path}':\n" + "-" * 50)

    if isinstance(data, dict):
        # Process state_dict
        if 'params' in data:
            data = data['params']
            print("Model state_dict 'params' key detected. Using this key instead.")
        if 'state_dict' in data:
            data = data['state_dict']
            print("Model state_dict 'state_dict' key detected. Using this key instead.")
        tree = build_tree(data.keys())
        print_tree(tree)
    else:
        # Process full model
        print("Full model detected. Hierarchical display requires state_dict keys.")
        for name, _ in data.named_modules():
            print(name)
