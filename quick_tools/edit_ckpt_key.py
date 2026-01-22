"""
Rename specific keys/attributes in a PyTorch checkpoint (state_dict) and save the modified file.
"""

import torch
import argparse


def rename_attributes_in_state_dict(state_dict, module_name, old_attr, new_attr):
    """Rename keys in the state_dict that match `module_name.old_attr` to `module_name.new_attr`."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(f"{module_name}") and old_attr in key:
            new_key = key.replace(old_attr, new_attr)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Rename attributes in a PyTorch .pth file.")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input .pth file")
    parser.add_argument("--output", "-o", required=True,
                        help="Path to output .pth file")
    parser.add_argument("--module", "-m", default="I_LCA",
                        help="Module name in the model as attribute (default: I_LCA)") # 模块名称在类中的属性名称，比如 I_LCA1
    parser.add_argument("--old", required=True,
                        help="Original attribute name (e.g., 'iel')")
    parser.add_argument("--new", required=True,
                        help="New attribute name (e.g., 'gdfn')")
    args = parser.parse_args()

    # Load the model or state_dict
    checkpoint = torch.load(args.input, map_location="cpu")

    # Case 1: If the file is a state_dict (only parameters)
    if isinstance(checkpoint, dict):
        if 'params' in checkpoint:
            checkpoint = checkpoint['params']

        new_state_dict = rename_attributes_in_state_dict(
            checkpoint, args.module, args.old, args.new
        )
        torch.save(new_state_dict, args.output)
        print(f"Success! Modified state_dict saved to {args.output}")

    # Case 2: If the file contains the full model (structure + parameters)
    else:
        def recursive_rename(module, old_name, new_name):
            """Recursively rename attributes in all submodules."""
            for name, child in module.named_children():
                if hasattr(child, old_name):
                    setattr(child, new_name, getattr(child, old_name))
                    delattr(child, old_name)
                recursive_rename(child, old_name, new_name)

        recursive_rename(checkpoint, args.old, args.new)
        torch.save(checkpoint, args.output)
        print(f"Success! Modified model saved to {args.output}")
