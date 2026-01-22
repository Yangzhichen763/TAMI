"""
Reshape selected tensors in a PyTorch state_dict using an einops.rearrange pattern.
"""

import torch
import argparse
import re
from einops import rearrange


def match_key(key, patterns):
    """判断 key 是否匹配任意 pattern"""
    for pat in patterns:
        if re.search(pat, key):
            return True
    return False


def reshape_tensor(tensor, pattern):
    """使用 einops.rearrange 执行形状转换"""
    try:
        # 自动构造变量名称字典： c=len(tensor)
        shape_dict = {"c": tensor.shape[0]}
        return rearrange(tensor, pattern, **shape_dict)
    except Exception as e:
        print(f"Failed to reshape tensor with pattern '{pattern}' and shape {tensor.shape}: {e}")
        return tensor


def reshape_state_dict(state_dict, key_patterns, modify_weight, modify_bias, pattern):
    """修改匹配 key 的参数形状"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if match_key(key, key_patterns):
            if modify_weight and "weight" in key:
                print(f"Reshaping {key}: {tuple(value.shape)} → pattern '{pattern}'")
                new_state_dict[key] = reshape_tensor(value, pattern)
            elif modify_bias and "bias" in key:
                print(f"Reshaping {key}: {tuple(value.shape)} → pattern '{pattern}'")
                new_state_dict[key] = reshape_tensor(value, pattern)
            else:
                new_state_dict[key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


if __name__ == "__main__":
    # e.g. python "~/Code/LLVE/.me/MemLLVE/quick_tools/reshape_ckpt_key.py" \
    # -i <path1> -o <path2> \
    # -k '.?(I_LCA|HV_LCA).?\.norm' \
    # --weight --bias --shape 'c 1 1 -> c'
    parser = argparse.ArgumentParser(description="Reshape specific weight/bias in PyTorch state_dict using einops pattern.")
    parser.add_argument("--input", "-i", required=True, help="Path to input .pth file")
    parser.add_argument("--output", "-o", default=None, help="Path to output .pth file")
    parser.add_argument("--key", "-k", nargs="+", required=True,
                        help="Key patterns to match (regex supported, e.g., 'I_LCA|HV_LCA', '.?(I_LCA|HV_LCA).?\.norm')")   # 可以在 regex101 先试一下
    parser.add_argument("--weight", action="store_true", help="Whether to modify weight tensors")
    parser.add_argument("--bias", action="store_true", help="Whether to modify bias tensors")
    parser.add_argument("--shape", required=True,
                        help="einops.rearrange pattern (e.g., 'c -> c 1 1' or 'c 1 1 -> c')")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    # Load checkpoint
    state_dict = torch.load(args.input, map_location="cpu")

    # Apply reshape
    new_state_dict = reshape_state_dict(state_dict, args.key, args.weight, args.bias, args.shape)

    # Save back
    torch.save(new_state_dict, args.output)

    print(f"Reshaped state_dict saved to: {args.output}")
