"""
Toggle a wrapper key (default: 'params') in a PyTorch checkpoint to unwrap/wrap a state_dict.
"""

import torch
import argparse
import os

try:
    import sys
    sys.path.append('.')

    from basic.utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass

def convert_checkpoint(input_path, output_path, unwrap_key="params"):
    """
    双向转换：
    ① 如果文件包含 'params' 字段，则去掉该字段并保存纯 state_dict。
    ② 如果文件不包含 'params' 字段，则添加该字段包装保存。
    """
    checkpoint = torch.load(input_path, map_location="cpu")

    # Case 1: 包含 'params' 字段 → 去除包装
    if isinstance(checkpoint, dict) and unwrap_key in checkpoint:
        state_dict = checkpoint[unwrap_key]
        torch.save(state_dict, output_path)
        print(f"{CP.false('-')} Removed '{unwrap_key}' wrapper. Saved pure state_dict to: {output_path}")

    # Case 2: 不包含 'params' 字段 → 添加包装
    elif isinstance(checkpoint, dict):
        wrapped = {unwrap_key: checkpoint}
        torch.save(wrapped, output_path)
        print(f"{CP.true('+')} Added '{unwrap_key}' wrapper. Saved wrapped state_dict to: {output_path}")

    else:
        print("The input file is not a recognized PyTorch state_dict or checkpoint.")
        print(f"Type: {type(checkpoint)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toggle 'params' wrapper in PyTorch checkpoint files.")
    parser.add_argument("--input", "-i", help="Path to input .pth file")
    parser.add_argument("--output", "-o", help="Path to output .pth file (default: auto-generate)")
    parser.add_argument("--unwrap_key", "-k", default="params", help="Key to unwrap (default: 'params')")
    args, extra = parser.parse_known_args()

    # 如果没有用 -i 指定，就尝试把第一个未识别参数当作 input
    if not args.input:
        if len(extra) > 0:
            args.input = extra[0]
        else:
            parser.error("the following arguments are required: --input/-i")

    # 自动生成输出路径
    if not args.output:
        args.output = args.input

    convert_checkpoint(args.input, args.output, args.unwrap_key)
