"""
Compare two PyTorch models/checkpoints and report parameter differences by module.
"""

import os.path as osp
import argparse
from collections import defaultdict
import shutil

# 为了在某些服务器中能够导入 basic 包
import sys
sys.path.append('.')
sys.path.append('..')

from basic.models import create_model
from basic.options.argparser import parse
from basic.utils.console.log import get_root_logger, ColorPrefeb as CP

import torch

logger = get_root_logger()


def compare_models_collapse(networkA, networkB, threshold=1e-6, collapse=True):
    result = {
        'total_params': 0,
        'num_different_params': 0,
        'max_abs_diff': 0,
        'different_modules': [],
        'identical': True
    }

    model1_params = dict(networkA.named_parameters())
    model2_params = dict(networkB.named_parameters())

    if set(model1_params.keys()) != set(model2_params.keys()):
        raise ValueError("两个模型的结构不同，无法比较参数")

    # 构建模块树结构
    # Build module tree structure
    module_tree = defaultdict(lambda: {'params': [], 'submodules': set(), 'parent': None})

    # 收集所有参数并构建模块关系
    # Collect all parameters and build module relationships
    for name in model1_params:
        parts = name.split('.')
        module_name = '.'.join(parts[:-1])
        param_name = parts[-1]

        # 添加参数到模块
        # Add the parameter to the module
        module_tree[module_name]['params'].append({
            'full_name': name,
            'name': param_name
        })

        # 更新父模块关系
        # Update the parent module relationship
        current = module_name
        while '.' in current:
            parent = current.rsplit('.', 1)[0]
            module_tree[parent]['submodules'].add(current)
            module_tree[current]['parent'] = parent
            current = parent

    # 后序遍历模块树，从叶子节点开始处理
    # Traverse the module tree in postorder, starting from the leaf nodes
    visited = set()
    module_stats = {}

    # 后序遍历函数
    # Postorder traversal function
    def postorder_traversal(module_name):
        nonlocal visited
        if module_name in visited:
            return

        # 先处理所有子模块
        # Process all submodules first
        for submodule in list(module_tree[module_name]['submodules']):
            postorder_traversal(submodule)

        # 处理当前模块
        # Process the current module
        stats = {
            'name': module_name,
            'total': 0,
            'different': 0,
            'max_diff': 0,
            'all_different': True,
            'any_same': False,
            'collapsed': False,
            'params': []
        }

        # 处理当前模块的参数
        # Process the parameters of the current module
        for param_info in module_tree[module_name]['params']:
            full_name = param_info['full_name']
            param1 = model1_params[full_name]
            param2 = model2_params[full_name]

            if param1.shape != param2.shape:
                raise ValueError(f"参数 {full_name} 的形状不同: {param1.shape} vs {param2.shape}")

            abs_diff = torch.abs(param1 - param2)
            max_diff = torch.max(abs_diff).item()
            mean_diff = torch.mean(abs_diff).item()

            param_size = param1.numel()
            stats['total'] += param_size
            result['total_params'] += param_size

            if max_diff > threshold:
                result['identical'] = False
                result['num_different_params'] += param_size
                result['max_abs_diff'] = max(result['max_abs_diff'], max_diff)
                stats['different'] += param_size
                stats['max_diff'] = max(stats['max_diff'], max_diff)
            else:
                stats['all_different'] = False
                stats['any_same'] = True

            # 记录参数差异
            # Record parameter differences
            param_diff = {
                'name': param_info['name'],
                'full_name': full_name,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'shape': param1.shape,
                'same': max_diff <= threshold
            }
            stats['params'].append(param_diff)

        # 处理子模块的统计
        # Process the statistics of submodules
        for submodule in module_tree[module_name]['submodules']:
            sub_stats = module_stats[submodule]
            stats['total'] += sub_stats['total']
            stats['different'] += sub_stats['different']
            stats['max_diff'] = max(stats['max_diff'], sub_stats['max_diff'])

            # 如果子模块有相同的参数，则当前模块不可能全部不同
            # If a submodule has any same parameters, the current module cannot be all different
            if not sub_stats['all_different'] or sub_stats['any_same']:
                stats['all_different'] = False
            if sub_stats['any_same']:
                stats['any_same'] = True

        # 决定是否折叠当前模块
        # Determine if the current module should be collapsed
        if stats['different'] > 0 and stats['all_different']:
            stats['collapsed'] = True

        module_stats[module_name] = stats
        visited.add(module_name)

    # 从根节点开始遍历
    # Traverse the module tree from the root node
    root_modules = [name for name in module_tree if module_tree[name]['parent'] is None]
    for root in root_modules:
        postorder_traversal(root)

    # 收集需要显示的模块
    # Collect modules to display
    def collect_results(module_name, depth=0):
        stats = module_stats[module_name]

        # 如果整个模块被折叠，直接添加摘要
        # If the module is collapsed, add a summary directly
        if stats['collapsed'] and collapse:
            result['different_modules'].append({
                'name': module_name,
                'total': stats['total'],
                'different': stats['different'],
                'max_diff': stats['max_diff'],
                'collapsed': True,
                'depth': depth
            })
            return

        # 添加当前模块（可能包含参数和子模块）
        # Add the current module (which may contain parameters and submodules)
        module_info = {
            'name': module_name,
            'total': stats['total'],
            'different': stats['different'],
            'max_diff': stats['max_diff'],
            'collapsed': False,
            'params': [p for p in stats['params'] if not p['same']],
            'depth': depth
        }
        result['different_modules'].append(module_info)

        # 递归处理子模块
        # Recursively process submodules
        for submodule in sorted(module_tree[module_name]['submodules']):
            collect_results(submodule, depth + 1)

    for root in root_modules:
        collect_results(root)

    if result['total_params'] > 0:
        result['percentage_different'] = (result['num_different_params'] /
                                          result['total_params'] * 100)
    else:
        result['percentage_different'] = 0

    return result


def check_network_diff():
    parser = argparse.ArgumentParser("Network Parameter Difference")
    parser.add_argument('--optA', '-A', type=str, default='/path/to/optionA.yml',
                        help='Path to first option YAML file.')
    parser.add_argument('--optB', '-B', type=str, default='/path/to/optionB.yml',
                        help='Path to second option YAML file.')
    parser.add_argument('--verbose', action='store_true',
                        help='Detailed parameters by module.')

    args = parser.parse_args()

    # [option] 解析配置文件
    def parse_option(opt_path):
        if osp.isfile(opt_path):
            opt = parse(opt_path, is_train=False)

            opt['device'] = "cpu"
            return opt
        else:
            logger.error('Option file not specified.')
            exit(1)
    optA, optB = parse_option(args.optA), parse_option(args.optB)

    ### [build & load model] 构建模型
    modelA = create_model(optA)  # see at basic/models/<method>/video_<method>_model.py
    modelB = create_model(optB)

    ### [compare] 比较模型参数
    result = compare_models_collapse(modelA.net, modelB.net, collapse=args.verbose)

    print(f"Total parameters                    : {result['total_params']}")
    print(f"Number of different parameters      : {result['num_different_params']}")
    print(f"Percentage of different parameters  : {result['percentage_different']:.2f}%")
    print(f"Max absolute difference             : {result['max_abs_diff']:.6f}")

    if result['identical']:
        print(CP.yes("All parameters are the same."))
        return

    print("Different modules:")
    indent = "  "
    screen_width = shutil.get_terminal_size().columns
    for module in result['different_modules']:
        depth_indent = indent * module['depth']

        if module['collapsed']:
            print(f"{depth_indent}+ {module['name']} ("
                  f"{module['different']}/{module['total']} #Params, "
                  f"Max: {CP.number(module['max_diff'], '.6f')}"
                  f")")
        else:
            if module['different'] > 0:
                module_name = module['name'].split('.')[-1]
                content = f"{depth_indent}- {module_name}"
                footage = f"{module['different']}/{module['total']}"
                footage = f"{footage:>{screen_width - len(content)}}"
                print(f"{content}{footage}")
                for param in module['params']:
                    if not param['same']:
                        param_indent = depth_indent + indent
                        name = param['name'].split('.')[-1]
                        content = f"{param_indent}- {name} "
                        tip = (f"Max:{param['max_diff']:.6f}, "
                               f"Avg:{param['mean_diff']:.6f}")
                        _content = f"{content}{CP.dim(tip)}"
                        footage = f"{tuple(param['shape'])}"
                        footage = f"{footage:<24}"
                        footage = f"{footage:>{screen_width - len(content) - len(tip)}}"
                        print(f"{_content}{CP.dim(footage)}")


if __name__ == '__main__':
    check_network_diff()
