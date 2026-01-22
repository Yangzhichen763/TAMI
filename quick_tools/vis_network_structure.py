"""
Export a model graph to TensorBoard for visualizing network structure.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import os

import sys
sys.path.append(".")
sys.path.append("..")

from basic.archs import define_network
from basic.options import get_default_argument_parser, parse_options
from basic.utils.console.log import get_striped_time_str


def vis_network_structure(net, test_input, log_dir: str, log_file: str):
    writer = SummaryWriter(os.path.join(log_dir, log_file))

    writer.add_graph(net, test_input)
    writer.close()

    print(f"Network structure visualization saved to {os.path.join(log_dir, log_file)}.")


if __name__ == '__main__':
    parser = get_default_argument_parser(f'Network Structure Visualization')
    for action in parser._actions:
        if action.dest == 'opt':
            action.default = f"/path/to/option.yml"
    parser.add_argument('--shape', '-s', nargs='+', type=int, default=[1, 3, 256, 256], help='Input tensor shape')
    opt, args, unknown = parse_options(parser, is_train=True, with_unknown=True)  # unknown 用来输入模型 test 中要传入的参数

    net = define_network(opt['network'])
    test_input = torch.rand(*args.shape)

    log_dir = ".plotlogs/network_structures"
    log_file = f"{net.__class__.__name__}_{get_striped_time_str()}"

    vis_network_structure(net, test_input, log_dir, log_file)

    print(f"Network structure visualization saved to {os.path.join(log_dir, log_file)}.")


    ''' custom usage example:
    # pip install tensorboard
    
    net = None # define your network here

    from torch.utils.tensorboard import SummaryWriter
    log_path = f".plotlogs/network_structures/track_on"
    writer = SummaryWriter()
    writer.add_graph(net, torch.rand(1, 3, 256, 256))
    writer.close()
    print(f"Network structure visualization saved to {log_path}.")
    '''
