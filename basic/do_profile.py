
import sys
sys.path.append('.')
sys.path.append('..')

from basic.options.argparser import get_default_argument_parser, parse_options


if __name__ == '__main__':
    parser = get_default_argument_parser('Model Profile')
    for action in parser._actions:
        if action.dest == 'opt':
            action.help = "Model configuration file"
    parser.add_argument('--input-size', '-i', nargs='+', type=int, default=[256, 256], help='Input image size')
    parser.add_argument('--batch-size', '-b', type=int, default=None, help='Batch size')
    parser.add_argument('--channels', '-c', type=int, default=None, help='Input channels')
    parser.add_argument('--verbose', action='store_true', help='Print detailed model parameters')
    parser.add_argument('--depth', '-d', default=5, type=int, help='Print model parameters for specified depth')
    parser.add_argument('--mode', '-m', default='alfp',
                        help='Model profile mode: "a" for memory allocation, "l" for latency, "f" for flops, "p" for parameters')
    parser.add_argument('--trace', '-t', action='store_true', help='Profile model with trace')
    parser.add_argument('--quite-frozen', action='store_true', help='Print only non-frozen parameters for verbose')
    parser.add_argument('--run-times', type=int, default=100, help='Number of runs for profiling average time')
    opt, args = parse_options(parser, is_train=False)

    from basic.archs import define_network
    from basic.metrics.summary import summary, print_detailed_num_params, print_frozen_params, profile_train
    from basic.utils.console.log import get_root_logger
    from basic.archs.memory.util import mem_engaged

    if 'network' not in opt:
        raise ValueError('Network architecture is not specified in the configuration file.')

    model = define_network(opt['network'])
    logger = get_root_logger(force_set_info=True)

    size = args.input_size
    if len(size) <= 2:
        size = (args.batch_size or 1, args.channels or 3, *size)
    elif len(size) == 3:
        if args.channels is not None:
            size = (1, args.channels, *size[1:])
        else:
            size = (1, *size)
    else:
        if args.batch_size is not None:
            size = (args.batch_size, *size[1:])

    if args.verbose:
        print_frozen_params(model)
        with mem_engaged():
            print_detailed_num_params(model, input_size=size, depth=args.depth, quite_frozen=args.quite_frozen)
    with mem_engaged():
        summary = summary(model, input_size=size, mode=args.mode, runs=args.run_times)
    if args.trace:
        profile_train(model, input_size=size, step=2)
