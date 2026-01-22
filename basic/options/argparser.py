import os
import os.path as osp
import argparse
import random

from .options import parse, NoneDict
from basic.utils.console.proctitle import set_proctitle_progress

from basic.utils.console.log import get_root_logger
logger = get_root_logger()


'''
参数解析工具函数
'''


def get_default_argument_parser(title) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(title)

    # [device] 设备参数
    parser.add_argument('--cpu', action='store_true',
                        help='whether to use cpu device')
    parser.add_argument('--gpus', '-g',                 type=str,   default='0',
                        help='CUDA_VISIBLE_DEVICES, gpu device number for training phase')
    parser.add_argument('--test-gpus', '-tg',           type=str,
                        help='CUDA_VISIBLE_DEVICES, gpu device number for validating phase if specified, otherwise use the same as training phase')
    #TODO: 训练和测试阶段分离，加快训练速度

    # [distributed] 分布式计算
    parser.add_argument('--local_rank',                 type=int,   default=0)
    parser.add_argument('--launcher',                   type=str,   default='none',
                        choices=['none', 'pytorch', 'slurm'],
                        help='job launcher')

    # [visualization] 训练过程可视化参数
    parser.add_argument('--proctitle', '-p',            type=str,   default=None,
                        help='process title')

    # [option] 输入配置文件
    parser.add_argument('--opt',                        type=str,   default='/path/to/option.yml',
                        help='Path to option YAML file.')
    parser.add_argument('--alias',                      type=str,   default=None,
                        help='Alias for option file.')

    # [debug] 调试参数
    # 并非训练的 debug，训练的 debug 需要在配置文件中设置
    parser.add_argument('--debug',                      action='store_true',
                        help='whether to use debug mode')

    # [custom] 更自由的参数设置，比如 -hp=offline,custom
    parser.add_argument('--hyperparams', '-hp',         type=lambda s: s.split(','),   default=None,
                        help='Custom hyperparameters for training, e.g. --hyperparams=offline,custom')

    return parser


def parse_options(parser: argparse.ArgumentParser, is_train=True, with_unknown=False):
    from basic.utils.console.log import ColorPrefeb as CP

    if with_unknown:
        args, unknown = parser.parse_known_args()

        # parse unknown args as dict
        unknown_args = {}
        for arg in unknown:
            if arg.startswith("--"):
                key = arg[2:]
                value = unknown[unknown.index(arg)]
                unknown_args[key] = value
        unknown_args = argparse.Namespace(**unknown_args)
    else:
        args = parser.parse_args()
        unknown_args = None

    # [pre-device]
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # [option] 解析配置文件
    if osp.isfile(args.opt):
        basename = osp.basename(args.opt)
        opt_name = '.'.join(basename.split('.')[:-1]) if '.' in basename else basename
        opt = parse(args.opt, is_train=is_train, alias=args.alias)
    else:
        if args.opt == '/path/to/option.yml':
            logger.error(f'[{CP.error("error")}] Option file not specified.')
        else:
            raise FileNotFoundError(f'Option file {args.opt} not found.')
        opt_name = 'default'
        opt = NoneDict()

    # [device] 设置 CUDA_VISIBLE_DEVICES，和 gpu 相关参数
    if args.cpu or (not args.gpus and not 'gpu_ids' in opt):
        args.device = 'cpu'
        opt['device'] = dict(train='cpu', val='cpu')
        opt['gpu_ids'] = dict(train='', val='')
    else:
        def process_gpus(gpus):
            if isinstance(gpus, list):
                gpus = ','.join(str(x) for x in gpus)
            elif not isinstance(gpus, str):
                gpus = str(gpus)
            return gpus

        if args.gpus is None:
            if 'gpu_ids' in opt:
                if isinstance(opt['gpu_ids'], dict):
                    opt['gpu_ids']['train'] = process_gpus(opt['gpu_ids']['train'])
                    opt['gpu_ids']['val'] = process_gpus(opt['gpu_ids']['val'])
                else:
                    gpu_ids = process_gpus(opt['gpu_ids'])
                    opt['gpu_ids'] = dict(train=gpu_ids, val=gpu_ids)
                args.gpus = opt['gpu_ids']['train']
            else:
                opt['gpu_ids'] = dict(train='0', val='0')
                args.gpus = '0'
        else:
            if 'gpu_ids' in opt:
                if isinstance(opt['gpu_ids'], dict):
                    opt['gpu_ids']['train'] = process_gpus(args.gpus)
                    opt['gpu_ids']['val'] = process_gpus(opt['gpu_ids']['val'])
                else:
                    gpu_ids = process_gpus(args.gpus)
                    opt['gpu_ids'] = dict(train=gpu_ids, val=gpu_ids)
            else:
                gpu_ids = process_gpus(args.gpus)
                opt['gpu_ids'] = dict(train=gpu_ids, val=gpu_ids)

        args.device = 'cuda'
        opt['device'] = dict(train='cuda', val='cuda')

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(f'[{CP.keyword("argparser")}] Set CUDA_VISIBLE_DEVICES to <{CP.keyword(args.gpus)}> in training phase.')

    # [random seed] 随机种子
    from basic.utils.misc import set_random_seed
    # Modified from RetinexFormer(https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/train.py#L56)
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed)

    # [distributed] 分布式计算设置
    from basic.utils.dist import get_dist_info, init_dist
    if args.launcher == 'none':
        opt['dist'] = False
        print(f'[{CP.keyword("argparser")}] Disable distributed.')
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print(f'[{CP.keyword("argparser")}] init dist .. ', args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()

    # [proctitle] 设置进程名
    args.proctitle = args.proctitle if args.proctitle is not None else opt_name
    set_proctitle_progress(args.proctitle, rolling=False)

    # [debug] 调试参数
    if args.debug:
        print(f'[{CP.keyword("argparser")}] Enable debug mode.')

        from basic.utils.console.log import turn_on_debug
        turn_on_debug()
    if unknown_args is not None and any('debug' in k for k in vars(unknown_args).keys()):
        for k in vars(unknown_args).keys():
            if k != 'debug' and 'debug' in k:
                k = (
                    k.replace("debug_", "").replace("_debug", "")
                     .replace("debug-", "").replace("-debug", "")
                )
                print(f'[{CP.keyword("argparser")}] Enable debug mode <{CP.keyword(k)}>.')

                from basic.utils.console.log import turn_on_debug
                turn_on_debug(k)

        # 去除 unknown_args 中的 debug 参数
        unknown_args = {k: v for k, v in vars(unknown_args).items() if k != 'debug' and 'debug' not in k}
        unknown_args = argparse.Namespace(**unknown_args)

    if with_unknown:
        return opt, args, unknown_args
    else:
        return opt, args
