import os
import pyiqa

import math
import time
import threading
import queue
import concurrent.futures
from typing import Dict, List, Optional


import sys
sys.path.append(".")
from basic.utils.registry import METRICS_REGISTRY

try:
    from basic.utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass

"""
Command usage: https://github.com/chaofengc/IQA-PyTorch?tab=readme-ov-file
pyiqa [metric_name(s)] -t [image_path or dir] -r [image_path or dir] --device [cuda or cpu] --verbose
"""


@METRICS_REGISTRY.register()
class IQA:
    def __init__(self, metric_type, any_gt_mean=False):
        super().__init__()
        self.metric_type = metric_type

        self.metric_func = pyiqa.create_metric(self.metric_type)
        self.any_gt_mean = any_gt_mean

    def __call__(self, *inputs):
        """
        Args:
            inputs (torch.Tensor): list of tensors of shape (N, C, H, W).
        """
        metric_func = self.metric_func
        metric_func.to(inputs[0].device)

        if metric_func.metric_mode == 'FR':
            assert len(inputs) == 2, f"Unsupported number of inputs for {self.metric_type} metric: {len(inputs)}"

            pred, gt = inputs
            if self.any_gt_mean:
                pred, gt = gt_mean(pred, gt)
            return metric_func(pred, gt)
        elif metric_func.metric_mode == 'NR':
            assert len(inputs) == 1, f"Unsupported number of inputs for {self.metric_type} metric: {len(inputs)}"

            return metric_func(*inputs)
        else:
            raise ValueError(f"Unsupported number of inputs: {len(inputs)}")


@METRICS_REGISTRY.register()
class IQAs:
    def __init__(self, *metric_types, **kwargs):
        super().__init__()
        self.metric_types = metric_types

        self.metric_funcs = [
            {
                'type': metric_type,
                'name': metric_type.split(' ')[0],
                'func': pyiqa.create_metric(metric_type.split(' ')[0]),
                'gt-mean': 'gt-mean' in metric_type.split(' '),
                'gt-mean-c': 'gt-mean-c' in metric_type.split(' '),
            }
            for metric_type in self.metric_types
        ]
        self.traditional = kwargs.get('traditional', False)

        if self.traditional:
            import basic.metrics as metrics
            traditional_func_dict = {
                'ssim': metrics.ssim.SSIM(),
                'psnr': metrics.psnr.PSNR(),
            }
            for d in self.metric_funcs:
                for k, v in traditional_func_dict.items():
                    if d['name'] == k:
                        d['func'] = v


    def __call__(self, *inputs):
        """
        Args:
            inputs (torch.Tensor): list of tensors of shape (N, C, H, W).
        """
        metrics = {}
        for d in self.metric_funcs:
            metric_type = d['type']
            metric_name = d['name']
            metric_func = d['func']
            any_gt_mean = d['gt-mean']
            any_gt_mean_c = d['gt-mean-c']

            try:
                metric_func.to(inputs[0].device)
            except:
                pass

            if metric_func.metric_mode == 'FR':
                assert len(inputs) >= 2, f"Unsupported number of inputs for {metric_name} metric: {len(inputs)}"

                pred, gt = inputs[:2]
                if any_gt_mean_c:
                    pred, gt = gt_mean(pred, gt, channel_wise=True)
                elif any_gt_mean:
                    pred, gt = gt_mean(pred, gt)
                metrics[metric_type] = metric_func(pred, gt)
            elif metric_func.metric_mode == 'NR':
                assert len(inputs) >= 1, f"Unsupported number of inputs for {metric_name} metric: {len(inputs)}"

                pred, = inputs[:1]
                metrics[metric_type] = metric_func(pred)
            else:
                raise ValueError(f"Unsupported number of inputs: {len(inputs)}")

        return metrics


@METRICS_REGISTRY.register()
class MultiThreadIQAs:
    def __init__(self, *metric_types, **kwargs):
        """Multi-threaded IQA metric runner.

        Key design:
        - Each metric gets its own ThreadPoolExecutor with n threads (default 2).
        - For each metric, we keep an instance pool of metric objects of size n.
          This avoids sharing a single metric instance across threads (often not thread-safe).
        """
        super().__init__()
        self.metric_types = metric_types
        self.n_threads_per_metric = int(kwargs.get('n_threads_per_metric', 2))
        if self.n_threads_per_metric <= 0:
            raise ValueError(f"n_threads_per_metric must be positive, got {self.n_threads_per_metric}")

        self.device = kwargs.get('device', None)
        self.traditional = kwargs.get('traditional', False)

        self.metric_specs = [
            {
                'type': metric_type,
                'name': metric_type.split(' ')[0],
                'gt-mean': 'gt-mean' in metric_type.split(' '),
                'gt-mean-c': 'gt-mean-c' in metric_type.split(' '),
            }
            for metric_type in self.metric_types
        ]

        # Per-metric executors and per-metric metric-instance pools.
        self._executors: Dict[str, concurrent.futures.ThreadPoolExecutor] = {}
        self._func_pools: Dict[str, queue.Queue] = {}
        self._func_mode: Dict[str, str] = {}

        for spec in self.metric_specs:
            metric_type = spec['type']
            metric_name = spec['name']

            ctor = self._resolve_metric_ctor(metric_name)
            func_pool: queue.Queue = queue.Queue(maxsize=self.n_threads_per_metric)
            metric_mode = None
            for _ in range(self.n_threads_per_metric):
                func = ctor()
                if metric_mode is None:
                    metric_mode = getattr(func, 'metric_mode', None)
                if self.device is not None:
                    try:
                        func.to(self.device)
                    except Exception:
                        pass
                func_pool.put(func)

            self._executors[metric_type] = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads_per_metric)
            self._func_pools[metric_type] = func_pool
            self._func_mode[metric_type] = metric_mode or 'FR'

    def _resolve_metric_ctor(self, metric_name: str):
        if not self.traditional:
            return lambda: pyiqa.create_metric(metric_name)

        import basic.metrics as metrics
        traditional_ctor_dict = {
            'ssim': lambda: metrics.ssim.SSIM(),
            'psnr': lambda: metrics.psnr.PSNR(),
        }
        return traditional_ctor_dict.get(metric_name, lambda: pyiqa.create_metric(metric_name))

    def shutdown(self, wait: bool = True):
        for ex in self._executors.values():
            ex.shutdown(wait=wait)

    def submit(self, pred, gt, gt_path: str, metrics_pool: dict, pool_lock: threading.Lock):
        """Submit all metrics for one sample.

        Updates metrics_pool[gt_path][metric_type] with float results (or NaN on failure).
        Returns list of futures.
        """
        futures: List[concurrent.futures.Future] = []
        for spec in self.metric_specs:
            metric_type = spec['type']
            executor = self._executors[metric_type]
            fut = executor.submit(
                self._compute_one_and_update,
                metric_type,
                spec,
                pred,
                gt,
                gt_path,
                metrics_pool,
                pool_lock,
            )
            futures.append(fut)
        return futures

    def __call__(self, *inputs):
        """Synchronous convenience wrapper.

        Keeps compatibility with the old call style: returns a dict mapping metric_type -> float.
        """
        assert len(inputs) >= 1
        pred = inputs[0]
        gt = inputs[1] if len(inputs) >= 2 else None

        if gt is None:
            any_fr = any(self._func_mode.get(mt, 'FR') == 'FR' for mt in self.metric_types)
            assert not any_fr, "FR metrics require (pred, gt) inputs"

        gt_path = '__single__'
        metrics_pool = {gt_path: {mt: None for mt in self.metric_types}}
        lock = threading.Lock()
        futures = self.submit(pred, gt, gt_path, metrics_pool, lock)
        if futures:
            concurrent.futures.wait(futures)
        return metrics_pool[gt_path]

    def _compute_one_and_update(self, metric_type, spec, pred, gt, gt_path, metrics_pool, pool_lock: threading.Lock):
        func_pool = self._func_pools[metric_type]
        func = func_pool.get()
        try:
            if self.device is None:
                try:
                    func.to(pred.device)
                except Exception:
                    pass

            metric_mode = getattr(func, 'metric_mode', self._func_mode.get(metric_type, 'FR'))
            if metric_mode == 'FR':
                pred_i, gt_i = pred, gt
                if spec.get('gt-mean-c', False):
                    pred_i, gt_i = gt_mean(pred_i, gt_i, channel_wise=True)
                elif spec.get('gt-mean', False):
                    pred_i, gt_i = gt_mean(pred_i, gt_i)
                out = func(pred_i, gt_i)
            elif metric_mode == 'NR':
                out = func(pred)
            else:
                raise ValueError(f"Unsupported metric_mode={metric_mode} for metric_type={metric_type}")

            if hasattr(out, 'item'):
                value = float(out.item())
            else:
                value = float(out)
        except Exception:
            value = float('nan')
        finally:
            func_pool.put(func)

        with pool_lock:
            # Dynamic update; caller may or may not have pre-initialized keys.
            if gt_path not in metrics_pool:
                metrics_pool[gt_path] = {}
            metrics_pool[gt_path][metric_type] = value

        return value


def gt_mean(pred, gt, channel_wise=False):
    """
    Adjusts the brightness of the input predicted image (pred) to match the average brightness of the target image (gt).

    Parameters:
        pred (torch.Tensor): The predicted image, with range [0, 1], and shape (B, C, H, W), and data type float32.
        gt (torch.Tensor): The target image (Ground Truth), with range [0, 1], and shape (B, C, H, W), and data type float32.

    Returns:
        gt (torch.Tensor): The unmodified target image.
        pred (torch.Tensor): The brightness-adjusted predicted image, with the same shape and data type as the input.
    """
    # Compute the mean brightness of the target (gt) and predicted (pred) images
    if channel_wise:
        mean_pred = pred.mean(dim=(2, 3), keepdim=True)
        mean_gt = gt.mean(dim=(2, 3), keepdim=True)
    else:
        mean_pred = pred.mean()
        mean_gt = gt.mean()

    # Adjust the brightness of the predicted image by scaling
    scaled_pred = pred * (mean_gt / mean_pred)

    # Ensure the output is within [0, 1] range
    pred = scaled_pred.clamp(0, 1)

    return pred, gt


def metrics_to_str(name, metrics, max_name_len=None):
    try:
        import torch
    except Exception:
        torch = None

    metrics_strs = [
        f'{CP.keyword(k)}: {CP.number(f"{v.item():.4f}")}' if (torch is not None and isinstance(v, torch.Tensor)) else f'{CP.keyword(k)}: {CP.number(f"{v:.4f}")}'
        for k, v in metrics.items()
    ]
    metrics_trimmed_strs = [
        f'{k}: {v.item():.4f}' if (torch is not None and isinstance(v, torch.Tensor)) else f'{k}: {v:.4f}'
        for k, v in metrics.items()
    ]
    if max_name_len is None:
        out_str = f"{name}: {', '.join([f'{s:<{16 + len(s) - len(t_s)}}' for t_s, s in zip(metrics_trimmed_strs, metrics_strs)])}"
    else:
        out_str = f"{name:<{max_name_len}}: {', '.join([f'{s:<{16 + len(s) - len(t_s)}}' for t_s, s in zip(metrics_trimmed_strs, metrics_strs)])}"
    return out_str


def compute_iqa_metrics(
        dataroot_pred, dataroot_gt,
        metric_types=("psnr", "ssim", "lpips", "niqe", "brisque", "nima", "musiq", "pi", "psnr gt-mean", "ssim gt-mean", "lpips gt-mean", "psnr gt-mean-c", "ssim gt-mean-c", "lpips gt-mean-c"),
        # "nima", "musiq", "pi", "maniqa", "clipiqa", "dists"
        verbose=True,
        **kwargs
):
    import torch
    import shutil
    from datetime import datetime
    from basic.datasets.simple_glob_dataset import PairedImageDataset
    paired_dataset = PairedImageDataset(dataroot_pred=dataroot_pred, dataroot_gt=dataroot_gt)
    iqa = IQAs(*metric_types, **kwargs)


    results = {
        key: [] for key in metric_types
    }
    max_name_len = max([len(os.path.basename(data['pred']['path'])) for data in paired_dataset] + [len('average')])
    start_t = time.time()
    for data in paired_dataset:
        pred, gt = data['pred']['image'].unsqueeze(0).cuda(), data['gt']['image'].unsqueeze(0).cuda()
        image_name = os.path.basename(data['pred']['path'])

        # Compute the IQA metrics
        metrics = iqa(pred, gt)

        # Collect the results
        for k, v in metrics.items():
            results[k].append(v.item())

        if verbose:
            # Print verbose output
            print(metrics_to_str(image_name, metrics, max_name_len=max_name_len))

    # Calculate the average values
    avg_results = {
        k: sum(v) / len(v) for k, v in results.items()
    }

    # Print the final results
    screen_width = shutil.get_terminal_size().columns
    if verbose:
        elapsed = time.time() - start_t
        ts = f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | elapsed: {elapsed:.2f}s "
        print(ts.center(screen_width, "="))
    print(metrics_to_str('average', avg_results))
    return avg_results


def compute_iqa_metrics_multi_thread(
        dataroot_pred, dataroot_gt,
        metric_types=
        ("psnr", "ssim", "lpips", "psnr gt-mean", "ssim gt-mean", "lpips gt-mean"),
        # ("psnr", "ssim", "lpips", "niqe", "brisque", "nima", "musiq", "pi", "psnr gt-mean", "ssim gt-mean", "lpips gt-mean", "psnr gt-mean-c", "ssim gt-mean-c", "lpips gt-mean-c"),
        # "nima", "musiq", "pi", "maniqa", "clipiqa", "dists"
        verbose=True,
        n_threads_per_metric: int = 2,
        poll_interval_s: float = 0.1,
        **kwargs
):
    import torch
    import shutil
    from datetime import datetime
    from collections import deque
    from basic.datasets.simple_glob_dataset import PairedImageDataset

    if poll_interval_s <= 0:
        raise ValueError(f"poll_interval_s must be > 0, got {poll_interval_s}")

    paired_dataset = PairedImageDataset(dataroot_pred=dataroot_pred, dataroot_gt=dataroot_gt)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    iqa = MultiThreadIQAs(*metric_types, device=device, n_threads_per_metric=n_threads_per_metric, **kwargs)

    # 指标池：key=gt图像路径，value={metric_type: value(None/float)}，动态更新
    metrics_pool: Dict[str, Dict[str, Optional[float]]] = {}
    pool_lock = threading.Lock()

    # queue：按读取顺序存储gt图像路径（支持peek）
    path_queue = deque()
    queue_lock = threading.Lock()
    reader_done = threading.Event()

    results: Dict[str, List[float]] = {k: [] for k in metric_types}
    max_name_len = len('average')

    all_futures: List[concurrent.futures.Future] = []

    def reader_worker():
        nonlocal max_name_len
        try:
            for data in paired_dataset:
                pred, gt = data['pred']['image'].unsqueeze(0).to(device), data['gt']['image'].unsqueeze(0).to(device)
                gt_path = data['gt']['path']

                # 动态创建 key；value 字典也按该样本动态建立/扩展
                with pool_lock:
                    if gt_path not in metrics_pool:
                        metrics_pool[gt_path] = {}
                    # 这里仅为该gt_path初始化需要的指标key，默认None（不是一次性初始化所有图片key）
                    for metric_type in metric_types:
                        metrics_pool[gt_path].setdefault(metric_type, None)

                with queue_lock:
                    path_queue.append(gt_path)

                max_name_len = max(max_name_len, len(os.path.basename(gt_path)))
                all_futures.extend(iqa.submit(pred, gt, gt_path, metrics_pool, pool_lock))
        finally:
            reader_done.set()

    start_t = time.time()
    t = threading.Thread(target=reader_worker, name='iqa_reader', daemon=True)
    t.start()

    # 每隔0.1秒检查一遍queue peek；如peek对应的指标都算完->出队->print->立刻检查下一个peek
    while True:
        time.sleep(poll_interval_s)

        while True:
            with queue_lock:
                head = path_queue[0] if len(path_queue) > 0 else None

            if head is None:
                break

            with pool_lock:
                head_metrics = metrics_pool.get(head, {})
                ready = all(head_metrics.get(mt, None) is not None for mt in metric_types)
                metrics_snapshot = dict(head_metrics) if ready else None

            if not ready:
                break

            with queue_lock:
                if len(path_queue) > 0 and path_queue[0] == head:
                    path_queue.popleft()

            # 收集结果并打印
            for k, v in metrics_snapshot.items():
                if v is None:
                    continue
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    continue
                results[k].append(float(v))

            if verbose:
                print(metrics_to_str(os.path.basename(head), metrics_snapshot, max_name_len=max_name_len))

        if reader_done.is_set():
            with queue_lock:
                queue_empty = len(path_queue) == 0
            if queue_empty:
                break

    # 确保任务结束并释放线程池
    if all_futures:
        concurrent.futures.wait(all_futures)
    iqa.shutdown(wait=True)

    # 计算平均（忽略 NaN/Inf）
    avg_results = {}
    for k, vals in results.items():
        clean = [v for v in vals if not (math.isnan(v) or math.isinf(v))]
        avg_results[k] = (sum(clean) / len(clean)) if len(clean) > 0 else float('nan')

    # Print the final results
    import shutil
    screen_width = shutil.get_terminal_size().columns
    if verbose:
        from datetime import datetime
        elapsed = time.time() - start_t
        ts = f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | elapsed: {elapsed:.2f}s "
        print(ts.center(screen_width, "="))
    print(metrics_to_str('average', avg_results))
    return avg_results


if __name__ == '__main__':
    """
    usage: 
    python basic/metrics/pyiqa_metrics.py -i <predicted_images_folder> -r <ground_truth_images_folder>] -g <gpu_indices>] [--traditional] [--multi-thread] [--num-threads <n>]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute image quality assessment (IQA) metrics between predicted and ground truth images."
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        type=str,
        help="Path to the folder containing predicted images."
    )
    parser.add_argument(
        '--reference', '-r',
        type=str,
        default=None,
        help="Path to the folder containing ground truth images."
    )
    # parser.add_argument(
    #     '--verbose',
    #     action='store_true',
    #     help="Enable verbose output."
    # )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Disable verbose output."
    )
    parser.add_argument(
        '--gpus', '-g',
        type=str,
        default=None,
        help="Comma-separated list of GPU indices to use."
    )
    parser.add_argument(
        '--traditional', '-t',
        action='store_true',
        help="Use traditional implementation instead of the iqa one."
    )
    parser.add_argument(
        '--multi-thread', '-m',
        action='store_true',
        help="Use multi-threaded IQA computation."
    )
    parser.add_argument(
        '--num-threads', '-n',
        type=int,
        default=2,
        help="Number of threads per metric for multi-threaded computation."
    )

    args = parser.parse_args()

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    import torch

    if args.reference is None:
        args.reference = args.input
        print(f"Using the same folder for predicted and ground truth images: {args.reference}")
    if args.multi_thread:
        df = compute_iqa_metrics_multi_thread(args.input, args.reference, verbose=not args.quiet, traditional=args.traditional, n_threads_per_metric=args.num_threads)
    else:
        df = compute_iqa_metrics(args.input, args.reference, verbose=not args.quiet, traditional=args.traditional)
