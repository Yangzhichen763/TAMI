import sys
import enum
import shutil
import os
import warnings
from collections import OrderedDict, defaultdict
from contextlib import contextmanager

from thop import profile
import torch
import torch.nn as nn
from torch.profiler import profile as torch_profile, ProfilerActivity
from torch.autograd.profiler import record_function

try:
    from .latency import timer
except:
    import time
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def record(self):
            self.end_time = time.time()

        def elapsed(self):
            return self.end_time - self.start_time

        def __float__(self):
            return self.elapsed()

    def timer():
        timer = Timer()
        timer.start()
        try:
            timer.record()
            yield timer  # Execute the code block inside the with statement
        finally:
            timer.record()

# from basic.utils.console.log import get_root_logger, ColorPrefeb as CP, auto_number_to_str, get_striped_time_str

try:
    from basic.utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass

try:
    from basic.utils.console.log import get_striped_time_str
except:
    def get_striped_time_str():
        import pytz
        from datetime import datetime

        china_tz = pytz.timezone('Asia/Shanghai')
        now = datetime.now(china_tz)

        timestamp_text = now.strftime('%Y%m%d_%H%M%S_%f')[:-3]
        return timestamp_text

try:
    from basic.utils.console.log import auto_number_to_str
except:
    def auto_number_to_str(value):
        value = f"{value:.4f}"
        return value

try:
    from basic.utils.console.log import get_root_logger
    logger = get_root_logger()
except:
    import logging
    logger = logging.getLogger('main')

try:
    from basic.utils.console.pbar import PbarContext
except ImportError:
    class PbarContext:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False  # 不吞异常

        def update(self, *args, **kwargs):
            pass

        def set_description(self, *args, **kwargs):
            pass

        def close(self):
            pass

try:
    from .util import clone_module
except:
    def clone_module(module):
        return module


try:
    import pandas as pd
except ImportError:
    logger.warning("Pandas is not available.")


any_summary = False

def is_summary():
    return any_summary


def on_summary(func):
    def wrapper(*args, **kwargs):
        global any_summary
        original_value = any_summary  # 保存原始值
        any_summary = True            # 设置为 True
        try:
            result = func(*args, **kwargs)  # 执行函数
        finally:
            any_summary = original_value     # 恢复原始值
        return result
    return wrapper


# 克隆模型，避免修改原模型（以及被之前的记录所影响）
@contextmanager
def get_model(model):
    _model = clone_module(model)
    yield _model
    del _model


#region ==[Latency, Params & FLOPs]==
@on_summary
def summary(
        model, input_size=(2, 3, 256, 256), device='cuda', runs=100,
        to_str=False, logger=None,
        mode='alfp',
        *model_args, **model_kwargs
):
    """
    Calculate the latency, FLOPs, and number of parameters of a model. Print the latency(ms), FLOPs(G), and number of parameters(M) of the model to the console.

    Args:
        model (nn.Module): The model to calculate the metrics for.
        input_size (tuple): The input size of the model.
        device (str): The device to run the model on.
        runs (int): The number of runs to average the latency over.
        to_str (bool): Whether to return the metrics as a string.
        logger (logging.Logger): The logger to use for printing.
        *model_args: Any additional positional arguments to pass to the model.
        **model_kwargs: Any additional keyword arguments to pass to the model.

    Returns:
        dict: A dict containing the latency(ms), FLOPs(G), and number of parameters(M) of the model.
    """
    if logger is None:
        logger = get_root_logger()

    def mode_to_str(mode):
        if mode == 'a':
            return 'allocation'
        elif mode == 'l':
            return 'latency'
        elif mode == 'p':
            return 'params'
        elif mode == 'f':
            return 'flops'
        else:
            raise ValueError(f"Invalid mode: {mode}")
    mode_str = ', '.join([mode_to_str(m) for m in mode])
    logger.info(f"Starting summary with mode: {mode_str}")
    summary = dict()

    ### [memory allocation]
    if 'a' in mode:
        with get_model(model) as train_model:
            train_model.eval()
            if 'cuda' in device:
                class MemTracker:
                    def __init__(self):
                        self.allocated = None
                        self.reserved = None
                @contextmanager
                def measure_memory_usage(tracker=None):
                    if tracker is None:
                        tracker = MemTracker()
                    torch.cuda.empty_cache()
                    with synchronize(device):
                        start_allocated_mem = torch.cuda.memory_allocated()
                        start_reserved_mem = torch.cuda.memory_reserved()
                        yield tracker
                    end_allocated_mem = torch.cuda.memory_allocated()
                    end_reserved_mem = torch.cuda.memory_reserved()
                    tracker.allocated = end_allocated_mem - start_allocated_mem
                    tracker.reserved = end_reserved_mem - start_reserved_mem

                torch.cuda.reset_peak_memory_stats()

                # load model
                with measure_memory_usage() as mem_model:
                    train_model = train_model.to(device)

                # load test tensor
                with measure_memory_usage() as mem_tensor:
                    test_tensor = torch.randn(input_size).to(device)

                # forward
                with measure_memory_usage() as mem_forward:
                    test_params = torch.tensor([1.0], device=device, requires_grad=True)    # 防止有些模型参数全冻结的情况
                    output = train_model(test_tensor + test_params, *model_args, **model_kwargs)

                # backward
                with measure_memory_usage() as mem_backward:  # TODO: 对于模型的任意输出，都能有效应对
                    if isinstance(output, (list, tuple)):
                        loss = sum(x.sum() for x in output)
                    else:
                        loss = output.sum()
                    loss.backward()

                # peak memory
                mem_peak = torch.cuda.max_memory_allocated()

                logger.info(f"Memory allocation: "
                            f"\n - model: {mem_model.reserved / 1024 ** 2:.3f} MB ({mem_model.allocated / 1024 ** 2:.3f} MB allocated)"
                            f"\n - test tensor: {mem_tensor.reserved / 1024 ** 2:.3f} MB ({mem_tensor.allocated / 1024 ** 2:.3f} MB allocated)"
                            f"\n - forward: {mem_forward.reserved / 1024 ** 2:.3f} MB ({mem_forward.allocated / 1024 ** 2:.3f} MB allocated)"
                            f"\n - backward: {mem_backward.reserved / 1024 ** 2:.3f} MB ({mem_backward.allocated / 1024 ** 2:.3f} MB allocated)"
                            f"\n - peak: {mem_peak / 1024 ** 2:.3f} MB")
                summary['memory_model'] = mem_model if not to_str else f"{mem_model.reserved / 1024 ** 2:.3f} MB"
                summary['memory_tensor'] = mem_tensor if not to_str else f"{mem_tensor.reserved / 1024 ** 2:.3f} MB"
                summary['memory_forward'] = mem_forward if not to_str else f"{mem_forward.reserved / 1024 ** 2:.3f} MB"
                summary['memory_backward'] = mem_backward if not to_str else f"{mem_backward.reserved / 1024 ** 2:.3f} MB"
                summary['memory_peak'] = mem_peak if not to_str else f"{mem_peak / 1024 ** 2:.3f} MB"

    ### [FLOPs]
    if 'f' in mode:
        # thop MACS
        macs, params = get_macs(model, input_size, device, *model_args, **model_kwargs)
        logger.info(f"MACs: {macs / 1e9:.3f} G")
        summary['macs'] = macs if not to_str else f"{macs / 1e9:.3f} G"

        # torch.profiler FLOPs
        flops_dict = get_flops(model, input_size, device, 10, easy_return=False, *model_args, **model_kwargs)
        flops = flops_dict['flops']
        flops_list = flops_dict['flops_list']
        flops_deltas = flops_dict['flops_deltas']
        mem_stats = flops_dict['mem_stats']

        table = flops_dict['tables'][0]
        if 'module' in table:
            with open(f"./.plotlogs/prof_flops_module_{get_striped_time_str()}.txt", "w") as f:
                f.write(str(table['module']))
        if 'op' in table:
            with open(f"./.plotlogs/prof_flops_op_{get_striped_time_str()}.txt", "w") as f:
                f.write(str(table['op']))

        summary['flops'] = flops if not to_str else f"{flops / 1e9:.3f} G"

        def count_max_duplicate(nums):
            return max(set(nums), key=nums.count)

        unique_flops_deltas = list(OrderedDict.fromkeys(flops_deltas))
        flops_delta = sum(flops_deltas) / len(flops_deltas) if len(flops_deltas) > 0 else 0
        logger.info(f"FLOPs delta list: {[f'{f / 1e9:.3f}' for f in unique_flops_deltas]} G")
        summary['flops_delta'] = flops_deltas if not to_str else f"{'.'.join([f'{f / 1e9:.3f}' for f in flops_deltas])} G"

        # unique_mem_stats = list({tuple(sorted(stat.items())): stat for stat in mem_stats}.values())
        cuda_bytes = [stat['peak_cuda_bytes'] for stat in mem_stats if 'peak_cuda_bytes' in stat]
        unique_cuda_bytes = list(OrderedDict.fromkeys(flops_deltas))
        cuda_byte_deltas = [cuda_bytes[i] - cuda_bytes[i-1] for i in range(1, len(cuda_bytes))]
        logger.info(f"Peak CUDA memory usage during FLOPs computation: {[f'{m / 1024 ** 2:.3f}' for m in unique_cuda_bytes]} MB")
        summary['peak_cuda_mem_bytes'] = cuda_bytes if not to_str else f"{'.'.join([f'{m / 1024 ** 2:.3f}' for m in cuda_bytes])} MB"

        logger.info(f"FLOPs start: {flops_list[0] / 1e9:.3f} G, max: {flops / 1e9:.3f} G, d: {count_max_duplicate(flops_deltas) / 1e9:.3f} G, "
                    f"CUDA Mem start: {cuda_bytes[0] / 1024 ** 2:.3f} MB, peak: {max(cuda_bytes) / 1024 ** 2:.3f} MB, d: {count_max_duplicate(cuda_byte_deltas) / 1024 ** 2:.3f} MB")

    ### [Params]
    if 'p' in mode:
        num_params = get_params(model)
        logger.info(f"Params: {num_params / 1e6:.3f} M ({num_params * 4 / 2 ** 20:.3f} MB for fp32)")
        summary['params'] = num_params if not to_str else f"{num_params / 1e6:.3f} M"

    ### [Latency]
    if 'l' in mode:
        runs, latency_dict = get_latency(model, input_size, device, runs, *model_args, **model_kwargs)
        logger.info(f"Latency: {latency_dict['avg'] * 1e3:.2f} ms ± {latency_dict['std'] * 1e3: .2f} ms per inference "
                    f"(mean ± std of {runs} runs), mid: {latency_dict['mid'] * 1e3:.2f} ms")
        summary['latency'] = latency_dict['avg'] if not to_str else f"{latency_dict['avg'] * 1e3:.2f} ms"

    return summary


@on_summary
def get_latency(model, input_size=(2, 3, 256, 256), device='cuda', runs=None, *model_args, **model_kwargs):
    global any_summary
    any_summary = True

    # assert single-batch input
    if len(input_size) >= 4 and input_size[0] > 1:
        runs = runs or input_size[0]
        input_size = (1, *input_size[1:])
    else:
        runs = runs or 1

    from basic.archs.memory.memory_enhanced_arch import MemoryEnhancedNet
    from basic.archs.memory.util import mem_engaged_if

    with get_model(model) as model, PbarContext(show_if=True, start=0, total=runs) as pbar_ctx:
        pbar_ctx.set_description(f'Latency Computing')
        model = model.to(device)
        test_tensor = torch.randn(input_size).to(device)

        # get average latency
        latency = []
        with synchronize(device), torch.no_grad():
            model.eval()
            for _ in range(runs):
                with mem_engaged_if(isinstance(model, MemoryEnhancedNet)), timer() as t:
                    _ = model(test_tensor, *model_args, **model_kwargs)
                latency.append(t.elapsed())
                pbar_ctx.update()

    if runs >= 20:
        b = runs // 20
        latency = sorted(latency)[b:-b] # remove outliers
    avg_latency = sum(latency) / len(latency)
    mid_latency = latency[len(latency) // 2]
    std_latency = (sum((x - avg_latency) ** 2 for x in latency) / len(latency)) ** 0.5

    any_summary = False
    return runs, dict(
        avg=avg_latency,
        mid=mid_latency,
        std=std_latency,
    )


@on_summary
def get_params(model):
    """
    Calculate the number of parameters in a model.

    Args:
        model (nn.Module): The model to calculate the number of parameters for.

    Returns:
        int: The number of parameters in the model.
    """
    num_params = sum(param.numel() for param in model.parameters())
    return num_params


@on_summary
def get_flops(
    model,
    input_size=(2, 3, 256, 256),
    device='cuda',
    runs=None,
    easy_return=True,
    print_table="module,op",               # False / "module" / "op"
    table_row_limit=-1,                 # -1 prints all rows
    table_group_by_input_shape=True,    # whether to group op table by input shape
    *model_args,
    **model_kwargs
):
    """
    Calculate FLOPs and parameter count of a model using torch.profiler.

    Args:
        model (nn.Module): Target model to profile.
        input_size (tuple): Input tensor shape.
        device (str): Device to run profiling on.
        runs (int): Number of profiling runs. If input_size[0] > 1, uses that as runs.
        easy_return (bool): If True, return (flops, params). If False, return a dict with details.
        print_table (bool/str): False / "module" / "op".
        table_row_limit (int): Number of rows printed in profiler table; -1 prints all rows.
        table_group_by_input_shape (bool): Whether to group operator-level table by input shape.
        *model_args/**model_kwargs: Extra args passed into model forward().

    Returns:
        (flops, params) if easy_return else dict with detailed stats.
    """

    # Ensure single-batch input (for consistent profiling); use batch>1 as multiple runs
    if len(input_size) >= 4 and input_size[0] > 1:
        runs = runs or input_size[0]
        input_size = (1, *input_size[1:])
    else:
        runs = runs or 1

    # =========================
    # Helpers
    # =========================
    @contextmanager
    def _suppress_fd_output(suppress_stdout: bool = False, suppress_stderr: bool = True):
        """
        Suppress OS-level stdout/stderr by redirecting file descriptors (fd 1/2) to /dev/null.
        This also suppresses C++ logs that bypass Python sys.stderr/sys.stdout.
        """
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved = {}
        try:
            if suppress_stdout:
                saved[1] = os.dup(1)
                os.dup2(devnull_fd, 1)
            if suppress_stderr:
                saved[2] = os.dup(2)
                os.dup2(devnull_fd, 2)
            yield
        finally:
            # Restore stdout/stderr fds
            for fd, old_fd in saved.items():
                try:
                    os.dup2(old_fd, fd)
                finally:
                    os.close(old_fd)
            os.close(devnull_fd)

    def _total_flops_from_prof(prof) -> int:
        """Sum all available FLOPs from profiler events."""
        total = 0
        try:
            events = prof.events()
        except Exception:
            # Compatibility fallback for different torch profiler internals
            events = getattr(prof, "function_events", None)
            if events is None:
                events = getattr(getattr(prof, "profiler", None), "function_events", [])

        for e in events:
            f = getattr(e, "flops", 0)
            if f:
                total += int(f)
        return total

    def _count_params(m: nn.Module) -> int:
        """Count total number of parameters in the model."""
        return sum(p.numel() for p in m.parameters())

    def _peak_mem_from_prof(prof):
        """
        Retrieve peak CPU and CUDA memory usage from profiler events.
        Some versions use cpu_memory_usage/cuda_memory_usage, others only expose self_* fields.
        """
        peak_cpu = 0
        peak_cuda = 0
        try:
            events = prof.events()
        except Exception:
            events = getattr(prof, "function_events", None)
            if events is None:
                events = getattr(getattr(prof, "profiler", None), "function_events", [])

        for e in events:
            cpu_m = getattr(e, "cpu_memory_usage", 0) or 0
            cuda_m = getattr(e, "cuda_memory_usage", 0) or 0

            # Fallback to self fields
            if not cpu_m:
                cpu_m = getattr(e, "self_cpu_memory_usage", 0) or 0
            if not cuda_m:
                cuda_m = getattr(e, "self_cuda_memory_usage", 0) or 0

            # Use max magnitude (memory release events may appear as negative)
            peak_cpu = max(peak_cpu, abs(int(cpu_m)))
            peak_cuda = max(peak_cuda, abs(int(cuda_m)))

        return {"peak_cpu_bytes": peak_cpu, "peak_cuda_bytes": peak_cuda}

    def _pick_sort_key(table_str: str):
        """
        torch.profiler.table requires sort_by to be a valid column name.
        Different PyTorch versions may provide different column names.
        """
        for k in ("self_flops", "flops", "cpu_time_total", "self_cpu_time_total"):
            if k in table_str:
                return k
        return "cpu_time_total"

    def add_module_record_hooks(model: nn.Module):
        """
        Wrap each module forward pass with record_function(module_name),
        so profiler can aggregate operator events per module (module-level table).
        """
        handles = []

        def pre_hook(mod, inp):
            name = getattr(mod, "_prof_name", mod.__class__.__name__)
            mod.__prof_ctx__ = record_function(name)
            mod.__prof_ctx__.__enter__()

        def post_hook(mod, inp, out):
            ctx = getattr(mod, "__prof_ctx__", None)
            if ctx is not None:
                ctx.__exit__(None, None, None)

        # Assign stable readable names: use named_modules path
        for module_name, mod in model.named_modules():
            mod._prof_name = module_name if module_name != "" else mod.__class__.__name__
            handles.append(mod.register_forward_pre_hook(pre_hook))
            handles.append(mod.register_forward_hook(post_hook))

        return handles

    def _build_inputs(test_tensor):
        """
        Build model inputs tuple while keeping your original argument injection behavior.
        """
        if len(model_args) > 0 and len(model_kwargs) > 0:
            return (test_tensor, *model_args, *model_kwargs.values())
        elif len(model_kwargs) > 0:
            return (test_tensor, *model_kwargs.values())
        elif len(model_args) > 0:
            return (test_tensor, *model_args)
        else:
            return (test_tensor,)

    # =========================
    # Main
    # =========================
    # Keep your original project dependencies
    from basic.archs.memory.memory_enhanced_arch import MemoryEnhancedNet
    from basic.archs.memory.util import mem_engaged_if

    flops_list = []
    params_list = []
    mem_list = []
    tables_list = []  # store profiler tables for each run if enabled

    with get_model(model) as model, \
         PbarContext(show_if=True, start=0, total=runs) as pbar_ctx, \
         warnings.catch_warnings():

        warnings.filterwarnings("ignore", category=UserWarning, message="This API is being deprecated")

        pbar_ctx.set_description('FLOPs Computing')

        model = model.to(device)
        test_tensor = torch.randn(input_size, device=device)

        old_cpp_log_level = os.environ.get("TORCH_CPP_LOG_LEVEL", None)
        os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

        try:
            with torch.no_grad():
                model.eval()

                for _run in range(runs):
                    with mem_engaged_if(isinstance(model, MemoryEnhancedNet)):
                        inputs = _build_inputs(test_tensor)

                        activities = [ProfilerActivity.CPU]
                        if torch.cuda.is_available() and str(device).startswith("cuda"):
                            activities.append(ProfilerActivity.CUDA)
                            torch.cuda.synchronize()

                        want_module_table = "module" in str(print_table).lower()
                        want_op_table = "op" in str(print_table).lower()

                        # Only attach module hooks when module-level table is required
                        handles = add_module_record_hooks(model) if want_module_table else []

                        try:
                            with _suppress_fd_output(suppress_stdout=False, suppress_stderr=True):
                                with torch_profile(
                                    activities=activities,
                                    record_shapes=True,
                                    with_flops=True,
                                    profile_memory=True,
                                ) as prof:
                                    _ = model(*inputs)
                        finally:
                            for h in handles:
                                h.remove()

                        if torch.cuda.is_available() and str(device).startswith("cuda"):
                            torch.cuda.synchronize()

                        flops = _total_flops_from_prof(prof)
                        params = _count_params(model)
                        mem_stat = _peak_mem_from_prof(prof)

                        # ===== build tables =====
                        run_tables = {}

                        if want_module_table:
                            ka = prof.key_averages(group_by_input_shape=False)
                            tmp = ka.table(sort_by="cpu_time_total", row_limit=10)
                            sort_key = _pick_sort_key(tmp)
                            run_tables["module"] = ka.table(sort_by=sort_key, row_limit=table_row_limit)

                        if want_op_table:
                            ka = prof.key_averages(group_by_input_shape=table_group_by_input_shape)
                            tmp = ka.table(sort_by="cpu_time_total", row_limit=10)
                            sort_key = _pick_sort_key(tmp)
                            run_tables["op"] = ka.table(sort_by=sort_key, row_limit=table_row_limit)

                        if run_tables:
                            tables_list.append(run_tables)

                            # # Print tables
                            # if want_module_table and "module" in run_tables:
                            #     print("\n========== Profiler Table (MODULE-level) ==========")
                            #     print(run_tables["module"])
                            #
                            # if want_op_table and "op" in run_tables:
                            #     print("\n========== Profiler Table (OP-level) ==========")
                            #     print(run_tables["op"])

                    flops_list.append(flops)
                    params_list.append(params)
                    mem_list.append(mem_stat)
                    pbar_ctx.update()

        finally:
            # Restore TORCH_CPP_LOG_LEVEL
            if old_cpp_log_level is None:
                os.environ.pop("TORCH_CPP_LOG_LEVEL", None)
            else:
                os.environ["TORCH_CPP_LOG_LEVEL"] = old_cpp_log_level

    flops = max(flops_list) if flops_list else 0
    params = max(params_list) if params_list else 0

    if easy_return:
        return flops, params

    flops_deltas = [flops_list[i] - flops_list[i - 1] for i in range(1, len(flops_list))]
    return {
        'flops': flops,
        'params': params,
        'flops_list': flops_list,
        'flops_deltas': flops_deltas,
        'mem_stats': mem_list,
        'tables': tables_list,   # profiler tables for each run (if enabled)
    }


@on_summary
def get_macs(model, input_size=(2, 3, 256, 256), device='cuda', runs=None, *model_args, **model_kwargs):
    """
    Calculate the FLOPs of a model.

    Args:
        model (nn.Module): The model to calculate the FLOPs for.
        input_size (tuple): The input size of the model.
        device (str): The device to run the model on.
        *model_args: Any additional positional arguments to pass to the model.
        **model_kwargs: Any additional keyword arguments to pass to the model.

    Returns:
        int: The FLOPs of the model.
    """
    # assert single-batch input
    if len(input_size) >= 4 and input_size[0] > 1:
        runs = runs or input_size[0]
        input_size = (1, *input_size[1:])
    else:
        runs = runs or 1

    import warnings
    from basic.archs.memory.memory_enhanced_arch import MemoryEnhancedNet
    from basic.archs.memory.util import mem_engaged_if

    # runs = 100
    flops_list = []
    params_list = []
    with get_model(model) as model, \
         PbarContext(show_if=True, start=0, total=runs) as pbar_ctx, \
         warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="This API is being deprecated")

        pbar_ctx.set_description(f'FLOPs Computing')
        model = model.to(device)
        test_tensor = torch.randn(input_size).to(device)

        with torch.no_grad():
            model.eval()

            for run in range(runs):
                with mem_engaged_if(isinstance(model, MemoryEnhancedNet)), timer() as t:

                    if len(model_args) > 0 and len(model_kwargs) > 0:
                        macs, params = profile(model, inputs=(test_tensor, *model_args, *model_kwargs.values()), verbose=False)
                    elif len(model_kwargs) > 0:
                        macs, params = profile(model, inputs=(test_tensor, *model_kwargs.values()), verbose=False)
                    elif len(model_args) > 0:
                        macs, params = profile(model, inputs=(test_tensor, *model_args), verbose=False)
                    else:
                        macs, params = profile(model, inputs=(test_tensor,), verbose=False)

                flops_list.append(macs * 2)
                params_list.append(params)
                # print(f"FLOPs: {flops / 1e9:.3f} G")
                pbar_ctx.update()
            flops = max(flops_list)
            params = max(params_list)
            return flops, params


@on_summary
def print_detailed_num_params(
        model, input_size=(2, 3, 256, 256), device='cuda',
        depth=5, quite_frozen=False
):
    """
    Calculate each layer's number of parameters in a model.
    Args:
        model (nn.Module): The model to calculate the number of parameters for.
    Returns:
        dict: A dict containing the number of parameters of each layer in the model.
    """
    try:
        from torchinfo import summary as torch_summary
    except ImportError:
        logger.error("Please install torchinfo to use this function.")
        return {}

    # get summary stats
    stats = torch_summary(model, input_size=input_size, device=device, verbose=0, depth=depth)# 表头

    col_width = shutil.get_terminal_size().columns
    indent = "  "

    # ==str== [max-width] get max-width
    width = dict(
        layer=30,
        output_shape=20,
        params=10,
        grad=5,
    )
    for i, layer in enumerate(stats.summary_list, 1):
        layer_depth = layer.depth
        if layer_depth > depth:
            continue

        width['layer'] = max(len(str(layer.class_name)) + layer_depth * len(indent), width['layer'])
        width['output_shape'] = max(len(str(layer.output_size)), width['output_shape'])
        width['params'] = max(len(str(layer.num_params)), width['params'])
        width['grad'] = max(len(str(layer.trainable)), width['grad'])

    # ==str== [gap] fill the gap between columns
    gap = col_width - len(width)
    for key, value in width.items():
        gap -= value
    for i, (key, value) in enumerate(width.items()):
        left = 1 if i < gap % len(width) else 0
        width[key] = value + gap // len(width) + left

    # ==str== [header]
    separator_main = "=" * col_width
    separator_sub = "-" * col_width
    header = (
        f"{'Layer (type)':<{width['layer']}} "
        f"{'Output Shape':<{width['output_shape']}} "
        f"{'Param #':<{width['params']}} "
        f"{'Grad':<{width['grad']}}"
    )

    # ==print== [header]
    print(separator_main)
    print(header)
    print(separator_sub)

    # ==print== [all]
    for i, layer in enumerate(stats.summary_list, 0):
        layer_depth = layer.depth
        if layer_depth > depth:
            continue

        # skip frozen modules if required
        if quite_frozen:
            if layer.trainable == "False":  # layer.trainable is str
                continue
            elif layer.trainable != "True" and layer.num_params == 0:
                do_skip = True
                _layer = layer
                while _layer:
                    _layer = _layer.parent_info
                    if _layer.trainable == "True":
                        do_skip = False
                        break
                if do_skip:
                    continue

        if layer_depth >= 1:
            indent_str = indent * (layer_depth - 1) + "- "
        else:
            indent_str = ""

        is_last_child = i + 1 < len(stats.summary_list) and stats.summary_list[i + 1].depth <= layer_depth
        is_final_layer = i + 1 >= len(stats.summary_list)
        if layer_depth >= depth or is_last_child or is_final_layer:
            params_str = f"{layer.num_params:,}"
            layer_info = (
                f"{indent_str}"
                f"{layer.class_name:<{width['layer'] - layer_depth * len(indent)}} "
                f"{str(layer.output_size):<{width['output_shape']}} "
                f"{params_str:<{width['params']}} "
                f"{str(layer.trainable):<{width['grad']}} "
            )
        else:
            params_str = f"({layer.num_params:,})"
            layer_info = (
                f"{indent_str}"
                f"{layer.class_name:<{width['layer'] - layer_depth * len(indent)}} "
                f"{' ':<{width['output_shape']}} "
                f"{params_str:<{width['params']}} "
            )
        print(layer_info)

    # ==str== [summary]
    max_right_column = max(
        len(f"{stats.total_params:,}"),
        len(f"{stats.trainable_params:,}"),
        len(f"{stats.total_mult_adds:,}"),
    )
    right_column = col_width - max_right_column - 2

    # ==print== [summary]
    print(separator_sub)
    print(f"{'Total params: ':<{right_column}}{stats.total_params:,}")
    print(f"{'Trainable params: ':<{right_column}}{stats.trainable_params:,}")
    print(f"{'Non-trainable params: ':<{right_column}}{stats.total_params - stats.trainable_params:,}")
    print(f"{'Total FLOPs: ':<{right_column}}{stats.total_mult_adds:,}")
    print(separator_main)


@on_summary
def print_frozen_params(
        model, depth=None, pre_message_str="",
        collapse_same_modules=True, with_frozen_symbal=True,
        logger=None
):
    if logger is None:
        logger = get_root_logger()


    frozen_symbal = '*'
    active_symbal = '~'
    no_param_symbal = '·'
    def get_frozen_symbal(is_frozen):
        if with_frozen_symbal:
            return frozen_symbal if is_frozen else active_symbal
        else:
            return no_param_symbal
    def get_frozen_str(is_frozen, str):
        if with_frozen_symbal:
            return CP.frozen(str) if is_frozen else CP.active(str)
        else:
            return str

    ### [collect] collect all modules and their states
    # Get all parameters, both frozen and trainable
    all_params = {name: not param.requires_grad for name, param in model.named_parameters()}

    # e.g 'backbone.conv1.weight' -> ['backbone', 'conv1', 'weight']
    param_hierarchy = {}
    for name, is_frozen in all_params.items():
        parts = name.split('.')
        for i in range(len(parts)):
            module_path = '.'.join(parts[:i+1])  # e.g. 'backbone.conv1'
            param_hierarchy[module_path] = is_frozen

    # print hierarchical modules with all parameters
    logs = []
    def collect_layer(module, prefix="", current_depth=0):
        if depth is not None and current_depth >= depth:
            return

        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            class_name = child.__class__.__name__

            # set no_param_symbal if this module has no parameters
            has_any_param = any(True for _ in child.parameters(recurse=True))
            if not has_any_param:
                module = dict(
                    level=full_name.count('.'),
                    class_name=class_name,
                    symbal=no_param_symbal,
                    collapse_count=1,
                )
            # set frozen or active symbal
            elif full_name in param_hierarchy:
                module = dict(
                    level=full_name.count('.'),
                    class_name=class_name,
                    is_all_frozen=param_hierarchy[full_name],
                    collapse_count=1,
                )
            # unknown state, mark with '?'
            else:
                # generally, are repeated modules
                module = dict(
                    level=full_name.count('.'),
                    class_name=class_name,
                    symbal='?',
                    collapse_count=1,
                )
                continue

            # add to logs
            level = module['level']
            class_name = module['class_name']
            if module.get('symbal', None) is not None:
                symbal = module['symbal']
                class_name = class_name
            elif module.get('is_all_frozen', None) is not None:
                is_all_frozen = module['is_all_frozen']
                symbal = get_frozen_symbal(is_all_frozen)
                class_name = get_frozen_str(is_all_frozen, class_name)
            else:
                raise ValueError(f"Unknown state for module {full_name}")
            logs.append(dict(
                message=f"{'  ' * level}{symbal} {class_name}",
                level=level,
            ))

            # recursively collect sub-modules
            if has_any_param:
                collect_layer(child, full_name, current_depth + 1)

    collect_layer(model)


    ### [collapse] collapse same modules
    def insert_after_first_non_space_space(text, s):
        # find first non-space character
        first_non_space = -1
        for i, char in enumerate(text):
            if char != ' ':
                first_non_space = i
                break

        if first_non_space == -1:
            return text  # if text is all spaces, return s

        # find the first space after first_non_space
        insert_pos = -1
        for i in range(first_non_space, len(text)):
            if text[i] == ' ':
                insert_pos = i
                break

        if insert_pos == -1:
            return text + ' ' + s  # if no space found, append s after text

        # insert s after the first space
        return text[:insert_pos + 1] + s + text[insert_pos + 1:]
    def collapse_modules(logs):
        n = len(logs)
        if n == 0:
            return []

        # Find all possible repeating patterns and their positions
        best_len = 1
        best_count = 1
        best_sub = logs[0]
        best_pos = 0

        for i in range(n):
            for l in range(1, (n - i) // 2 + 1):
                sub_modules = logs[i:i + l]
                # only sub_modules or same-level-same-name modules can be collapsed
                if any(
                        sub_module['level'] < sub_modules[0]['level']
                        or (sub_module['level'] == sub_modules[0]['level']
                            and sub_module['message'] != sub_modules[0]['message'])
                        for sub_module in sub_modules
                ):
                    continue

                count = 1
                j = i + l
                while j + l <= n and logs[j:j + l] == sub_modules:
                    count += 1
                    j += l

                # greater l first, then count
                if count <= 1:
                    continue
                if (count * l > best_count * best_len
                        or (count * l == best_count * best_len and count > best_count)
                ):
                    best_count = count
                    best_len = l
                    best_sub = sub_modules
                    best_pos = i

        if best_count == 1:
            return logs

        # Process the prefix
        prefix = logs[:best_pos]
        # Process the middle repeating part
        middle = collapse_modules(best_sub)
        middle[0]['message'] = insert_after_first_non_space_space(middle[0]['message'], f"(×{best_count}) ")
        # Process the suffix
        suffix = logs[best_pos + best_count * best_len:]

        # Recursively process the prefix and suffix
        compressed_prefix = collapse_modules(prefix)
        compressed_suffix = collapse_modules(suffix)

        return compressed_prefix + middle + compressed_suffix

    logs = collapse_modules(logs) if collapse_same_modules else logs
    logs = [log['message'] for log in logs]

    header = (f"Model parameter status ("
              f"{CP.frozen(f'{frozen_symbal}:frozen')}, "
              f"{CP.active(f'{active_symbal}:trainable')}, "
              f"{no_param_symbal}:w/o params)")
    log = '\n'.join(logs)
    logger.info(f"{pre_message_str}"
                f"{header}:"
                f"\n{log}\n"
                f"{header}")  # repeat header for better readability (log lines may be extra long)


def check_gradients(model, sample_input, device='cuda'):
    output = model(sample_input.to(device))

    fake_grad = torch.rand_like(output)
    output.backward(fake_grad)

    for name, param in model.named_parameters():
        if param.grad is None:
            logger.warning(f"Gradient for parameter {name} is None!")
        else:
            grad_mean = param.grad.mean().item()
            ratio = (param.grad / (param.data + 1e-8)).abs().mean()
            logger.info(
                f"{name} Gradient stats: "
                f" - gradient: {grad_mean:.6f}"
                f" - grad/param ratio: {ratio:.2e}")

    model.zero_grad()
#endregion


#region ==[Profile]==
def profile_test(
        model, input_size=(2, 3, 256, 256), device='cuda', logger=None, warpup_step=3,
        *model_args, **model_kwargs
):
    import os

    if logger is None:
        logger = get_root_logger()

    with get_model(model) as test_model:
        test_model.eval()
        test_model.to(device)

        test_tensor = torch.randn(input_size).to(device)

        for _ in range(warpup_step):
            with synchronize(device):
                output = test_model(test_tensor, *model_args, **model_kwargs)

        with torch.autograd.profiler.profile(enabled=True, use_cuda=device != 'cpu', record_shapes=False, profile_memory=True) as prof:
            output = test_model(test_tensor, *model_args, **model_kwargs)
        logger.info(prof.table())

        os.makedirs("./.log/model_trace", exist_ok=True)
        prof.export_chrome_trace(f"./.log/model_trace/{model.__class__.__name__}_{get_striped_time_str()}.json")

def profile_train(
        model, input_size=(2, 3, 256, 256), device='cuda', logger=None, step=3,
        *model_args, **model_kwargs
):
    from torch.profiler import profile, record_function

    if logger is None:
        logger = get_root_logger()

    with get_model(model) as train_model:
        train_model.to(device)

        test_tensor = torch.randn(input_size).to(device)

        with profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],              # 分析 CPU 和 CUDA 活动
                schedule=torch.profiler.schedule(
                    wait=1,     # 前1步不采样
                    warmup=1,   # 第2步作为热身，不计入结果
                    active=3,   # 采集后面3步的性能数据
                    repeat=2),  # 重复2轮
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./.log/model_trace/tensorboard/{model.__class__.__name__}_{get_striped_time_str()}'),  # 保存日志以供 TensorBoard 可视化
                record_shapes=True,     # 记录输入张量的形状
                profile_memory=True,    # 分析内存分配
                with_stack=True         # 记录操作的调用堆栈信息
            ) as profiler:
            for step in range(step):
                with record_function("model_forward"):
                    outputs = train_model(test_tensor)
                with record_function("model_backward"):
                    loss = outputs.sum()
                    loss.backward()

                profiler.step()  # 更新 profiler 的步骤

                try:
                    train_model.reset_memory()
                finally:
                    pass
    profiler.export_chrome_trace(f"./.log/model_trace/{model.__class__.__name__}_{get_striped_time_str()}.json")

# TODO: 使用 torch-scan 进行性能分析
#endregion


class ReduceOp(enum.Enum):
    MEAN = enum.auto()
    SUM = enum.auto()
    MAX = enum.auto()
    MIN = enum.auto()
    LAST = enum.auto()
    FIRST = enum.auto()


class Stats:
    def __init__(self, name):
        if 'pandas' not in sys.modules:
            self.stats = StatsList(name)
        else:
            self.stats = StatsDataframe(name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def append(self, header, **kwargs):
        """
        Append a set of measurements to the dataframe.

        Args:
            header (str):
            **kwargs: The measurements to append.
            e.g. stats.append(latency=0.1, flops=100, params=1000000)
        """
        self.stats.append(header, **kwargs)

    def summary(self, reduce_op=ReduceOp.MEAN):
        """
        Format the measurements in the dataframe as a string.

        Args:
            reduce_op (ReduceOp): The reduce operation to apply to the measurements.

        Returns:
            str: The formatted measurements.
        """
        measurements = self.reduce(reduce_op)

        out_strs = []
        for key, value in measurements.items():
            if isinstance(value, float):
                value = auto_number_to_str(value)
                out_strs.append(f"{CP.keyword(key)}: {CP.number(value)}")
            else:
                out_strs.append(f"{key}: {value}")
        out_str = ", ".join(out_strs)
        return out_str

    def summary_dict(self, reduce_op=ReduceOp.MEAN):
        """
        Get the measurements without `header` and `name` as a dictionary.

        Args:
            reduce_op (ReduceOp): The reduce operation to apply to the measurements.

        Returns:
            dict: The formatted measurements.
        """
        out_dict = {}
        measurements = self.reduce(reduce_op)
        for key, value in measurements.items():
            if key in ["name", "header"]:
                continue
            out_dict[key] = value
        return out_dict

    def reduce(self, reduce_op=ReduceOp.MEAN):
        """
        Reduce the measurements in the dataframe.

        Args:
            reduce_op (ReduceOp): The reduce operation to apply to the measurements.

        Returns:
            OrderedDict: The reduced measurements.
        """
        return self.stats.reduce(reduce_op)


class StatsList:
    def __init__(self, name):
        self.name = name
        self.measurements = []

    def append(self, header, **kwargs):
        """
        Append a set of measurements to the dataframe.

        Args:
            header (str): The name of the file to append.
            **kwargs: The measurements to append.
            e.g. stats.append(latency=0.1, flops=100, params=1000000)
        """
        measurements = OrderedDict(name=self.name, header=header)
        for key, value in kwargs.items():
            measurements[key] = value
        self.measurements.append(measurements)

    def reduce(self, reduce_op=ReduceOp.MEAN):
        if self.measurements is None:
            return None

        # Group measurements by key
        grouped_data = defaultdict(list)
        for measurement in self.measurements:
            for key, value in measurement.items():
                if key != "name" and key != "header":
                    grouped_data[key].append(value)

        reduced_measurements = OrderedDict()
        for key, values in grouped_data.items():
            if reduce_op == ReduceOp.MEAN:
                reduced_measurements[key] = sum(values) / len(values)
            elif reduce_op == ReduceOp.SUM:
                reduced_measurements[key] = sum(values)
            elif reduce_op == ReduceOp.MAX:
                reduced_measurements[key] = max(values)
            elif reduce_op == ReduceOp.MIN:
                reduced_measurements[key] = min(values)
            elif reduce_op == ReduceOp.LAST:
                reduced_measurements[key] = values[-1]
            elif reduce_op == ReduceOp.FIRST:
                reduced_measurements[key] = values[0]
            else:
                raise ValueError(f"Invalid reduce_op: {reduce_op}")

        reduced_measurements["name"] = self.name
        return reduced_measurements


class StatsDataframe:
    def __init__(self, name):
        self.name = name
        self.measurements = []

    def append(self, header, **kwargs):
        """
        Append a set of measurements to the dataframe.

        Args:
            header (str): The name of the file to append.
            **kwargs: The measurements to append.
            e.g. stats.append(latency=0.1, flops=100, params=1000000)
        """
        measurements = OrderedDict(name=self.name, header=header)
        measurements.update(kwargs)
        self.measurements.append(measurements)

    def reduce(self, reduce_op=ReduceOp.MEAN):
        if self.measurements is None:
            return None

        measurements = pd.DataFrame(self.measurements).drop(columns=["name", "header"])
        if reduce_op == ReduceOp.MEAN:
            reduced_measurements = measurements.mean(numeric_only=True)
        elif reduce_op == ReduceOp.SUM:
            reduced_measurements = measurements.sum(numeric_only=True)
        elif reduce_op == ReduceOp.MAX:
            reduced_measurements = measurements.max(numeric_only=True)
        elif reduce_op == ReduceOp.MIN:
            reduced_measurements = measurements.min(numeric_only=True)
        elif reduce_op == ReduceOp.LAST:
            reduced_measurements = measurements.iloc[-1]
        elif reduce_op == ReduceOp.FIRST:
            reduced_measurements = measurements.iloc[0]
        else:
            raise ValueError(f"Invalid reduce_op: {reduce_op}")

        reduced_measurements["name"] = self.name
        return reduced_measurements


@contextmanager
def synchronize(device='cuda'):
    if 'cuda' in device:
        torch.cuda.synchronize()
        try:
            yield
        finally:
            torch.cuda.synchronize()
    else:
        yield


