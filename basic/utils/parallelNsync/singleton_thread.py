
import functools
import queue
import threading
import time
import os

"""
在训练过程中开启多线程，会减慢训练速度（因为是并发而不是并行）
可以选择在训练过程中保存，并使用另一个进程翻译保存结果
"""

try:
    from basic.utils.console.log import get_root_logger
    logger = get_root_logger(force_set_info=True)
    def print(*args, **kwargs):
        logger.info(*args, **kwargs)
except ImportError:
    from builtins import print as original_print
    def print(*args, **kwargs):
        original_print(*args, **kwargs)

wait_duration = 0.5
auto_shut_duration = 20 # 时间越久，线程重新创建的周期越长，主进程结束时，线程结束的也越慢
max_workers = 1

_workers = {}
_dead_workers = {}
_tasks = queue.PriorityQueue()
_num_tasks = 0
_num_current_tasks = 0

_current_tick = time.time()


def _any_shut():
    global _current_tick
    if time.time() - _current_tick > auto_shut_duration:
        return True
    return False


def _update_tick():
    global _current_tick
    _current_tick = time.time()


def _tick(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _update_tick()
        result = func(*args, **kwargs)
        _update_tick()
        return result
    return wrapper


@_tick
def _add_task(func, *args, priority=0, **kwargs):
    item = (priority, time.time(), func, args, kwargs)
    _tasks.put(item)

    global _num_current_tasks, _num_tasks
    _num_current_tasks += 1
    _num_tasks += 1

    # print(f"Task added. Total tasks: {_num_tasks}, Current tasks: {_num_current_tasks}")


@_tick
def _run_tasks():
    _, _, func, args, kwargs = _tasks.get(block=False)
    global _num_current_tasks
    try:
        # print(f"running {_num_current_tasks} / {_num_tasks}")
        func(*args, **kwargs)
    finally:
        _num_current_tasks -= 1
        _tasks.task_done()

        # print(f"{_num_current_tasks} / {_num_tasks}")


def _run_tasks_worker(id: int):
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')

    # print(f"Parallel Thread {id} started.")
    while not (_any_shut() and _tasks.empty()):
        if _tasks.empty():
            time.sleep(wait_duration)
            continue

        try:
            _run_tasks()
        except queue.Empty:
            time.sleep(wait_duration)
        # print(time.time() - _current_tick, _current_tick)

    # 停止当前线程，并不使得主进程停止
    _dead_workers[id] = _workers[id]
    # print(f"Parallel Thread {id} stopped.")


def _create_workers():
    global _workers

    # 如果出现现成缺少，创建补足
    for i in range(max_workers):
        if i in _workers:
            continue

        consume_worker = threading.Thread(target=_run_tasks_worker, kwargs=dict(id=i))
        consume_worker.start()

        _workers[i] = consume_worker

    # Don't wait for thread join to prevent blocking the main process
    # for i in _workers:
    #     _workers[i].join()


def check_and_run():
    global _workers, _dead_workers

    for id, worker in _dead_workers.items():
        worker.join()
        # print(f"Worker {id} deleted.")
        del _workers[id]
    _dead_workers = {}

    if len(_workers) < max_workers:
        _update_tick()
        _create_workers()


def thread_parallel(priority=0):
    """
    注意！如果装饰的函数有返回值，则返回值会被忽略

    Args:
        priority (int, optional): Task priority. Defaults to 0.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            check_and_run()
            _add_task(func, *args, **kwargs, priority=priority)
        return wrapper
    return decorator

