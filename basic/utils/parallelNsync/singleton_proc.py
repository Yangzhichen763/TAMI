import functools
import importlib
import pickle
import time
import os
from multiprocessing import Process, Value, Manager

wait_duration = 0.5
auto_shut_duration = 120  # 时间越久，进程重新创建的周期越长，主进程结束时，子进程结束的也越慢
max_workers = 4  # os.cpu_count()

class PriorityQueue:
    def __init__(self, manager):
        self._queue = manager.list()
        self._lock = manager.Lock()

    def put(self, item):
        with self._lock:
            self._queue.append(item)
            # Sort by priority then by timestamp
            self._queue.sort(key=lambda x: (x[0], x[1]))

    def get(self, block=True):
        with self._lock:
            if not block and not self._queue:
                raise Exception("Queue empty")
            return self._queue.pop(0) if self._queue else None

    def empty(self):
        with self._lock:
            return len(self._queue) == 0

# Global variables
_manager = None
_tasks = None
_current_tick = None
_tick_lock = None
_workers = None       # Manager dict, 只存 PID
_local_workers = {}   # 本地字典, 存 Process 对象（主进程用）

def check_and_init():
    global _manager, _tasks, _current_tick, _tick_lock, _workers

    if _manager is None:
        _manager = Manager()

    if _workers is None:
        _workers = _manager.dict()

    if _tasks is None:
        _tasks = PriorityQueue(_manager)

    if _current_tick is None:
        _current_tick = _manager.Value('d', time.time())

    if _tick_lock is None:
        _tick_lock = _manager.Lock()

def _init(tasks, current_tick, tick_lock, workers, *args, **kwargs):
    from multiprocessing.managers import BaseManager
    global _manager
    address = kwargs.pop('address', None)
    authkey = kwargs.pop('authkey', None)
    _manager = BaseManager(address=address, authkey=authkey)
    _manager.connect()

    global _tasks, _current_tick, _tick_lock, _workers
    _tasks = tasks
    _current_tick = current_tick
    _tick_lock = tick_lock
    _workers = workers

def _any_shut():
    with _tick_lock:
        if time.time() - _current_tick.value > auto_shut_duration:
            return True
    return False

def _update_tick():
    with _tick_lock:
        _current_tick.value = time.time()

def _tick(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _update_tick()
        result = func(*args, **kwargs)
        _update_tick()
        return result
    return wrapper

def _is_pickleable(obj):
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False

@_tick
def _add_task(func, *args, priority=0, **kwargs):
    module_name = func.__module__
    func_name = func.__name__

    # 确保所有参数可序列化
    safe_args = []
    for a in args:
        if not _is_pickleable(a):
            raise ValueError(f"Argument {a} is not pickleable. Move it out or convert it.")
        safe_args.append(a)

    safe_kwargs = {}
    for k, v in kwargs.items():
        if not _is_pickleable(v):
            raise ValueError(f"Kwarg {k}={v} is not pickleable. Move it out or convert it.")
        safe_kwargs[k] = v

    item = (priority, time.time(), module_name, func_name, safe_args, safe_kwargs)
    _tasks.put(item)

@_tick
def _run_tasks():
    try:
        item = _tasks.get(block=False)
        if item:
            _, _, module_name, func_name, args, kwargs = item
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            func(*args, **kwargs)
    except Exception as e:
        print(f"Error in _run_tasks: {e}")
        time.sleep(wait_duration)

def _run_tasks_worker(id: int, **kwargs):
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')

    _init(**kwargs) # 如果设置 start strategy 为 spawn，需要传入共享变量

    while not _any_shut() or not _tasks.empty():
        if _tasks.empty():
            time.sleep(wait_duration)
            continue

        try:
            _run_tasks()
        except Exception:
            time.sleep(wait_duration)


def _create_workers():
    global _workers, _local_workers

    # 获取 manager 的服务器地址和密钥
    address = _manager._address
    authkey = _manager._authkey

    # 确保 manager 启动完成
    time.sleep(0.1)

    for i in range(max_workers):
        if i in _local_workers and _local_workers[i].is_alive():
            continue

        p = Process(target=_run_tasks_worker, kwargs=dict(
            id=i,
            address=address, authkey=authkey,
            tasks=_tasks, current_tick=_current_tick, tick_lock=_tick_lock, workers=_workers
        ))
        p.daemon = False
        p.start()

        _local_workers[i] = p      # 本地存进程对象
        _workers[i] = p.pid        # Manager dict 只存 PID（可共享）

def check_and_run():
    global _workers

    if len(_workers) < max_workers or any(
        pid is None or not _local_workers.get(i, None) or not _local_workers[i].is_alive()
        for i, pid in _workers.items()
    ):
        _update_tick()
        _create_workers()

def process_parallel(priority=0):
    """
    进程版并行装饰器
    注意！如果装饰的函数有返回值，则返回值会被忽略

    Args:
        priority (int, optional): Task priority. Defaults to 0.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            check_and_init()
            check_and_run()
            _add_task(func, *args, **kwargs, priority=priority)
        return wrapper
    return decorator