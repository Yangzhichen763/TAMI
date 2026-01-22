import os
import queue
import time

'''
多线程计算
'''


def multi_thread_process(
        iterable: list, func, on_result,
        multi_thread=True, num_threads=None
):
    if multi_thread:
        # Multi-thread processing
        from concurrent.futures import ThreadPoolExecutor, as_completed

        length = len(iterable)
        max_workers = num_threads or min(os.cpu_count(), length)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for i in iterable:
                future = pool.submit(func, i)
                futures[future] = i

            for future in as_completed(futures):
                result = future.result()
                on_result(futures[future], result)
    else:
        # Single-thread processing
        for i in iterable:
            result = func(i)
            on_result(i, result)


#region ==[Producers and Consumers]==
class Signal:
    def __init__(self):
        self.flag = False

    def turn(self, flag=True):
        self.flag = flag

    def is_true(self):
        return self.flag

    def is_false(self):
        return not self.flag


def _produce_worker(produce_func, resource_queue, signal):
    for resource in produce_func():
        resource_queue.put(resource)

    signal.turn(True)


def _consume_worker(consume_func, resource_queue, signal):
    # mp.set_sharing_strategy('file_system') 是多进程有用的，在多线程中没有效果
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')

    while signal.is_false() or not resource_queue.empty():
        if resource_queue.empty():
            time.sleep(0.1)
            continue

        try:
            resource = resource_queue.get(block=False)
            consume_func(resource)
            resource_queue.task_done()
        except queue.Empty:
            time.sleep(0.1)


def queue_thread_process(
        produce_func, consume_func,
        multi_thread=True, num_threads=None
):
    """
    When the rate of production and consumption are unusually uncoordinated,
    using queue and multi-thread processing can improve the efficiency of processing.

    e.g.
    ```python
    import time
    import random
    from pbar import pbar_context

    total = 100

    with pbar_context(total=total) as gen_pbar:
        with pbar_context(total=total) as con_pbar:
            def produce_func():
                for i in range(total):
                    rand = random.random() * 0.1
                    time.sleep(rand)

                    gen_pbar.update()
                    yield i

            def consume_func(i):
                rand = random.random()
                time.sleep(rand)

                con_pbar.update()

            queue_thread_process(produce_func, consume_func, multi_thread=True, num_threads=4)
    ```
    """
    if multi_thread:
        # Multi-thread processing
        import threading
        resource_queue = queue.Queue()
        signal = Signal()
        max_workers = num_threads or os.cpu_count()

        # [produce worker] 生产者线程
        produce_worker = threading.Thread(target=_produce_worker, args=(produce_func, resource_queue, signal))
        produce_worker.start()

        # [consume workers] 消费者线程
        consume_workers = []
        for i in range(max_workers):
            consume_worker = threading.Thread(target=_consume_worker, args=(consume_func, resource_queue, signal))
            consume_worker.start()
            consume_workers.append(consume_worker)

        # [wait for completion] 等待所有线程完成
        produce_worker.join()
        for worker in consume_workers:
            worker.join()
    else:
        # Single-thread processing
        for resource in produce_func():
            consume_func(resource)
#endregion


#region ==[Threading Context]==
import threading

class ThreadWithContext:
    def __init__(self, target, args=()):
        self.target = target
        self.args = args
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self.target, args=self.args)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.thread.join()
#endregion