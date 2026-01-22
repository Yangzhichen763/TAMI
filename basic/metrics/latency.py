import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def record(self):
        self.end_time = time.time()

    def reset(self):
        self.start_time = None
        self.end_time = None

    def restart(self):
        self.reset()
        self.start()

    def elapsed(self):
        return self.end_time - self.start_time

    def __str__(self):
        return f"{self.elapsed():.4f}"

    def __format__(self, format_spec):
        return format(self.elapsed(), format_spec)

    def __float__(self):
        return self.elapsed()


@contextmanager
def timer():
    """
    Context manager to measure the execution time of a block of code.
    """
    timer = Timer()
    timer.start()
    try:
        timer.record()
        yield timer # Execute the code block inside the with statement
    finally:
        timer.record()


if __name__ == "__main__":
    with timer() as t:
        time.sleep(1)
    print(f"Inference time: {t:.4f} seconds")