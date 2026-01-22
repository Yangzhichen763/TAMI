import contextlib


__all__ = [
    'mem_engaged',
    'no_mem_engaged',
    'mem_engaged_if',
    'is_mem_engaged',
]


mem_engaged_flag = False


@contextlib.contextmanager
def mem_engaged():
    global mem_engaged_flag
    last_flag = mem_engaged_flag
    mem_engaged_flag = True
    yield
    mem_engaged_flag = last_flag


@contextlib.contextmanager
def no_mem_engaged():
    global mem_engaged_flag
    last_flag = mem_engaged_flag
    mem_engaged_flag = False
    yield
    mem_engaged_flag = last_flag


@contextlib.contextmanager
def mem_engaged_if(condition):
    global mem_engaged_flag
    last_flag = mem_engaged_flag
    mem_engaged_flag = condition
    yield
    mem_engaged_flag = last_flag


def is_mem_engaged():
    return mem_engaged_flag