from contextlib import contextmanager
import time


@contextmanager
def timer(name):
    """
    Time Each Process
    """
    t0 = time.time()
    yield
    print('[{}] done in {} s'.format(name, round(time.time() - t0, 0)))
