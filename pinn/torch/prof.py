import time
from contextlib import contextmanager

@contextmanager
def walltime(name: str, enabled: bool = True):
    """
    Simple wall-clock timing context manager.

    Prints elapsed milliseconds on exit. Designed for low-overhead coarse timing.
    """
    if not enabled:
        yield
        return
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    print(f"[TIMER] {name}: {(t1 - t0) * 1000:.2f} ms")
