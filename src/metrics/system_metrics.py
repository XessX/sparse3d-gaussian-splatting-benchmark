import time
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    result = {"elapsed_sec": None}
    try:
        yield result
    finally:
        result["elapsed_sec"] = time.perf_counter() - start
