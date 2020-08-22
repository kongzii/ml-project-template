import time
import functools

SEED_VALUE = 0


def timing(f):
    @functools.wraps(f)
    def wrap(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        print(f"Function: {f.__name__} args:[{args}, {kwargs}] took: {time.time() - ts} seconds.")
        return result
    return wrap
