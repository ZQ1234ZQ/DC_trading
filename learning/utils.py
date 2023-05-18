from functools import wraps
from time import time

from rich.console import Console

console = Console()


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        console.log(f"func:{f.__name__} took: {te-ts:2.4f} sec")
        return result

    return wrap
